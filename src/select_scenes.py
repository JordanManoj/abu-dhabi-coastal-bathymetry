"""Rank Sentinel-2 L2A (Collection-1) scenes of tile 40RBN for bathymetry, using cheap COG overviews.

Why: SDB needs scenes that are clear (no cloud), calm (little sun glint) and clean
(low turbidity). The catalogue cloud percentage alone does not measure glint or
turbidity, so every candidate is scored on 80 m overviews (~1 MB per scene):
  - cloud/shadow/no-data fraction inside the AOI (SCL classes 0,1,3,8,9,10)
  - glint: median SWIR-1 (B11) reflectance over water. Water absorbs SWIR almost
    completely, so any B11 signal over water is surface reflection (glint) or haze.
  - turbidity: median red (B04) over water deeper than the shallow flats is not known
    yet, so we use the 75th percentile of B04 over water as a sediment proxy.
Usage:  python src/select_scenes.py
Output: data/interim/scene_scores.csv
"""
import numpy as np
import pandas as pd
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from common import INTERIM, S2_COLLECTION, S2_TILE, STAC_URL, aoi_utm_bounds

OVERVIEW_LEVEL = 2  # COG overview index: 10 m * 2**(level+1) = 80 m for 10 m bands


def read_ov(href, bounds, level=OVERVIEW_LEVEL):
    with rasterio.open(href, overview_level=level) as src:
        win = from_bounds(*bounds, transform=src.transform)
        return src.read(1, window=win, boundless=True, fill_value=0)


def to_reflectance(dn, asset):
    """DN -> surface reflectance using the asset's own scale/offset (raster extension).

    Collection-1 items declare scale 1e-4 and offset -0.1 (the processing-baseline
    04.00+ BOA_ADD_OFFSET). Reading them from metadata instead of hard-coding avoids
    the double-offset bug we hit with the older 'sentinel-2-l2a' collection, whose
    DNs were already offset-corrected.
    """
    rb = asset.to_dict()["raster:bands"][0]
    r = dn.astype("float32") * rb.get("scale", 1.0) + rb.get("offset", 0.0)
    r[dn == 0] = np.nan
    return r


def score(item, bounds):
    a = item.assets
    b11 = read_ov(a["swir16"].href, bounds, OVERVIEW_LEVEL - 1)  # 20 m band: one level less
    shape = b11.shape
    scl = read_ov(a["scl"].href, bounds, OVERVIEW_LEVEL - 1)
    green = read_ov(a["green"].href, bounds)[: shape[0], : shape[1]]
    red = read_ov(a["red"].href, bounds)[: shape[0], : shape[1]]
    b11, green, red = (to_reflectance(x, a[k]) for x, k in ((b11, "swir16"), (green, "green"), (red, "red")))
    bad = np.isin(scl, [0, 1, 3, 8, 9, 10])
    # water: SCL water class OR MNDWI>0.2 (SCL misses very shallow bright water)
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (green - b11) / (green + b11)
    water = ((scl == 6) | (mndwi > 0.2)) & ~bad & np.isfinite(green)
    if water.sum() < 1000:
        return None
    return dict(
        id=item.id,
        date=item.properties["datetime"][:10],
        cloud_cover=item.properties["eo:cloud_cover"],
        sun_elev=item.properties.get("view:sun_elevation"),
        bad_frac=float(bad.mean()),
        nodata_frac=float((scl == 0).mean()),
        water_frac=float(water.mean()),
        glint_b11=float(np.nanmedian(b11[water])),
        turb_red_p75=float(np.nanpercentile(red[water], 75)),
        blue_p50=float(np.nanmedian(to_reflectance(read_ov(a['blue'].href, bounds)[: shape[0], : shape[1]], a['blue'])[water])),
    )


def main():
    w, s, e, n = aoi_utm_bounds()
    items = list(Client.open(STAC_URL).search(
        collections=[S2_COLLECTION], datetime="2019-01-01/2024-12-31",  # ATL24 v002 years only
        query={"eo:cloud_cover": {"lt": 5}, "grid:code": {"eq": f"MGRS-{S2_TILE}"}},
        max_items=2000).items())
    # one item per date (Earth Search sometimes has reprocessed duplicates)
    items = list({i.properties["datetime"][:10]: i for i in items}.values())
    print(f"{len(items)} candidate scenes with <5% cloud")
    # Resumable: every scored scene is appended to a partial file immediately, so an
    # interrupted run (the first one was killed by low system memory) continues where it stopped.
    INTERIM.mkdir(parents=True, exist_ok=True)
    partial = INTERIM / "scene_scores_partial.csv"
    done = set(pd.read_csv(partial).id) if partial.exists() else set()
    todo = [i for i in sorted(items, key=lambda i: i.properties["datetime"]) if i.id not in done]
    print(f"{len(done)} already scored, {len(todo)} to go", flush=True)
    for k, it in enumerate(todo):
        try:
            r = score(it, (w, s, e, n))
        except Exception as ex:  # a broken COG should not stop the ranking
            print("skip", it.id, ex, flush=True)
            continue
        if r:
            pd.DataFrame([r]).to_csv(partial, mode="a", header=not partial.exists(), index=False)
        if k % 25 == 0:
            print(k, it.id, r and {x: round(r[x], 4) for x in ("bad_frac", "glint_b11", "turb_red_p75")}, flush=True)
    df = pd.read_csv(partial)
    df.to_csv(INTERIM / "scene_scores.csv", index=False)
    print(df.describe().round(4))


if __name__ == "__main__":
    main()
