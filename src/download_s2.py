"""Download the AOI window of the selected Sentinel-2 Collection-1 scenes.

Bands: B01 coastal (60 m), B02 blue, B03 green, B04 red, B08 NIR (10 m), B11 SWIR-1 (20 m)
and the SCL scene classification (20 m). Coarser bands are resampled to the 10 m grid
with nearest neighbour (keeps the original values; no invented detail).
Values are stored as surface reflectance * 10000 (uint16, offset already applied, 0 = nodata),
so the rest of the pipeline never has to know about processing baselines.

Usage:  python src/download_s2.py [--k 10]
Output: data/raw/s2/<scene_id>/<band>.tif
"""
import argparse

import numpy as np
import pandas as pd
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from common import INTERIM, RAW, S2_COLLECTION, S2_CRS, STAC_URL, aoi_utm_bounds

BANDS = {"B01": "coastal", "B02": "blue", "B03": "green", "B04": "red", "B08": "nir", "B11": "swir16", "SCL": "scl"}


def pick_scenes(k):
    """Best k scenes: normal coverage, clear, then lowest glint + turbidity (rank sum).

    Restricted to 2019-2024, the years covered by ATL24 v002, so the seabed seen by the
    camera and by the laser is as close in time as possible (reclamation and dredging
    change the Abu Dhabi coast).
    Every 40RBN scene misses the western edge of the AOI (>= 10 % no-data at the tile/swath
    boundary), so 'full coverage' is impossible: we require the usual coverage
    (no-data <= 15 %) and < 1 % cloud/shadow inside the covered part.
    """
    s = pd.read_csv(INTERIM / "scene_scores.csv")
    s = s[(s.date < "2025-01-01") & (s.nodata_frac <= 0.15) & (s.bad_frac - s.nodata_frac < 0.01)].copy()
    s["rank"] = s.glint_b11.rank() + s.turb_red_p75.rank()
    return s.sort_values("rank").head(k)


def read_band(asset, bounds, shape):
    with rasterio.open(asset.href) as src:
        win = from_bounds(*bounds, transform=src.transform)
        a = src.read(1, window=win, out_shape=shape, resampling=Resampling.nearest, boundless=True, fill_value=0)
    if asset.extra_fields.get("roles") and "reflectance" not in asset.extra_fields["roles"]:
        return a  # SCL: class codes, no scaling
    rb = asset.to_dict()["raster:bands"][0]
    refl = a.astype("float32") * rb.get("scale", 1.0) + rb.get("offset", 0.0)
    out = np.clip(np.round(refl * 10000), 1, 65535).astype("uint16")
    out[a == 0] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    x0, y0, x1, y1 = aoi_utm_bounds()
    shape = (int((y1 - y0) / 10), int((x1 - x0) / 10))
    transform = from_origin(x0, y1, 10, 10)
    chosen = pick_scenes(args.k)
    chosen.to_csv(INTERIM / "chosen_scenes.csv", index=False)
    print(chosen[["id", "date", "glint_b11", "turb_red_p75", "bad_frac"]].to_string(index=False))
    client = Client.open(STAC_URL)
    for sid in chosen.id:
        item = next(client.search(collections=[S2_COLLECTION], ids=[sid]).items())
        d = RAW / "s2" / sid
        d.mkdir(parents=True, exist_ok=True)
        for band, key in BANDS.items():
            out = d / f"{band}.tif"
            if out.exists():
                continue
            a = read_band(item.assets[key], (x0, y0, x1, y1), shape)
            tmp = out.with_suffix(".part.tif")  # atomic: a killed run never leaves a half file
            with rasterio.open(tmp, "w", driver="GTiff", height=shape[0], width=shape[1], count=1, dtype=a.dtype,
                               crs=S2_CRS, transform=transform, nodata=0, compress="deflate", tiled=True) as dst:
                dst.write(a, 1)
            tmp.replace(out)
        print("downloaded", sid, flush=True)


if __name__ == "__main__":
    main()
