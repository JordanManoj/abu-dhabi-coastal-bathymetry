"""Turn raw ATL24 seafloor photons into tide-free depth samples on the Sentinel-2 10 m grid.

Steps (each one is a decision recorded in DECISIONS.md):
 1. Quality filter: ensemble confidence >= 0.6 (low_confidence_flag == 0) and
    sensor_depth_exceeded == 0.
 2. Tide-free depth: ATL24 'depth' (surface_h - ortho_h) is the INSTANTANEOUS depth, which
    contains the tide at the moment of the pass (pass medians range -0.67 .. +1.28 m here,
    while the median depth is only ~1.5 m). The seafloor height ortho_h (EGM08) does not
    depend on the tide, so we use  depth = MSL - ortho_h, with MSL = the median sea-surface
    height (EGM08) over all passes = local mean sea level in the geoid frame.
 3. Pixel aggregation: photons -> 10 m Sentinel-2 pixels (UTM 40N, same grid as the
    imagery), separately per pass. Keep a pixel if it has >= MIN_PHOTONS photons and a
    robust spread (MAD) <= MAX_MAD m. Median depth is the label. This removes isolated
    false-positive photons (e.g. the 'deep' photons that fall on the mainland).
Usage:  python src/prepare_atl24.py
Output: data/interim/atl24_pixels.parquet, data/interim/atl24_summary.json
"""
import glob
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from common import INTERIM, RAW, S2_CRS, aoi_utm_bounds

MIN_CONF = 0.6
MIN_PHOTONS = 3
MAX_MAD = 0.5   # m; robust spread of photon depths inside one 10 m pixel
MAX_DEPTH = 30  # m; the Gulf AOI never exceeds ~25 m in ATL24 here, deeper = noise


def mad(x):
    return float(np.median(np.abs(x - np.median(x))))


def main():
    df = pd.concat([gpd.read_parquet(f) for f in sorted(glob.glob(str(RAW / "atl24" / "atl24_*.parquet")))])
    n_raw = len(df)
    df = df[(df.low_confidence_flag == 0) & (df.confidence >= MIN_CONF) & (df.sensor_depth_exceeded == 0)]
    n_conf = len(df)

    df["pass_id"] = df.rgt.astype(str) + "_" + df.cycle.astype(str)
    df["date"] = df.index.strftime("%Y-%m-%d")
    pass_surface = df.groupby("pass_id").surface_h.median()
    msl = float(pass_surface.median())
    df["depth_inst"] = df.surface_h - df.ortho_h        # what ATL24 calls depth (contains tide)
    df["depth"] = msl - df.ortho_h                        # tide-free depth below local MSL
    df["tide"] = df.pass_id.map(pass_surface) - msl       # water level of the pass relative to MSL

    g = df.to_crs(S2_CRS)
    x0, y0, x1, y1 = aoi_utm_bounds()
    g["col"] = ((g.geometry.x - x0) // 10).astype(int)
    g["row"] = ((y1 - g.geometry.y) // 10).astype(int)

    agg = g.groupby(["pass_id", "row", "col"]).agg(
        rgt=("rgt", "first"), cycle=("cycle", "first"), gt=("gt", "first"), date=("date", "first"),
        night=("night_flag", "first"), tide=("tide", "first"),
        depth=("depth", "median"), depth_inst=("depth_inst", "median"),
        n=("depth", "size"), mad=("depth", mad), conf=("confidence", "median"),
    ).reset_index()
    n_pix_all = len(agg)
    agg = agg[(agg.n >= MIN_PHOTONS) & (agg.mad <= MAX_MAD) & (agg.depth > 0) & (agg.depth <= MAX_DEPTH)]
    agg["x"] = x0 + (agg.col + 0.5) * 10
    agg["y"] = y1 - (agg.row + 0.5) * 10
    INTERIM.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(INTERIM / "atl24_pixels.parquet", index=False)

    summary = dict(
        photons_raw=n_raw, photons_after_confidence=n_conf, msl_egm08_m=round(msl, 3),
        pass_tide_min_m=round(float(pass_surface.min() - msl), 2), pass_tide_max_m=round(float(pass_surface.max() - msl), 2),
        passes=int(agg.pass_id.nunique()), rgts=int(agg.rgt.nunique()),
        date_first=agg.date.min(), date_last=agg.date.max(),
        pixel_samples_before_filter=n_pix_all, pixel_samples=len(agg), unique_pixels=int(agg[["row", "col"]].drop_duplicates().shape[0]),
        depth_min=round(float(agg.depth.min()), 2), depth_median=round(float(agg.depth.median()), 2),
        depth_p95=round(float(agg.depth.quantile(0.95)), 2), depth_max=round(float(agg.depth.max()), 2),
        samples_by_depth_bin={str(k): int(v) for k, v in pd.cut(agg.depth, [0, 2, 5, 10, 15, 20, 30]).value_counts().sort_index().items()},
        samples_by_rgt={str(k): int(v) for k, v in agg.rgt.value_counts().sort_index().items()},
    )
    (INTERIM / "atl24_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
