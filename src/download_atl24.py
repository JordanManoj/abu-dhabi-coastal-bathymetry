"""Download ICESat-2 ATL24 (v002) seafloor photons over the AOI via SlideRule Earth.

SlideRule subsets the ATL24 granules server-side to our polygon, so we download
a few MB of points instead of ~16.6 GB of full HDF5 granules (132 granules cross
the AOI). The public SlideRule cluster needs no NASA Earthdata login.

Only class 40 (bathymetry) photons are requested: each one already carries the
sea-surface height (surface_h) used for depth, so sea-surface photons are not needed
(asking for every class returned tens of millions of photons and filled the RAM).
The request is split by year so that one failed year can be retried on its own.

Usage:  python src/download_atl24.py
Output: data/raw/atl24/atl24_<year>.parquet
"""
import json
import time
from pathlib import Path

import sliderule

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "atl24"


def load_aoi():
    gj = json.loads((ROOT / "aoi" / "abu_dhabi_coast.geojson").read_text())
    ring = gj["features"][0]["geometry"]["coordinates"][0]
    return [{"lon": x, "lat": y} for x, y in ring]


def main():
    sliderule.init(verbose=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in range(2018, 2027):
        out = OUT_DIR / f"atl24_{year}.parquet"
        if out.exists():
            print(year, "already downloaded")
            continue
        parms = {
            "t0": f"{year}-01-01T00:00:00Z",
            "t1": f"{year + 1}-01-01T00:00:00Z",
            # compact=False keeps the quality flags (low_confidence_flag, invalid_kd, ...)
            # and uncertainties (sigma_tvu, sigma_thu); confidence_threshold 0 = keep all,
            # we choose our own threshold later, visibly, in prepare_atl24.py
            "atl24": {"class_ph": ["bathymetry"], "compact": False, "confidence_threshold": 0.0},
        }
        t = time.time()
        gdf = sliderule.run("atl24x", parms, aoi=load_aoi())
        if gdf is None or len(gdf) == 0:
            # ATL24 v002 is released a few times per year; recent years can be empty
            print(f"{year}: no ATL24 granules released yet")
            continue
        print(f"{year}: {len(gdf):,} photons in {time.time() - t:.0f} s")
        gdf.to_parquet(out)


if __name__ == "__main__":
    main()
