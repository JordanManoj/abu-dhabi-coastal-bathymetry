"""Shared paths, AOI geometry and constants for the Abu Dhabi bathymetry pipeline."""
import json
import os
from pathlib import Path

from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
FAILS = RESULTS / "failures"
AOI_FILE = ROOT / "aoi" / "abu_dhabi_coast.geojson"

# Sentinel-2 MGRS tile 40RBN (UTM zone 40N) covers the whole AOI, so we stay in its
# native grid (EPSG:32640, 10 m) and never resample the reflectance.
S2_TILE = "40RBN"
S2_CRS = "EPSG:32640"
S2_COLLECTION = "sentinel-2-c1-l2a"  # ESA Collection-1: whole archive reprocessed with baseline 05.xx
STAC_URL = "https://earth-search.aws.element84.com/v1"

# GDAL settings for fast, anonymous reads of cloud-optimised GeoTIFFs on AWS
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
)
os.environ.update(GDAL_ENV)


def aoi_lonlat_bounds():
    gj = json.loads(AOI_FILE.read_text())
    xs, ys = zip(*gj["features"][0]["geometry"]["coordinates"][0])
    return min(xs), min(ys), max(xs), max(ys)


def aoi_utm_bounds():
    """AOI bounding box in UTM 40N, snapped outward to the 10 m Sentinel-2 grid."""
    w, s, e, n = aoi_lonlat_bounds()
    t = Transformer.from_crs("EPSG:4326", S2_CRS, always_xy=True)
    xs, ys = zip(*[t.transform(x, y) for x in (w, e) for y in (s, n)])
    snap = lambda v, f: f(v / 10) * 10
    import math
    return snap(min(xs), math.floor), snap(min(ys), math.floor), snap(max(xs), math.ceil), snap(max(ys), math.ceil)
