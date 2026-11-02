"""Download a backup reference grid (GMRT, which uses GEBCO in this area) for the AOI.

GMRT GridServer returns a GeoTIFF subset with no login. In the shallow Gulf the
underlying data is the GEBCO compilation (15 arc-second, ~460 m cells), so it is far
too coarse to validate 10 m SDB pixel by pixel; we only use it as a regional sanity
check (are deep/shallow zones in the right place, and in the right depth range?).
Usage:  python src/download_gebco.py
Output: data/raw/gebco/gmrt_abu_dhabi.tif  (elevation, m; negative = below sea level)
"""
import requests

from common import RAW, aoi_lonlat_bounds

URL = "https://www.gmrt.org/services/GridServer"


def main():
    w, s, e, n = aoi_lonlat_bounds()
    r = requests.get(URL, params=dict(minlongitude=w, maxlongitude=e, minlatitude=s, maxlatitude=n,
                                      format="geotiff", resolution="max", layer="topo"), timeout=300)
    r.raise_for_status()
    out = RAW / "gebco" / "gmrt_abu_dhabi.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    print("saved", out, len(r.content) // 1024, "kB")


if __name__ == "__main__":
    main()
