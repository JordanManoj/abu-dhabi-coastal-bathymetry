# Data: how to get and prepare it

Raw and intermediate data are **not** in git (see `.gitignore`). Everything is downloaded by
scripts, with no accounts or API keys needed. Run from the repo root with the venv active.

| Step | Command | What it does | Output | Size / time |
|---|---|---|---|---|
| 1 | `python src/download_atl24.py` | ICESat-2 ATL24 v002 seafloor photons (class 40) inside `aoi/abu_dhabi_coast.geojson`, 2018–2026, subset server-side by SlideRule Earth | `data/raw/atl24/atl24_<year>.parquet` | ~45 MB, ~2 min |
| 2 | `python src/prepare_atl24.py` | Confidence filter, tide-free depth below local MSL, aggregation to the Sentinel-2 10 m grid | `data/interim/atl24_pixels.parquet`, `atl24_summary.json` | seconds |
| 3 | `python src/check_tide_consistency.py` | Repeat-pass agreement: tide-free vs instantaneous depth | `results/tide_consistency.json` | ~1 min |
| 4 | `python src/select_scenes.py` | Scores every Sentinel-2 Collection-1 L2A scene of tile 40RBN (< 5 % cloud) for cloud, glint (B11) and turbidity (red) on 80 m overviews | `data/interim/scene_scores.csv` | ~45 min (network) |
| 5 | `python src/download_s2.py --k 10` | Downloads the AOI window of the 10 best scenes (2019–2024): B01, B02, B03, B04, B08, B11, SCL | `data/raw/s2/<scene>/<band>.tif` | ~1–2 GB |
| 6 | `python src/build_composite.py` | Per-scene cloud mask + SWIR glint removal, multi-date median, MNDWI water index; also the single best scene | `data/processed/composite.tif`, `single.tif` | ~5 min |
| 7 | `python src/extract_samples.py --image composite` | Joins labels and features at the same pixel | `data/processed/samples_composite.parquet` | seconds |
| 8 | `python src/download_gebco.py` | Backup regional reference (GMRT, GEBCO-based here, ~460 m native) | `data/raw/gebco/gmrt_abu_dhabi.tif` | 3.4 MB |

## Sources and licences
- **ICESat-2 ATL24 v002** (NASA / NSIDC), doi:10.5067/ATLAS/ATL24.002. Public domain (NASA).
  Accessed through SlideRule Earth (https://slideruleearth.io), which needs no login.
  Alternative: `earthaccess` + NASA Earthdata login, download the 132 full granules (~16.6 GB).
- **Sentinel-2 L2A Collection-1** (ESA Copernicus), served as cloud-optimised GeoTIFFs by
  Element 84 Earth Search (`sentinel-2-c1-l2a`). Copernicus open licence.
- **GMRT / GEBCO**: open, attribution required (GMRT: Ryan et al. 2009; GEBCO Compilation Group).

## Grid and units
- All rasters and labels are on the **UTM 40N (EPSG:32640) 10 m grid** of tile 40RBN, clipped to
  the AOI bounding box (x 215790–277430, y 2689200–2734630; 4,543 × 6,164 pixels).
- Reflectance is surface reflectance (unitless, 0–1); the downloaded tiles store it × 10000 as
  uint16 with the processing-baseline offset already applied.
- **Depth** = metres below local mean sea level (positive down). Local MSL = +0.461 m above the
  EGM08 geoid, estimated from the median ATL24 sea surface over all passes.
