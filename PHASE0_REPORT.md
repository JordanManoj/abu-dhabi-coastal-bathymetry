# Phase 0 report: data-access checks

Date: 24 October 2026. Goal: confirm, before building anything, that the data needed for
satellite-derived bathymetry over Abu Dhabi exists and is usable.

**Verdict: GO.** ATL24 coverage is good (not thin), so there is no need for the fallback
coastline. Sentinel-2 has hundreds of clear scenes. Two risks found early and designed for:
**tide** in the ICESat-2 depths, and **interleaving tracks** (spatial leakage).

## 1. ICESat-2 ATL24 over the AOI (24.3–24.7°N, 54.2–54.8°E)

Search: NASA CMR, collection ATL24 v002 (`C4085168939-NSIDC_CPRD`).
- **132 granules** intersect the AOI (2018-10-19 → 2024), ~16.6 GB as full HDF5 files.
- Access used: **SlideRule Earth** `atl24x`, subset to the AOI on the server, no login
  (`src/download_atl24.py`). 2025–2026 are not released yet in v002 (empty response).

| Item | Value | Source |
|---|---|---|
| Seafloor (class 40) photons | 862,202 | `src/download_atl24.py` |
| After confidence ≥ 0.6 and sensor-depth flag | 784,213 | `data/interim/atl24_summary.json` |
| Passes (tracks = RGT + cycle) | 102 raw, 95 after filtering | same |
| Reference ground tracks (RGTs) | 7 (5 carry > 99.9 % of samples) | same |
| Dates | 2018-10-19 → 2024-11-05 | same |
| 10 m pixel samples (≥ 3 photons, MAD ≤ 0.5 m) | 97,730 (88,894 unique pixels) | same |
| Depth below local MSL | min 0.02, median 1.48, p95 8.21, max 28.5 m | same |
| Samples per depth bin | 0–2 m: 60,524 · 2–5: 24,945 · 5–10: 10,001 · 10–15: 1,983 · 15–20: 216 · 20–30: 61 | same |
| Night passes | 67 % of photons (lower solar noise) | raw files |
| Vertical uncertainty `sigma_tvu` | median 0.17 m | raw files |

Map: `results/figures/phase0_atl24_tracks.png`.

**Findings that changed the design:**
1. **Tide.** Water level at the passes spans −1.21 to +0.81 m around MSL, but the median depth is
   1.5 m. ATL24's depth is instantaneous, so I label with the tide-free seabed height referenced
   to local MSL (MSL = median sea surface = +0.461 m EGM08). Repeat passes over the same pixel
   now agree to a median of 0.10 m instead of 0.34 m (`results/tide_consistency.json`).
2. **Track geometry.** Only 7 RGTs, but due to off-pointing the cycles of one RGT are spread
   across ~30 km and interleave with other RGTs. Splitting by track alone can still leave
   neighbours in training, so I add a spatial buffer and measure leakage explicitly.
3. **False positives.** A few "deep" photons were classified on the mainland (~54.7°E, 24.3°N).
   They are removed by pixel aggregation (≥ 3 photons, low spread) and the water mask.
4. **Shallow-heavy labels.** 87 % of samples are < 5 m and only 277 are > 15 m, so the 10–20 m
   bin will be the weakest and its numbers must be read with that in mind.

## 2. Sentinel-2 L2A

- Collection-1 reprocessed archive on Earth Search (`sentinel-2-c1-l2a`), tile **40RBN**, which
  fully contains the AOI. **470 scenes** with < 5 % cloud between 2019 and 2026 (no account).
- The older Earth Search collection mixes processing baselines (PB 02.14 water reflectance
  ~5× PB 05.13 in blue) and has an offset trap. Collection-1 is consistent (all PB 05.xx).
- Every scene was scored on 80 m overviews for cloud, glint (B11 over water) and turbidity (red
  over water): `data/interim/scene_scores.csv`.
  Within ATL24's years (2019–2024): **308 scenes scored**, 90 meet the selection rule (normal
  coverage, < 1 % cloud in the covered part). Every 40RBN scene has ≥ 10 % no-data at the swath
  edge, so 'full coverage' could not be required. The 10 best (lowest glint + turbidity) are all
  winter scenes (Nov–Mar) and were downloaded (`src/download_s2.py`, 1.3 GB).

## 3. Backup validation data

- **GMRT GridServer** subset (GEBCO-based in this area), `data/raw/gebco/gmrt_abu_dhabi.tif`.
  Delivered at ~55 m spacing, but the underlying GEBCO data here is 15 arc-second (~460 m) and
  largely interpolated. Usable only as a regional sanity check, not for pixel validation.
- Official charts and multibeam surveys of Abu Dhabi are not open.

## 4. Decision

Proceed with ATL24 as ground truth over the brief's AOI. No fallback coastline needed.
