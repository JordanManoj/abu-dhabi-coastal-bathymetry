# Research notes: Abu Dhabi Coastal Bathymetry (Satellite-Derived Bathymetry)

Research done before building each component. For every technique, dataset and metric:
what it is, why it fits, alternatives, sources, and pitfalls. Where a number is quoted it
comes from the cited source; where it comes from this repo, the script is named.

Contents
0. Decision matrix
0b. How I searched
1. Physics of light in water
2. Sentinel-2 L2A (and the Collection-1 reprocessing)
3. ICESat-2, ATL03 and ATL24
4. Refraction correction
5. Tides and vertical datums
6. Sun glint and its correction
7. Land/water masking, optically deep and turbid water
8. Stumpf (2003) log-ratio method
9. Lyzenga (1978/1985) linear-transform method
10. Machine-learning SDB: Random Forest and XGBoost (and why not deep learning)
11. Spatial leakage in geospatial ML and how to split
12. Metrics
13. Typical depth limits of optical SDB
14. Backup reference data (GEBCO / GMRT)
15. Related work and how my approach differs
16. Source list
17. Where each research finding shows up in the results

---

## 0. Decision matrix (every component)

Status: **Used** · **Tested and failed** (with why) · **Rejected from research only** (never run here).
Evidence points to `results/` or a decision ID in `DECISIONS.md`.

| Component | Chosen | Alternatives considered | Why chosen | Why not the alternatives | Status / evidence |
|---|---|---|---|---|---|
| Area of interest | 24.3–24.7°N, 54.2–54.8°E (brief) | larger Gulf box; single lagoon | 0–20 m water incl. banks, lagoons, channels and shelf; inside one S2 tile (40RBN) | larger = more tiles and water types; a lagoon = too little depth range | Used (D01) |
| Imagery / sensor | Sentinel-2 L2A, 10 m | Landsat 8/9 (30 m); PlanetScope (3 m); WorldView | free, ~5-day revisit, blue/green/coastal bands, standard for SDB | Landsat coarser; Planet/WorldView commercial | Used |
| Imagery archive | Collection-1 (`sentinel-2-c1-l2a`, Earth Search) | older `sentinel-2-l2a`; Copernicus Data Space | one processing baseline (05.xx) for 2019–2024; no account | old collection mixed baselines (blue water ~5x different) and had an offset trap | Older collection tested and failed: negative reflectance (D07, D08) |
| Atmospheric correction | Sen2Cor (standard L2A) | ACOLITE; C2RCC | best of the three for Abu Dhabi SDB in Gafoor et al. 2022 | worse in that study | Used; ACOLITE/C2RCC rejected from research only |
| Labels / ground truth | ICESat-2 ATL24 v002 | own ATL03 classification; boat survey; nautical charts | NASA's refraction-corrected, ensemble-classified seafloor product; open | ATL03 classification = weeks of work; surveys/charts not open | Used: 862,202 photons (D02) |
| Label access route | SlideRule Earth `atl24x` (server-side subset) | earthaccess + 132 full granules (16.6 GB, login) | 45 MB, no login, named in the ATL24 user guide | ~300x more data, credentials | Used (D02, D03) |
| Label datum | tide-free depth below local MSL (MSL − ortho_h) | ATL24 instantaneous depth; tide model (FES2014/TPXO) | seabed height is tide-independent; MSL estimated from 95 passes | instantaneous depth contains ±1 m tide; tide models weak in lagoons | Instantaneous depth tested and failed: repeat-pass disagreement 0.34 vs 0.10 m (`results/tide_consistency.json`, D05) |
| Label quality filter | confidence ≥ 0.6, sensor-depth flag 0 | none; ≥ 0.8 | NSIDC's own low-confidence threshold | stricter removes mostly the rare deep returns | Used (D04) |
| Label aggregation | median per 10 m pixel per pass, ≥ 3 photons, MAD ≤ 0.5 m | every photon a sample; 20/30 m cells | same grid as features; removes isolated false photons | photon samples = near-duplicates; coarser cells lose detail | Used: 97,730 samples (D06) |
| Scene selection | score 308 scenes (cloud, B11 glint, red turbidity) on overviews; 10 best | catalogue cloud % only; all clear scenes | glint and turbidity matter as much as cloud | cloud % ignores glint/turbidity | Used; all 10 picks are winter scenes (D10, D11) |
| Glint correction | R − R_B11 (SWIR-1) | Hedley 2005 NIR regression; none | SWIR absorbed in ~1.5 mm: pure surface signal | NIR sees bright sand under < 0.5 m water, so it would remove depth signal | Used; Hedley rejected from research (physics) (D12) |
| Compositing | per-pixel median of 10 corrected scenes | best single scene; mean | robust to residual glint, haze, boats, tide of one date | single scene noisier | Single scene tested and failed: RF 1.085 vs 0.996 m, Stumpf 2.44 vs 1.93 m (`results/summary.json`, D13) |
| Water mask | MNDWI > 0, ≥ 3 scenes, connected body ≥ 0.1 km² | NDWI (NIR); SCL water class; MNDWI alone | SWIR-based index works over bright shallow water | NDWI fails there; MNDWI alone = ~60,000 sabkha speckles | MNDWI alone tested and failed (D14) |
| Features | 5 bands, 3 Stumpf ratios, 4 Lyzenga terms; no coordinates | raw bands only; + lat/lon | physics-informed; coordinates memorise neighbourhoods | bands-only worse; lat/lon = leakage | Bands-only tested: 1.044 vs 0.990 m (`results/metrics_primary.csv`, D23); lat/lon rejected from research |
| Log-ratio validity | blue, green > 0.001 for all models | clip to a floor; keep all | the method's definition (ln(1000R) > 0) | without it Stumpf R² ≈ 0 (ratios −16 … +19) | Without rule tested and failed (D16) |
| Lyzenga R∞ | 5th percentile of water reflectance | 0.5th percentile; manual deep polygon | robust; AOI has no optically deep water | 0.5th gave negative R∞ | 0.5th tested and failed (D15) |
| Baseline | Stumpf log-ratio (+ switching + Lyzenga) | none | industry standard; ML must beat it | — | Used: RMSE 2.04 / 1.74 / 1.73 m (`results/metrics_primary.csv`) |
| Switching baseline | red ratio < 2 m, green > 4 m, red only where red > 0.01 | red wherever defined | our data: red ratio saturates at ~5 m | without red > 0.01: bias −0.82 m | First version tested and failed (D17) |
| Model | Random Forest (final), XGBoost | CNN/U-Net; SVR; MLP; geospatial foundation model | 12-feature tabular problem, 96k samples; standard in SDB | CNNs need dense labels (ICESat-2 = lines); foundation models land-trained, overkill | RF/XGB used; CNN/FM/SVR/MLP rejected from research only (D18, D22) |
| Pre-training / reuse | reuse ATL24 ensemble (CNN + XGBoost + 4 methods) and Sen2Cor; trees trained from scratch | ImageNet / Prithvi backbones | no pre-trained S2 → Gulf depth model exists; trees train in minutes | backbones need dense patches; different task | Used; backbones rejected from research only |
| Hyper-parameters | fixed a priori (RF 300 trees, leaf 5, 50 % features; XGB 800, lr 0.03, depth 6) | grid / random search | no tuning on test folds; RF insensitive | proper tuning needs nested spatial CV (time) | Used; tuning not done (limitation) |
| Sample weighting | none | depth-balanced inverse-frequency weights | best shallow accuracy; deep errors conservative | balanced: 0–5 m RMSE 0.65 → 0.75, bias turns positive (unsafe) | Balanced tested and failed (`results/metrics_balanced.csv`, D21) |
| Split | leave-one-RGT-out + 1 km buffer | random K-fold; by pass; by pass + 500 m | strict and keeps ~54k training samples per fold | random/pass leak (0.71/0.79 m); pass + 500 m deletes 2/3 of data | All tested (`results/metrics_composite_*.csv`, D19) |
| Buffer distance | 1 km | 0–2,000 m | RF error plateaus at ~1 km | larger = data loss, extrapolation | Tested (`results/buffer_sensitivity.csv`, `variogram.csv`) |
| Metrics | RMSE, MAE, bias, R², by depth bin | single overall RMSE; % error only | comparable with literature; bias = safety; bins expose depth limit | overall number hides deep failure | Used |
| Backup validation | GMRT/GEBCO grid (regional) | GEBCO portal; charts | open API | charts not open | Used: r = 0.91 (`results/figures/gebco_check.png`, D25) |
| Domain transfer | train and test in the target region | transfer a model trained elsewhere | ICESat-2 gives local labels, so no transfer needed | — | Used; transfer to other Gulf coasts untested (limitation) |
| Compute | laptop CPU, 16 GB RAM, steps in separate processes | GPU / cloud | trees need no GPU; RAM was the constraint | — | Used; resumable scripts after a memory kill (D10) |

## 0b. How I searched

- **Primary documents first:** the NSIDC ATL24 v002 user guide (field definitions, classifier
  ensemble, limitations), the SlideRule Earth client source (`atl24x` endpoint), the Earth
  Search STAC catalogue (asset metadata: scale/offset, processing baseline).
- **Papers:** Google Scholar / publisher pages for the classic SDB methods (Lyzenga, Stumpf),
  the ICESat-2 bathymetry papers (Parrish 2019, Thomas 2021, Magruder 2025), ML SDB
  (Sagawa 2019, Gafoor 2022, Chybicki 2023), glint (Hedley 2005, Kay 2009) and spatial
  validation (Roberts 2017, Ploton 2020).
- **Local study:** Gafoor et al. (2022) from Khalifa University is the closest prior work:
  Sentinel-2 SDB in Abu Dhabi. I read it fully because it tells me what fails here.
- **Reference tables:** pure-water absorption from Pope & Fry (1997) and Hale & Querry (1973),
  taken from the OMLC data files.

---

## 1. Physics of light in water

**What it is.** Sunlight goes through the water, reflects off the seabed and comes back.
On both legs water absorbs it. The deeper the water, the less bottom-reflected light reaches
the satellite. Absorption depends strongly on wavelength, which is what makes depth
measurable: if two bands are absorbed at different rates, their *ratio* changes with depth.

**Numbers (pure water, absorption a(λ), penetration 1/a):**

| Band (Sentinel-2) | λ | a (m⁻¹) | 1/a (m) | Source |
|---|---|---|---|---|
| B01 coastal | 443 nm | 0.0070 | ~144 | Pope & Fry 1997 (442.5 nm) |
| B02 blue | 490 nm | 0.015 | ~67 | Pope & Fry 1997 |
| B03 green | 560 nm | 0.062 | ~16 | Pope & Fry 1997 |
| B04 red | 665 nm | 0.43 | ~2.3 | Pope & Fry 1997 |
| B08 NIR | ~842 nm | ~4.3 (850 nm) | ~0.23 | Hale & Querry 1973 |
| B11 SWIR-1 | 1610 nm | ~672 (1600 nm) | ~0.0015 (1.5 mm) | Hale & Querry 1973 |

The light travels down *and* up, so the useful depth is roughly half of 1/a, and real
seawater absorbs and scatters more than pure water (phytoplankton, dissolved organic matter,
suspended sediment). The overall attenuation is summarised by the diffuse attenuation
coefficient **Kd**.

**What this means for the project.**
- Blue and green carry depth information over 0–20+ m in clear water: they are the core of
  every SDB method.
- Red only "sees" the bottom in the first few metres. That makes it useful in *very* shallow
  water, where blue/green are nearly saturated (Caballero & Stumpf 2020's switching model uses a
  red ratio there). Most of my labels are < 2 m, so I keep red.
- NIR is nearly gone after ~0.2 m, but over bright sand in < 0.5 m it still contains bottom
  signal. This matters for glint correction (Section 6).
- SWIR-1 is absorbed within millimetres: any SWIR signal over water is surface reflection or
  haze. That makes it the cleanest glint estimate.

**Turbidity.** Suspended sediment scatters light back before it reaches the bottom. The
water then looks brighter and "shallower" than it is. The ratio methods partially cancel
this, but not completely. Gafoor et al. (2022) found that chlorophyll-a, sediment and dust
did *not* significantly affect their Abu Dhabi estimates, while tide did.

**Pitfalls.** Bottom type changes reflectance as much as depth does: dark seagrass at 2 m
can look like sand at 8 m. Ratio methods were invented to reduce exactly this problem.

**Outcome in this project.** Confirmed in the data: the blue/green ratio rises 0.944 → 1.10 from 0–1 m to
15–20 m; blue/red rises 1.09 → 1.80 from 0 to 5 m, then saturates. Past ~10 m the signal fades: all 6
worst error areas are 9–11 m deep (`results/failures/failure_cases.csv`).

---

## 2. Sentinel-2 L2A and Collection-1

**What it is.** ESA's twin (now triple) optical satellites. 13 bands, 10 m for B02/B03/B04/B08,
20 m for red-edge/SWIR, 60 m for B01. Revisit ~5 days. L2A = surface reflectance after Sen2Cor
atmospheric correction, with a Scene Classification Layer (SCL: cloud, shadow, water, ...).

**Why it suits the problem.** Free, frequent, 10 m pixels, and its blue/green bands sit where
water is most transparent. It is the standard sensor for regional SDB (Thomas et al. 2021,
Caballero & Stumpf 2019, Gafoor et al. 2022).

**Alternatives.** Landsat 8/9 (30 m, 16 days; Sagawa 2019 used it), PlanetScope (3 m, commercial,
less radiometrically stable), WorldView (2 m, commercial, 8 bands incl. coastal/yellow).

**Atmospheric correction choice.** Gafoor et al. (2022) compared Sen2Cor, ACOLITE and C2RCC
for Abu Dhabi SDB and found **Sen2Cor best** (R² 0.7 vs 0.6 for ACOLITE and −0.2 for C2RCC in
their initial comparison). This is the main reason I used the standard L2A product instead of
running ACOLITE myself.

**Pitfalls I found and fixed.**
- **Processing baseline offset.** Since baseline 04.00 (Jan 2022) L2A values carry an offset:
  reflectance = DN/10000 − 0.1. Earth Search's older `sentinel-2-l2a` collection has already
  removed that offset for some items (`earthsearch:boa_offset_applied`); applying it again gave
  negative reflectance. Fix: read scale/offset from each asset's metadata (DECISIONS D08).
- **Mixed baselines.** In that older collection, a 2021 scene (PB 02.14) had blue water
  reflectance ~5× a 2026 scene (PB 05.13). ESA's **Collection-1** reprocessing puts the whole
  archive on baseline 05.xx; Earth Search serves it as `sentinel-2-c1-l2a` (DECISIONS D07).

Sources: ESA Sentinel-2 user handbook / SentiWiki; Earth Search STAC (element84); Gafoor et
al. 2022.

**Outcome in this project.** 308 Collection-1 scenes scored, 10 winter scenes composited. The composite
beats the best single scene for every model (RF 0.996 vs 1.085 m on 92,493 common pixels).

---

## 3. ICESat-2, ATL03 and ATL24

**What it is.** ICESat-2 (launched 2018) carries ATLAS, a photon-counting green (532 nm) laser
altimeter. It fires 10,000 pulses per second split into 6 beams (3 strong, 3 weak) in three
pairs ~3.3 km apart; beams in a pair are ~90 m apart. Footprints are ~11 m and ~0.7 m apart along
track. Green light penetrates water, so in clear shallow water some photons come back from
the seabed.
- **ATL03**: every detected photon with latitude, longitude, height and time. Bathymetry has to
  be extracted by the user (noise, sea surface, seabed classification + refraction correction).
- **ATL24** (v001 April 2025, v002 May 2026): NASA's official coastal and nearshore bathymetry
  product. Each photon is classified (0 unclassified, 1 other, **40 bathymetry**, 41 sea
  surface) by a **stacking ensemble** of several classifiers (CoastNet CNN, BathyPathFinder graph
  method, OpenOceans histogram/GMM, Median Filter, Quantile Trees using XGBoost, C-SHELPh), with
  blunder detection and, in v002, an extra density-based step. Heights are
  **refraction-corrected**, given as ellipsoid height and orthometric height (EGM08), with the
  sea-surface height and uncertainties (`sigma_tvu`, `sigma_thu`) per photon.

**Key fields I use (ATL24 v002 user guide):**
- `ortho_h`: refraction-corrected orthometric (EGM08) height of the photon
- `surface_h`: orthometric height of the sea surface at the photon
- `confidence` (0–1, ensemble) and `low_confidence_flag` (= confidence < 0.6)
- `sensor_depth_exceeded`: 1 if the depth is beyond the sensor's capability
- `night_flag`: night passes have less solar background noise

**Why it suits the problem.** It is free, global, accurate (decimetre-level vertical
uncertainty; `sigma_tvu` median 0.17 m in my AOI) and already classified. It gives labels where
no open survey exists, which is the whole case for "fully space-borne" SDB (Thomas et al.
2021). Using ATL24 instead of classifying ATL03 myself is also a **reuse** decision: NASA's
classifier is an ensemble of several published models trained on hundreds of labelled
tracks; I could not match it in a week.

**Access route.** 132 ATL24 v002 granules (~16.6 GB) cross my AOI, almost all of each granule
outside it. The ATL24 user guide points to **SlideRule Earth** for on-demand subsetting; its
public cluster needs no login. I downloaded only class-40 photons inside the AOI
(`src/download_atl24.py`).

**What my AOI has (from `src/prepare_atl24.py`).** 862,202 bathymetry photons, 2018-10-19 to
2024-11-05, 102 passes (95 after filtering) from **7 reference ground tracks (RGTs)**, 5 of
which carry almost all the data. Depth 0–28.5 m, median 1.5 m, 95th percentile 8.2 m.

**Pitfalls.**
- Labels are sparse lines, not a surface; the model must interpolate between them.
- Bathymetry is rare compared with surface photons ("very high class imbalance", user guide);
  false positives exist. I saw "deep" photons on the mainland near 54.7°E.
- Because of off-pointing, repeat passes of one RGT are spread ~30 km apart in my AOI, and
  different RGTs interleave. That has consequences for splitting (Section 11).
- Depth is *instantaneous* (contains tide): see Section 5.

**Outcome in this project.** 862,202 → 784,213 photons after the confidence filter → 97,730 pixel labels →
96,200 evaluation samples in 0–20 m. Some false seafloor photons remain as label noise in the failure cases.

Sources: NSIDC ATL24 v002 user guide (2026); Magruder et al. 2025 (Earth and Space Science
12(10), doi:10.1029/2025EA004390); ATL24 ATBD (Magruder et al. 2026, doi:10.5067/UIN5PPVS10WW);
Parrish et al. 2019.

---

## 4. Refraction correction

**What it is.** When the laser enters the water, two things change: light slows down
(refractive index n ≈ 1.34, so speed ≈ c/1.34) and its direction bends (Snell's law). An
altimeter converts travel time into distance using the speed of light in air, so the
uncorrected seabed appears roughly **25 % too deep** (1 − 1/1.34 ≈ 0.25), with a small
horizontal shift as well. ICESat-2 points close to nadir, so the vertical part dominates.

**How ATL24 handles it.** ATL24 applies the correction itself, using a **global 0.25°
refractive-index layer built from temperature and salinity** (Dietrich & Parrish 2025,
cited in the user guide). This matters in the Gulf, where water is warm and very saline, so a
single n = 1.33 would be slightly wrong.

**Decision.** I use ATL24's corrected `ortho_h` directly and do **not** re-apply a
correction (double correction is a classic mistake when people mix ATL03 recipes with ATL24).

**Outcome in this project.** Used as delivered; the code contains no separate refraction step.

Sources: Parrish et al. 2019 (Remote Sensing 11(14):1634, the first ICESat-2 bathymetry
validation, which established the refraction approach); ATL24 user guide; Dietrich & Parrish 2025.

---

## 5. Tides and vertical datums

**What it is.** Depth "below the water surface" changes with tide. Abu Dhabi's tide is
mixed (semi-diurnal/diurnal), with a range on the order of 1–2 m (Gafoor et al. 2022, Fig. 15
shows levels up to 1.7 m at the Sentinel-2 overpass).

**Why it matters here.** In my data the water level at the 95 passes ranges from
**−1.21 m to +0.81 m** around mean sea level, while the median depth is only **1.5 m**.
Using ATL24's instantaneous depth as the label would put most of the label variance into tide.
Gafoor et al. (2022) also found tide to be the main error source for Abu Dhabi SDB.

**Options.**
1. Use instantaneous depth (`surface_h − ortho_h`): simple, but tide-contaminated.
2. Correct each pass with a global tide model (FES2014, TPXO): adds a model and its coastal
   errors in a complex shallow lagoon system.
3. **Use the tide-free seabed height** `ortho_h` and reference it to the **local mean sea
   level** estimated as the median `surface_h` over all passes: `depth = MSL − ortho_h`.

**Decision: option 3.** The seabed does not move with the tide; 95 passes sample random tide
phases, so their median surface is a good estimate of MSL in the geoid frame. No external model.

**Evidence (`src/check_tide_consistency.py`).** For 10,561 pairs of passes over the same 10 m
pixel, the median disagreement drops from **0.34 m (instantaneous) to 0.10 m (tide-free)**,
RMS 0.61 → 0.43 m.

**Residual issue.** The Sentinel-2 images were also taken at some tide state. A multi-date
median composite averages it out partly. Remaining tide effects become noise in the model.

**Outcome in this project.** Repeat-pass disagreement 0.34 → 0.10 m. Residual: the per-pass error bias
correlates only weakly with the pass tide (r = +0.16; `results/summary.json`, per_pass_bias).

---

## 6. Sun glint and its correction

**What it is.** Direct sunlight reflected from wave facets into the sensor. It adds a bright,
patchy, nearly wavelength-independent signal that has nothing to do with depth.

**Methods.**
- **Hedley et al. (2005)** (revising Hochberg et al. 2003): regress each visible band on NIR
  over deep water and subtract `b_i × (NIR − NIR_min)`. The standard for high-res reef imagery.
- **SWIR-based** correction (used in ACOLITE and reviewed by Kay et al. 2009): same idea with
  a SWIR band.
- Avoid it: choose low-glint scenes and use a multi-date median.

**Decision.** (1) Rank scenes by median B11 over water (a glint/haze score) and use the best
ones, (2) subtract **B11** (SWIR-1) from each visible band, per scene, before compositing, and
(3) take a per-pixel median across scenes.

**Why SWIR and not NIR here.** From the absorption table, NIR penetrates ~0.2 m and SWIR-1
only ~1.5 mm. Abu Dhabi has huge areas of < 1 m bright sand, which is where most of my labels are.
There, NIR still contains bottom signal, so Hedley-NIR would subtract depth information. SWIR
is safely "black" over any water.

**Pitfall.** B11 is 20 m, so the correction is slightly coarser than the 10 m bands. It also
removes a little haze (fine: it is also not depth).

**Outcome in this project.** Water reflectance after correction is plausible (blue median 0.048);
2–4 % of mixed shoreline pixels go slightly negative and are handled by the validity rule.

Sources: Hedley, Harborne & Mumby 2005 (IJRS 26(10):2107–2112); Kay, Hedley & Lavender 2009
(Remote Sensing 1(4):697); Hale & Querry 1973.

---

## 7. Land/water masking, optically deep and turbid water

- **Land/water:** MNDWI = (Green − SWIR1)/(Green + SWIR1) (Xu 2006). It uses SWIR rather than
  NIR (NDWI, McFeeters 1996) for the same reason as above: NIR is contaminated by the bottom
  in very shallow water. Combined with the SCL cloud/shadow mask.
- **Clouds/shadow:** SCL classes 3 (shadow), 8–9 (cloud), 10 (cirrus), plus 0/1 (no-data,
  saturated), applied per scene before the median.
- **Optically deep water:** where no bottom signal returns, the ratio saturates and every
  method just outputs noise. Caballero & Stumpf and Viaña-Borja et al. (2023) remove these
  pixels. I flag predictions beyond the training range as unreliable in the map (no ATL24
  sample deeper than 28.5 m exists).

**Outcome in this project.** MNDWI alone gave ~60,000 inland speckles; the connected-body filter keeps 18
water bodies. 12.9 M water pixels mapped.
- **Turbid water:** scenes are ranked by the 75th percentile of red reflectance over water
  (sediment brightens red), and the clearest are composited.

---

## 8. Stumpf (2003) log-ratio method (the baseline)

**What it is.** Depth is linear in the ratio of log-reflectances:

    pSDB = ln(n·R_blue) / ln(n·R_green),     depth = m1 · pSDB − m0

with n a constant (commonly 1000) that keeps both logs positive. Only two parameters (m1, m0)
are fitted on known depths.

**Why it works.** Green is absorbed faster than blue, so as depth increases green drops
faster and the ratio rises. Because both bands see the same bottom, a change in bottom
brightness affects both logs similarly and the ratio stays more stable than either band
(the "variable bottom type" advantage over Lyzenga).

**Why it is my baseline.** It is the most widely used SDB method, physically motivated, needs
very little calibration, and is what NOAA and many hydrographic offices use (Caballero & Stumpf
2019, 2020). Thomas et al. (2021) used it with ICESat-2 calibration too. If ML cannot beat it,
ML is not worth it.

**Variants.** Blue/red ratio for very shallow water and a **switching model** that blends
the red and green ratio results by depth (Caballero & Stumpf 2020; used by Viaña-Borja et al.
2023). Gafoor et al. (2022) proposed a four-visible-band ratio for Abu Dhabi.

**Pitfalls.** Linear only over a limited depth range; noisy at depth; cannot resolve features
smaller than ~4–5 pixels beyond 15–20 m (Stumpf et al. 2003); needs a deep-water mask.

**Outcome in this project.** First run R² ≈ 0 (undefined log-ratios); after the validity rule RMSE 2.04 m,
R² 0.32 (held-out RGT + 1 km). Switching variant: 1.74 m overall, 0.87 m in 0–5 m.

Source: Stumpf, Holderied & Sinclair 2003, Limnology and Oceanography 48(1, part 2): 547–556.

---

## 9. Lyzenga (1978/1985) linear-transform method

**What it is.** Assumes exponential attenuation, so after removing the deep-water signal
R∞, `X_i = ln(R_i − R∞,i)` is linear in depth. Depth is a multiple linear regression on
the X_i of several bands. Lyzenga (1985) calibrated it with airborne lidar depths, which is
conceptually what I do with ICESat-2.

**Why it is relevant.** It is the other classical baseline. Thomas et al. (2021) found Lyzenga
had the lowest RMSE of their three ICESat-2-trained methods (≈10–15 % of depth).

**Pitfalls.** Needs an estimate of R∞ (optically deep water, which my AOI barely has); five or
more parameters; sensitive to bottom type and very low-albedo bottoms (Stumpf 2003).

**Use in this project.** Lyzenga-style features `ln(R_i − R∞,i)` are fed to the ML models,
and a multi-linear Lyzenga model is a second classical baseline.

**Outcome in this project.** RMSE 1.73 m, R² 0.51; the best linear method in 5–20 m (2.66 / 6.00 m), as in
Thomas et al. 2021.

Sources: Lyzenga 1978, Applied Optics 17(3):379–383; Lyzenga 1985, IJRS 6(1):115–125.

---

## 10. Machine-learning SDB

**Random Forest** (Breiman 2001): an average of many decision trees, each on a bootstrap
sample with random feature subsets. Handles non-linear band/depth relations and bottom-type
interactions, needs little tuning, and gives feature importances.

**XGBoost** (Chen & Guestrin 2016): gradient-boosted trees, each tree fitting the residual of
the previous ones, with regularisation. Usually a bit more accurate than RF on tabular data,
and fast on CPU.

**Evidence that they suit SDB.**
- Sagawa et al. 2019: RF + multi-temporal Landsat-8, RMSE 1.41 m over 0–20 m.
- Gafoor et al. 2022 (Abu Dhabi): gradient boosting better than linear regression on band ratios.
- An ISPRS 2024 study: RF on Sentinel-2 bands + Stumpf ratios, RMSE 1.25 m, R² 0.854, better
  than multivariate Stumpf (RMSE 2.20 m).
- Chybicki et al. 2023 (PLOS ONE): compared NNs, trees, RF and analytical models for Sentinel-2.
- A Lund University thesis used ATL24 as the target for Sentinel-2 RF/CatBoost/MLP models.

**Features I give them.** Glint-corrected reflectance of B01–B04 and B08, the Stumpf ratios
(blue/green, blue/red, coastal/green), and Lyzenga log-features. Explicitly **no latitude/longitude**:
with track-shaped labels, coordinates let the model memorise "this area is 3 m deep" and
it would not generalise to new tracks (a spatial-leakage mechanism, Section 11).

**Why not deep learning / pre-trained models (build vs reuse).**
- Each sample is one pixel with ~15 numbers. Tree ensembles are the strong default for small
  tabular data; there is no pre-trained network for "Sentinel-2 pixel → Gulf depth".
- CNNs/U-Nets use spatial context and can be better (Hsu & Moortgat 2026 compared RF with four
  CNNs and found that the way training blocks are sampled mattered most), but they need *dense*
  labels. ATL24 gives thin lines; training a U-Net on lines is a research project of its own.
- Geospatial foundation models (Prithvi, SatMAE...) are pre-trained on land; fine-tuning them for
  water depth with line labels on a 4 GB GPU is not justified for a first version. Listed as a
  next step.
- What I **reuse**: NASA's ATL24 ensemble classifier and refraction correction, ESA's Sen2Cor
  atmospheric correction, and the published physics features (Stumpf/Lyzenga). What I **build**:
  the tide-free labelling, pixel matching, glint-corrected composite, leakage-aware evaluation
  and the models.

**Outcome in this project.** RF RMSE 0.99 m, MAE 0.47 m, R² 0.84; XGBoost 1.02 m. Physics-only features
0.989 m vs raw bands 1.044 m. Top feature: blue/green ratio (permutation importance 0.42).

---

## 11. Spatial leakage in geospatial ML and how to split

**The problem.** Nearby pixels have nearly the same depth and nearly the same reflectance
(spatial autocorrelation). If a random split puts one pixel in training and its neighbour in
test, the test score measures memorisation of the neighbourhood, not the ability to map
new areas. Ploton et al. (2020) showed a random-forest biomass map that looked like it explained
over half the variance under random CV and had almost no predictive power under spatial CV.
Roberts et al. (2017) recommend blocked cross-validation when the goal is to predict new
locations.

**Specific to ICESat-2.** Photons along a track are ~0.7 m apart, so a random split of points
is almost pure leakage. Splitting **by track** is the obvious fix (and the brief requires it). In
my AOI that is not enough on its own: repeat passes of one RGT and passes of different RGTs
interleave, sometimes only tens of metres apart, and 10,561 pixel pairs are hit by two
different passes.

**Decision.**
- Group = one ICESat-2 pass (track: RGT + cycle). Grouped K-fold: a track is never in train and
  test at the same time.
- Plus a **spatial buffer**: in each fold, training samples closer than B metres to any test
  sample are dropped. The buffer distance is chosen by a sensitivity analysis (error vs B).
- Report, side by side, random split vs track split vs track + buffer, to *measure* the leakage.

**Pitfall.** Blocking too aggressively forces extrapolation and overstates the error
(Roberts 2017). That is why I show the curve, not a single number.

**Outcome in this project.** RF RMSE: random 0.71 → by pass 0.79 → pass + 500 m 0.86 → RGT + 500 m 0.91 →
RGT + 1 km 0.99 m; plateau at ~1 km (2 km: 1.00). The depth variogram keeps rising to 5 km.

Sources: Roberts et al. 2017 (Ecography 40(8):913–929); Ploton et al. 2020 (Nature
Communications 11:4540); Wadoux et al. 2021 (critique of over-pessimistic spatial CV).

---

## 12. Metrics

- **RMSE** (penalises large errors; the standard in SDB papers, so results are comparable).
- **MAE** and **median absolute error** (robust; easier to explain: "typical error").
- **Bias** (mean error): SDB methods often under-estimate deep water.
- **R²** for comparison with literature (Gafoor 2022, ISPRS 2024).
- **By depth bin** (0–5, 5–10, 10–20 m), because errors grow with depth and the data is
  dominated by < 2 m: one overall number would hide the deep-water performance.
- **Relative error** (RMSE / mean depth), as Thomas et al. 2021 report (~10–15 %).
- IHO S-44 (hydrographic survey standard) defines allowed vertical uncertainty as
  sqrt(a² + (b·d)²), e.g. Order 1b: a = 0.5 m, b = 0.013. SDB is not meant to meet survey
  orders; I quote it only to show where SDB sits (reconnaissance/planning, not navigation charts).

**Outcome in this project.** All metrics reported per bin; the 10–20 m bin (n = 2,166) is where every
method fails (RF 3.57 m, Stumpf 7.22 m).

---

## 13. Typical depth limits of optical SDB

- Rule of thumb: optical SDB works to about the Secchi depth or slightly beyond, roughly
  ~1–1.5× Secchi, in practice 10–30 m in clear water.
- Stumpf et al. (2003): ratio method retrieves > 25 m in clear water.
- Thomas et al. (2021): ICESat-2 + Sentinel-2 maps to ~26 m (NASA press release).
- Viaña-Borja et al. (2023): to 14 m in the Mediterranean.
- ICESat-2 itself: up to ~40 m in the clearest waters (ATL24 user guide); theoretical 100 m.
- The Gulf is shallow, warm and often hazy/turbid near the coast. My ATL24 labels stop at 28.5 m
  and are rare beyond 15 m (277 of 97,730 samples), so I expect the reliable range to end
  around 10–15 m.

**Outcome in this project.** Reliable to ~10 m: RF RMSE 0.65 / 1.77 / 3.57 m in 0–5 / 5–10 / 10–20 m, with
under-estimation beyond 10 m (bias −2.1 m).

---

## 14. Backup reference data

- **GEBCO** global grid: 15 arc-second (~460 m) compilation; in the shallow Gulf it is mostly
  interpolated from sparse soundings and charts.
- **GMRT** (Global Multi-Resolution Topography): same GEBCO base here, served through an open
  API (`src/download_gebco.py`). Used only as a regional sanity check, not pixel validation.
- Nautical charts (UKHO/Abu Dhabi Ports) are not open.

**Outcome in this project.** SDB averaged into GMRT cells: r = 0.91, median |diff| 1.27 m (310,588 cells).

---

## 15. Related work and how my approach differs

| Work | Labels | Method | Split | Notes |
|---|---|---|---|---|
| Stumpf et al. 2003 | Airborne lidar | Log-ratio | — | The baseline method |
| Thomas et al. 2021 | ICESat-2 ATL03 (own classification) | Lyzenga, Stumpf, SVR | Hold-out vs NOAA DEM | Florida, Crete, Bermuda; fully space-borne |
| Sagawa et al. 2019 | Survey data | RF, multi-temporal Landsat-8 | Areas | RMSE 1.41 m, 0–20 m |
| Gafoor et al. 2022 (Abu Dhabi) | Boat echo-sounding | Linear + gradient boosting on ratios | **Random ~70/30** | Sen2Cor best; tide = main error |
| Lund thesis (ATL24) | ATL24 | Stumpf, RF, CatBoost, MLP | — | Validated ATL24 vs multibeam |
| **This project** | **ATL24 v002, tide-free** | Stumpf, Lyzenga, RF, XGBoost | **Track-grouped + spatial buffer** | Abu Dhabi; leakage measured |

**What is different in mine:**
1. **No field survey**: labels are ATL24 (NASA's official product), so the method can be run
   anywhere ICESat-2 has crossed clear water.
2. **Tide handled in the label**: tide-free depth below MSL, with a measured improvement in
   repeat-pass consistency, addressing Gafoor et al.'s main error source.
3. **Honest evaluation**: split by ICESat-2 track with a spatial buffer, and the size of the
   leakage error is reported, which a random split (as in Gafoor 2022) cannot show.
4. **Consistent imagery**: Collection-1 reprocessed archive + SWIR glint correction +
   multi-date median.

---

## 16. Source list

- NSIDC (2026). ATL24 v002 User Guide. https://nsidc.org/sites/default/files/documents/user-guide/atl24-v002-userguide.pdf ; dataset doi:10.5067/ATLAS/ATL24.002
- Magruder, L. A. et al. (2025). ICESat-2 Coastal and Nearshore Bathymetry Product Algorithm Development. Earth and Space Science 12(10). doi:10.1029/2025EA004390
- Magruder, L. A. et al. (2026). ATL24 ATBD. NASA. doi:10.5067/UIN5PPVS10WW
- Parrish, C. E. et al. (2019). Validation of ICESat-2 ATLAS Bathymetry and Analysis of ATLAS's Bathymetric Mapping Performance. Remote Sensing 11(14):1634. https://www.mdpi.com/2072-4292/11/14/1634
- Thomas, N. et al. (2021). Space-Borne Cloud-Native Satellite-Derived Bathymetry (SDB) Models Using ICESat-2 and Sentinel-2. Geophysical Research Letters 48(6). doi:10.1029/2020GL092170
- Stumpf, R. P., Holderied, K., Sinclair, M. (2003). Determination of water depth with high-resolution satellite imagery over variable bottom types. Limnology and Oceanography 48(1, part 2):547–556. doi:10.4319/lo.2003.48.1_part_2.0547
- Lyzenga, D. R. (1978). Passive remote sensing techniques for mapping water depth and bottom features. Applied Optics 17(3):379–383. doi:10.1364/AO.17.000379
- Lyzenga, D. R. (1985). Shallow-water bathymetry using combined lidar and passive multispectral scanner data. International Journal of Remote Sensing 6(1):115–125.
- Caballero, I., Stumpf, R. P. (2019). Retrieval of nearshore bathymetry from Sentinel-2A and 2B satellites in South Florida coastal waters. Estuarine, Coastal and Shelf Science 226.
- Caballero, I., Stumpf, R. P. (2020). Towards routine mapping of shallow bathymetry in environments with variable turbidity: contribution of Sentinel-2A/B satellites mission. Remote Sensing 12(3):451.
- Viaña-Borja, S. P. et al. (2023). Semi-automated bathymetry using Sentinel-2 for coastal monitoring in the Western Mediterranean. IJAEOG 120.
- Gafoor, F. A., Al-Shehhi, M. R., Cho, C.-S., Ghedira, H. (2022). Gradient Boosting and Linear Regression for Estimating Coastal Bathymetry Based on Sentinel-2 Images. Remote Sensing 14(19):5037. doi:10.3390/rs14195037
- Sagawa, T. et al. (2019). Satellite Derived Bathymetry Using Machine Learning and Multi-Temporal Satellite Images. Remote Sensing 11(10):1155. doi:10.3390/rs11101155
- Chybicki, A. et al. (2023). Study of various machine learning approaches for Sentinel-2 derived bathymetry. PLOS ONE. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10503737/
- ISPRS Archives XLVIII-4-W8-2023, 387 (2024): RF on Sentinel-2 bands + Stumpf ratios. https://isprs-archives.copernicus.org/articles/XLVIII-4-W8-2023/387/2024/
- Lund University student thesis using ATL24 for Sentinel-2 SDB. https://lup.lub.lu.se/student-papers/record/9222699
- Hsu & Moortgat (2026). RF vs CNNs for Sentinel-2 SDB (arXiv 2606.02764).
- Hedley, J. D., Harborne, A. R., Mumby, P. J. (2005). Simple and robust removal of sun glint for mapping shallow-water benthos. IJRS 26(10):2107–2112. doi:10.1080/01431160500034086
- Kay, S., Hedley, J. D., Lavender, S. (2009). Sun glint correction of high and low spatial resolution images of aquatic scenes. Remote Sensing 1(4):697–730.
- Pope, R. M., Fry, E. S. (1997). Absorption spectrum (380–700 nm) of pure water. II. Applied Optics 36:8710–8723. Data: https://omlc.org/spectra/water/data/pope97.txt
- Hale, G. M., Querry, M. R. (1973). Optical constants of water in the 200 nm to 200 µm region. Applied Optics 12:555–563. Data: https://omlc.org/spectra/water/data/hale73.txt
- Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features. IJRS 27(14):3025–3033.
- Breiman, L. (2001). Random Forests. Machine Learning 45:5–32.
- Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40(8):913–929.
- Ploton, P. et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. Nature Communications 11:4540. doi:10.1038/s41467-020-18321-y
- Wadoux, A. M. J.-C. et al. (2021). Spatial cross-validation is not the right way to evaluate map accuracy. Ecological Modelling 457.
- SlideRule Earth: https://slideruleearth.io ; client `sliderule` 5.7.0 (`atl24x` endpoint)
- Earth Search STAC (Element 84): https://earth-search.aws.element84.com/v1 (`sentinel-2-c1-l2a`)
- GMRT GridServer: https://www.gmrt.org/services/GridServer ; GEBCO: https://www.gebco.net
- IHO S-44 Standards for Hydrographic Surveys, Edition 6.1 (2022).

---

## 17. Where each research finding shows up in the results

| Research finding | Source | Where it shows up here | Confirmed? |
|---|---|---|---|
| Blue/green penetrate deepest; red only a few metres | Pope & Fry 1997 | blue/red ratio saturates at ~5 m; switching model best in 0–5 m (0.87 m) | Yes |
| SWIR is absorbed within millimetres | Hale & Querry 1973 | basis of the glint correction and the MNDWI mask | Used as design input |
| Log-ratio is robust to bottom type, 2 parameters | Stumpf 2003 | top RF feature (importance 0.42); linear fit alone fails past 5 m | Partly |
| Lyzenga is the best classic method with ICESat-2 labels | Thomas 2021 | lowest classic RMSE (1.73 m), best in 5–20 m | Yes |
| ICESat-2 alone can calibrate Sentinel-2 | Thomas 2021 | R² 0.84 with zero in-situ data | Yes |
| Tide is the main SDB error in Abu Dhabi | Gafoor 2022 | ±1 m tide in labels; removing it: 0.34 → 0.10 m repeat-pass error | Yes, and handled |
| Sen2Cor is the best atmospheric correction here | Gafoor 2022 | used as-is | Adopted, not re-tested |
| Tree ensembles beat linear SDB | Sagawa 2019; Gafoor 2022; ISPRS 2024 | RF 0.99 vs Stumpf 2.04 m | Yes |
| Random splits over-state spatial accuracy | Roberts 2017; Ploton 2020 | random 0.71 vs honest 0.99 m (~28 %) | Yes |
| Over-blocking measures extrapolation | Roberts 2017 | baseline error also rises with the buffer; plateau at 1 km | Yes |
| Multi-scene compositing helps | Caballero & Stumpf 2020; Sagawa 2019 | composite beats single scene for every model | Yes |
| Optical SDB is limited to roughly the Secchi depth (10–30 m) | Stumpf 2003; Thomas 2021 | Gulf limit ~10 m; worst errors all at 9–11 m | Yes (lower end) |
| ATL24 contains false positives (class imbalance) | ATL24 user guide | mainland "seafloor" photons; 2–4 m labels in deep water | Yes |
