"""Build the glint-corrected multi-date median composite (and a single-scene version).

Per scene:
  1. mask cloud, cloud shadow, cirrus, saturated and no-data pixels (SCL 0,1,3,8,9,10)
  2. sun-glint removal: R_corr = R - R_B11. Water absorbs SWIR-1 (1610 nm) within
     millimetres, so B11 over water is surface reflection (glint/sky) + haze, which is
     almost spectrally flat. NIR (the classic Hedley et al. 2005 choice) still sees the
     bottom over the < 1 m sand flats of Abu Dhabi, so NIR subtraction would delete depth
     signal exactly where we have most labels.
Across scenes:
  3. per-pixel median of the corrected reflectance (robust to residual glint, haze,
     boats, and the tide state of any single date)
  4. water mask: median MNDWI = (G - SWIR1)/(G + SWIR1) > 0 (McFeeters/Xu water index;
     B11 is used rather than NIR because shallow-water NIR is contaminated by the bottom)

Usage:  python src/build_composite.py
Output: data/processed/composite.tif  bands: B01 B02 B03 B04 B08 (corrected, reflectance),
        B11 (raw median), MNDWI, n_valid
        data/processed/single.tif     same layout, best-ranked scene only
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from common import INTERIM, PROCESSED, RAW

VIS = ["B01", "B02", "B03", "B04", "B08"]
BAD_SCL = [0, 1, 3, 8, 9, 10]
STRIP = 256
OUT_BANDS = VIS + ["B11", "MNDWI", "n_valid"]


def load_strip(scene_dir, win):
    rd = lambda b: rasterio.open(scene_dir / f"{b}.tif").read(1, window=win)
    scl = rd("SCL")
    swir = rd("B11").astype("float32") / 1e4
    vis = np.stack([rd(b).astype("float32") / 1e4 for b in VIS])
    valid = ~np.isin(scl, BAD_SCL) & (swir > 0) & (vis > 0).all(0)
    corr = vis - swir[None]
    corr[:, ~valid] = np.nan
    swir[~valid] = np.nan
    return corr, swir


def build(scene_ids, out_path):
    dirs = [RAW / "s2" / s for s in scene_ids]
    with rasterio.open(dirs[0] / "B02.tif") as ref:
        prof, H, W = ref.profile, ref.height, ref.width
    prof.update(count=len(OUT_BANDS), dtype="float32", nodata=np.nan, predictor=3, compress="deflate")
    with rasterio.open(out_path, "w", **prof) as dst:
        for r0 in range(0, H, STRIP):
            win = Window(0, r0, W, min(STRIP, H - r0))
            stacks = [load_strip(d, win) for d in dirs]
            corr = np.stack([s[0] for s in stacks])   # scenes x 5 x h x w
            swir = np.stack([s[1] for s in stacks])   # scenes x h x w
            with np.errstate(all="ignore"):
                import warnings
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(corr, axis=0)
                swir_med = np.nanmedian(swir, axis=0)
                green_raw = med[2] + swir_med
                mndwi = (green_raw - swir_med) / (green_raw + swir_med)
            n_valid = np.isfinite(corr[:, 2]).sum(0).astype("float32")
            out = np.concatenate([med, swir_med[None], mndwi[None], n_valid[None]]).astype("float32")
            dst.write(out, window=win)
            for i, name in enumerate(OUT_BANDS, start=1):
                dst.set_band_description(i, name)
    print("saved", out_path)


def main():
    chosen = pd.read_csv(INTERIM / "chosen_scenes.csv")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    build(list(chosen.id), PROCESSED / "composite.tif")
    build([chosen.id.iloc[0]], PROCESSED / "single.tif")


if __name__ == "__main__":
    main()
