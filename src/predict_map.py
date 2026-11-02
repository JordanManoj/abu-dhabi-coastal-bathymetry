"""Train the chosen model on ALL labelled pixels and predict depth for every water pixel.

The cross-validated numbers in results/metrics_*.csv describe how good this map is;
the map model itself uses all labels (standard practice once the evaluation is done).
Pixels are flagged as unreliable when the prediction is outside the labelled range
(> 20 m), since nothing in the training data tells us what happens beyond it.

Usage:  python src/predict_map.py [--model xgb] [--image composite]
Output: results/depth_<model>.tif  (float32, metres below MSL, NaN = land/cloud)
        results/models/<model>.joblib
"""
import argparse

import joblib
import numpy as np
import pandas as pd
import rasterio

import features as F
from common import PROCESSED, RESULTS
from train_eval import make_model

MAX_RELIABLE = 20.0
STRIP = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--image", default="composite")
    args = ap.parse_args()
    df = pd.read_parquet(PROCESSED / f"samples_{args.image}.parquet")
    cols, model = make_model(args.model)
    model.fit(df[cols].to_numpy(), df.depth.to_numpy())
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=model, cols=cols), RESULTS / "models" / f"{args.model}.joblib", compress=3)  # ~400 MB raw; not in git

    bands, meta = F.load_composite(args.image)
    import json
    r_deep = json.loads((PROCESSED / f"r_deep_{args.image}.json").read_text())
    water = F.water_mask(bands, min_valid=3 if args.image == "composite" else 1)
    H, W = water.shape
    depth = np.full((H, W), np.nan, dtype="float32")
    for r0 in range(0, H, STRIP):
        sl = slice(r0, min(r0 + STRIP, H))
        m = water[sl]
        if not m.any():
            continue
        refl = {b: bands[b][sl][m] for b in F.BANDS}
        feats = F.compute(refl, r_deep)
        X = np.column_stack([feats[c] for c in cols])
        # same validity rule as training; red/coastal ratios may be NaN (trees handle it)
        ok = F.valid_pixels(refl) & np.isfinite(np.column_stack([refl[b] for b in F.BANDS])).all(1)
        pred = np.full(len(X), np.nan, dtype="float32")
        pred[ok] = model.predict(X[ok])
        block = depth[sl]
        block[m] = np.clip(pred, 0, None)
    prof = meta["profile"].copy()
    prof.update(count=2, dtype="float32", nodata=np.nan, compress="deflate", predictor=3)
    out = RESULTS / f"depth_{args.model}.tif"
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(depth, 1)
        dst.write((depth > MAX_RELIABLE).astype("float32"), 2)
        dst.set_band_description(1, "depth_m_below_MSL")
        dst.set_band_description(2, "beyond_training_range")
    v = depth[np.isfinite(depth)]
    print(f"saved {out}: {v.size:,} water pixels, depth p5/p50/p95 = {np.percentile(v, [5, 50, 95]).round(2)}, "
          f"{(v > MAX_RELIABLE).mean():.1%} beyond {MAX_RELIABLE} m")


if __name__ == "__main__":
    main()
