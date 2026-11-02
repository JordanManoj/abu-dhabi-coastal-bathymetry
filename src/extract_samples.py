"""Join the ATL24 depth labels with the Sentinel-2 features at the same 10 m pixel.

Deep-water reflectance R_deep (needed by the Lyzenga features) is the 5th percentile of
each band over water pixels: the darkest water, as in Lyzenga (1978). Not the 0.5th: that
picks the 2-4 % of mixed shoreline pixels that are negative after glint removal.
Samples on pixels that the composite calls land (MNDWI <= 0) or with < 3 clear scenes
are dropped: this also removes ATL24 false positives on land.

Usage:  python src/extract_samples.py [--image composite|single]
Output: data/processed/samples_<image>.parquet, data/processed/r_deep_<image>.json
"""
import argparse
import json

import numpy as np
import pandas as pd

import features as F
from common import INTERIM, PROCESSED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="composite")
    args = ap.parse_args()
    bands, _ = F.load_composite(args.image)
    water = F.water_mask(bands, min_valid=3 if args.image == "composite" else 1)
    r_deep = {b: float(np.nanpercentile(bands[b][water], 5)) for b in F.BANDS}
    (PROCESSED / f"r_deep_{args.image}.json").write_text(json.dumps(r_deep, indent=2))

    s = pd.read_parquet(INTERIM / "atl24_pixels.parquet")
    H, W = bands["B02"].shape
    s = s[(s.row >= 0) & (s.row < H) & (s.col >= 0) & (s.col < W)].copy()
    r, c = s.row.to_numpy(), s.col.to_numpy()
    s["is_water"] = water[r, c]
    refl = {b: bands[b][r, c] for b in F.BANDS}
    s["is_valid"] = F.valid_pixels(refl)
    feats = F.compute(refl, r_deep)
    for k, v in feats.items():
        s[k] = v
    n0 = len(s)
    print(f"dropped: {(~s.is_water).sum():,} not water, {(s.is_water & ~s.is_valid).sum():,} invalid log-ratio")
    s = s[s.is_water & s.is_valid].dropna(subset=F.BANDS + ["pSDB_bg"])
    s.to_parquet(PROCESSED / f"samples_{args.image}.parquet", index=False)
    print(f"{args.image}: {n0:,} label pixels -> {len(s):,} water samples "
          f"({s.pass_id.nunique()} passes, {s.rgt.nunique()} RGTs); R_deep = {r_deep}")


if __name__ == "__main__":
    main()
