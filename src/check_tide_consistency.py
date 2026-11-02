"""Check the tide decision: do repeat passes over the same 10 m pixel agree better
with tide-free depth (MSL - ortho_h) than with ATL24's instantaneous depth?

Usage:  python src/check_tide_consistency.py
Output: results/tide_consistency.json
"""
import itertools
import json

import numpy as np
import pandas as pd

from common import INTERIM, RESULTS


def main():
    d = pd.read_parquet(INTERIM / "atl24_pixels.parquet")
    pairs = []
    for (_, _), g in d.groupby(["row", "col"]):
        if len(g) < 2:
            continue
        for a, b in itertools.combinations(g.itertuples(), 2):
            if a.pass_id != b.pass_id:
                pairs.append((a.depth - b.depth, a.depth_inst - b.depth_inst, abs(a.tide - b.tide)))
    p = np.array(pairs)
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    out = dict(
        pixel_pairs=len(p),
        rms_diff_instantaneous_depth_m=round(rms(p[:, 1]), 3),
        rms_diff_tide_free_depth_m=round(rms(p[:, 0]), 3),
        median_abs_diff_instantaneous_m=round(float(np.median(np.abs(p[:, 1]))), 3),
        median_abs_diff_tide_free_m=round(float(np.median(np.abs(p[:, 0]))), 3),
        median_tide_difference_between_passes_m=round(float(np.median(p[:, 2])), 3),
    )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tide_consistency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
