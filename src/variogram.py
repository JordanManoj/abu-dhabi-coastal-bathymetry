"""Empirical semivariogram of the depth labels: how far apart must two pixels be before
their depths stop being related? This gives a data-driven scale for the CV buffer.

gamma(h) = 0.5 * mean[(z_i - z_j)^2] over pairs at distance ~h. It rises with h and levels
off at the 'range'. Computed on a random 15,000-sample subset (pairs grow as n^2).
Usage:  python src/variogram.py
Output: results/variogram.csv
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import PROCESSED, RESULTS

EDGES = [0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]


def main():
    df = pd.read_parquet(PROCESSED / "samples_composite.parquet")
    df = df[df.depth <= 20].sample(15000, random_state=0)
    xy, z = df[["x", "y"]].to_numpy(), df.depth.to_numpy()
    pairs = cKDTree(xy).query_pairs(EDGES[-1], output_type="ndarray")
    d = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    sq = 0.5 * (z[pairs[:, 0]] - z[pairs[:, 1]]) ** 2
    rows = [dict(h_min=a, h_max=b, n_pairs=int(((d >= a) & (d < b)).sum()),
                 gamma=float(sq[(d >= a) & (d < b)].mean())) for a, b in zip(EDGES[:-1], EDGES[1:])]
    out = pd.DataFrame(rows)
    out["gamma_over_variance"] = out.gamma / z.var()
    out.to_csv(RESULTS / "variogram.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
