"""Leakage sensitivity: error vs. spatial buffer around the held-out ground track (RGT).

If nearby training pixels leak information, the error rises with the buffer and then
flattens once the buffer exceeds the leakage range (Roberts et al. 2017). Too large a
buffer starts to measure extrapolation instead, so we look for the plateau.
Usage:  python src/buffer_sensitivity.py [--buffers 0,100,250,500,1000,2000]
        (results are appended per buffer, so long runs can be split into several calls)
Output: results/buffer_sensitivity.csv
"""
import argparse

import pandas as pd

from common import PROCESSED, RESULTS
from train_eval import evaluate

BUFFERS = [0, 100, 250, 500, 1000, 2000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffers", default=",".join(map(str, BUFFERS)))
    args = ap.parse_args()
    df = pd.read_parquet(PROCESSED / "samples_composite.parquet")
    df = df[df.depth <= 20].reset_index(drop=True)
    out = RESULTS / "buffer_sensitivity.csv"
    for b in map(float, args.buffers.split(",")):
        res, _ = evaluate(df, ["stumpf_switch", "rf"], "rgt_buffer", b)
        res = res[res.depth_bin == "0-20"]
        print(b, res[["model", "rmse", "mae", "mean_train_size"]].round(3).values.tolist(), flush=True)
        res.to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
