"""Train and evaluate the baselines and ML models with leakage-aware cross-validation.

Models
  stumpf   : depth = m1 * ln(1000 B2)/ln(1000 B3) - m0        (Stumpf et al. 2003, baseline)
  stumpf_switch : red-ratio model in very shallow water, green-ratio deeper (Caballero & Stumpf 2020)
  lyzenga  : multiple linear regression on ln(Ri - Rdeep_i)    (Lyzenga 1978/1985)
  rf       : Random Forest on all features
  xgb      : XGBoost on all features
Split schemes (every sample is predicted once, by a model that never saw its fold)
  random        : KFold on samples (what a naive split does; leaky)
  track         : GroupKFold by ICESat-2 pass (RGT + cycle)
  track_buffer  : as 'track', and training samples within BUFFER m of any test sample are dropped
  rgt_buffer    : leave-one-RGT-out (all cycles of a ground track held out together) + buffer
Usage:  python src/train_eval.py [--image composite] [--schemes random,track,track_buffer,rgt_buffer]
        [--buffer 500] [--models stumpf,lyzenga,rf,xgb] [--tag name]
Output: results/metrics_<tag>.csv  (overall + per depth bin, per model and scheme)
        results/oof_<tag>.parquet  (out-of-fold predictions for figures)
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from xgboost import XGBRegressor

import features as F
from common import PROCESSED, RESULTS

SEED = 42
BINS = [(0, 5), (5, 10), (10, 20), (0, 20)]
MIN_GROUP_SAMPLES = 100  # RGTs 1210/1271 have < 10 samples; they stay in training only


class StumpfSwitch:
    """Switching log-ratio model in the spirit of Caballero & Stumpf (2020).

    Two Stumpf models: blue/red (sensitive in very shallow water, saturates by ~4-5 m in our
    data) and blue/green (works deeper). Use the red estimate below A m, the green estimate
    above B m, and a linear blend in between. A, B come from our own per-bin analysis of
    where the blue/red ratio stops rising (DECISIONS.md).
    """
    A, B, RED_FIT_MAX = 2.0, 4.0, 5.0
    RED_MIN = 0.01  # red reflectance below this = no bottom signal in red (median 0.016 at 2-5 m)

    def _red_ok(self, br, red):
        return np.isfinite(br) & (red > self.RED_MIN)

    def fit(self, X, y):
        bg, br, red = X[:, 0], X[:, 1], X[:, 2]
        self.green = LinearRegression().fit(bg[:, None], y)
        ok = self._red_ok(br, red) & (y <= self.RED_FIT_MAX)
        self.red = LinearRegression().fit(br[ok][:, None], y[ok])
        return self

    def predict(self, X):
        bg, br, red = X[:, 0], X[:, 1], X[:, 2]
        sg = self.green.predict(bg[:, None])
        ok = self._red_ok(br, red)
        sr = np.where(ok, self.red.predict(np.nan_to_num(br)[:, None]), sg)
        w = np.clip((sr - self.A) / (self.B - self.A), 0, 1)  # 0 -> red, 1 -> green
        return (1 - w) * sr + w * sg


class DepthBalanced:
    """Wraps a regressor and weights training samples by the inverse frequency of their
    depth bin (capped), so the rare deep samples are not drowned by the 0-2 m majority."""
    EDGES = [0, 2, 5, 10, 15, 30]
    CAP = 20.0

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        b = np.digitize(y, self.EDGES[1:-1])
        freq = np.bincount(b, minlength=len(self.EDGES) - 1) / len(y)
        w = np.minimum(1.0 / freq[b], self.CAP * (1.0 / freq.max()))
        self.model.fit(X, y, sample_weight=w / w.mean())
        return self

    def predict(self, X):
        return self.model.predict(X)


def make_model(name):
    if name == "stumpf":
        return ["pSDB_bg"], LinearRegression()
    if name == "stumpf_switch":
        return ["pSDB_bg", "pSDB_br", "B04"], StumpfSwitch()
    if name == "lyzenga":
        return [f"lyz_{b}" for b in F.BANDS[:4]], LinearRegression()
    if name == "rf":
        return F.FEATURES, RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                                 n_jobs=-1, random_state=SEED)
    if name == "rf_balanced":  # experiment: depth-balanced sample weights
        return F.FEATURES, DepthBalanced(make_model("rf")[1])
    if name == "rf_bands":   # ablation: raw reflectance only
        return F.BANDS, make_model("rf")[1]
    if name == "rf_physics":  # ablation: Stumpf ratios + Lyzenga transforms only
        return [c for c in F.FEATURES if c not in F.BANDS], make_model("rf")[1]
    if name == "xgb":
        return F.FEATURES, XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8,
                                        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                                        tree_method="hist", n_jobs=-1, random_state=SEED)
    raise ValueError(name)


def folds(df, scheme, n_splits=5):
    """Yield (train_idx, test_idx) positional indices."""
    idx = np.arange(len(df))
    if scheme == "random":
        yield from KFold(n_splits, shuffle=True, random_state=SEED).split(idx)
    elif scheme.startswith("track"):
        yield from GroupKFold(n_splits).split(idx, groups=df.pass_id)
    elif scheme.startswith("rgt"):
        big = df.rgt.map(df.rgt.value_counts()) >= MIN_GROUP_SAMPLES
        for tr, te in LeaveOneGroupOut().split(idx[big], groups=df.rgt[big]):
            test = idx[big][te]
            yield np.setdiff1d(idx, test), test
    else:
        raise ValueError(scheme)


def apply_buffer(df, tr, te, buffer_m):
    """Drop training samples closer than buffer_m to any test sample."""
    if buffer_m <= 0:
        return tr
    xy = df[["x", "y"]].to_numpy()
    d, _ = cKDTree(xy[te]).query(xy[tr], k=1, distance_upper_bound=buffer_m)
    return tr[~np.isfinite(d)]


def metrics(y, p):
    e = p - y
    return dict(n=len(y), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                medae=float(np.median(np.abs(e))), bias=float(np.mean(e)),
                r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)) if len(y) > 1 else np.nan)


def evaluate(df, models, scheme, buffer_m):
    oof = {m: np.full(len(df), np.nan) for m in models}
    train_sizes = []
    for k, (tr, te) in enumerate(folds(df, scheme)):
        tr = apply_buffer(df, tr, te, buffer_m if "buffer" in scheme else 0)
        train_sizes.append(len(tr))
        for m in models:
            cols, model = make_model(m)
            model.fit(df.iloc[tr][cols].to_numpy(), df.iloc[tr].depth.to_numpy())
            oof[m][te] = model.predict(df.iloc[te][cols].to_numpy())
    rows = []
    y = df.depth.to_numpy()
    for m in models:
        ok = np.isfinite(oof[m])
        for lo, hi in BINS:
            sel = ok & (y > lo) & (y <= hi)
            rows.append(dict(model=m, scheme=scheme, buffer_m=buffer_m if "buffer" in scheme else 0,
                             depth_bin=f"{lo}-{hi}", mean_train_size=int(np.mean(train_sizes)),
                             **metrics(y[sel], oof[m][sel])))
    return pd.DataFrame(rows), oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="composite")
    ap.add_argument("--schemes", default="random,track,track_buffer,rgt_buffer")
    ap.add_argument("--models", default="stumpf,stumpf_switch,lyzenga,rf,xgb")
    ap.add_argument("--buffer", type=float, default=500)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.image
    df = pd.read_parquet(PROCESSED / f"samples_{args.image}.parquet")
    df = df[df.depth <= 20].reset_index(drop=True)  # evaluation range of the brief (0-20 m)
    models = args.models.split(",")
    all_rows, oof_frames = [], []
    for scheme in args.schemes.split(","):
        t = time.time()
        res, oof = evaluate(df, models, scheme, args.buffer)
        all_rows.append(res)
        o = df[["pass_id", "rgt", "row", "col", "x", "y", "depth"]].copy()
        o["scheme"] = scheme
        for m in models:
            o[f"pred_{m}"] = oof[m]
        oof_frames.append(o)
        print(f"--- {scheme} ({time.time() - t:.0f} s)")
        print(res[res.depth_bin == "0-20"][["model", "n", "rmse", "mae", "bias", "r2", "mean_train_size"]].round(3).to_string(index=False))
    RESULTS.mkdir(exist_ok=True)
    pd.concat(all_rows).to_csv(RESULTS / f"metrics_{tag}.csv", index=False)
    pd.concat(oof_frames).to_parquet(RESULTS / f"oof_{tag}.parquet", index=False)


if __name__ == "__main__":
    main()
