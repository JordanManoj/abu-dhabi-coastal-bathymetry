"""All result figures and summary tables (every number comes from files in results/).

Usage:  python src/make_figures.py
Output: results/figures/*.png, results/failures/*.png, results/summary.json
"""
import json

import matplotlib
import numpy as np
import pandas as pd
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from rasterio.warp import Resampling, reproject  # noqa: E402

import features as F  # noqa: E402
from common import FAILS, FIGS, RAW, RESULTS  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 10})
DEPTH_CMAP = LinearSegmentedColormap.from_list("depth", ["#e8f6f3", "#7fd1c7", "#2a9d8f", "#21678f", "#1d3557", "#0b1a33"])
NAMES = {"stumpf": "Stumpf (blue/green)", "stumpf_switch": "Stumpf switching", "lyzenga": "Lyzenga",
         "rf": "Random Forest", "xgb": "XGBoost", "rf_bands": "RF: bands only",
         "rf_physics": "RF: physics features only", "rf_balanced": "RF: depth-balanced"}
SUMMARY = {}


def rgb_background(bands, step):
    # add B11 back: the stored bands are glint-corrected (minus SWIR), which turns land black
    rgb = np.dstack([bands[b] + bands["B11"] for b in ("B04", "B03", "B02")])[::step, ::step]
    rgb = np.clip(np.nan_to_num(rgb + 0.003) / 0.25, 0, 1) ** 0.6
    return rgb


def load_depth(model="rf"):
    with rasterio.open(RESULTS / f"depth_{model}.tif") as src:
        return src.read(1), src.transform, src.crs


def fig_depth_map(bands, depth, step=4):
    fig, ax = plt.subplots(figsize=(14, 10.5))
    ax.imshow(rgb_background(bands, step) * 0.85)
    d = depth[::step, ::step]
    im = ax.imshow(np.ma.masked_invalid(d), cmap=DEPTH_CMAP, vmin=0, vmax=20)
    from scipy.ndimage import gaussian_filter
    filled = np.where(np.isfinite(d), d, 0)
    smooth = gaussian_filter(filled, 3) / np.maximum(gaussian_filter(np.isfinite(d).astype(float), 3), 1e-6)
    smooth[~np.isfinite(d)] = np.nan  # smoothed (~120 m) only for contours, so noise does not draw speckle
    cs = ax.contour(smooth, levels=[2, 5, 10], colors="white", linewidths=0.5, alpha=0.7)
    ax.clabel(cs, fmt="%d m", fontsize=7)
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("Depth below mean sea level (m)")
    ax.set_title("Satellite-derived bathymetry, Abu Dhabi coast (Random Forest on Sentinel-2, trained on ICESat-2 ATL24)\n"
                 "10 m pixels. Reliable to ~10 m; deeper values are under-estimated (see error by depth)")
    ax.axis("off")
    fig.savefig(FIGS / "depth_map.png", dpi=120)
    plt.close(fig)


def fig_scatter(oof):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)
    oof = oof.dropna(subset=["pred_rf"])  # 3 samples of the two tiny RGTs are never held out
    y = oof.depth.to_numpy()
    for ax, m in zip(axes, ["stumpf", "stumpf_switch", "rf"]):
        p = oof[f"pred_{m}"].to_numpy()
        hb = ax.hexbin(y, p, gridsize=70, extent=(0, 20, 0, 20), bins="log", cmap="viridis", mincnt=1)
        ax.plot([0, 20], [0, 20], "r--", lw=1)
        e = p - y
        ax.set_title(f"{NAMES[m]}\nRMSE {np.sqrt(np.mean(e**2)):.2f} m, MAE {np.mean(np.abs(e)):.2f} m")
        ax.set_xlabel("ICESat-2 ATL24 depth (m)")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
    axes[0].set_ylabel("Predicted depth (m)")
    plt.colorbar(hb, ax=axes, shrink=0.8, label="samples (log)")
    fig.suptitle("Predicted vs measured, held-out ground tracks (leave-one-RGT-out + 1 km buffer)", y=1.02)
    fig.savefig(FIGS / "pred_vs_measured.png")
    plt.close(fig)


def fig_error_by_bin(metrics):
    m = metrics[metrics.model.isin(["stumpf", "stumpf_switch", "lyzenga", "xgb", "rf"])]
    piv = m.pivot_table(index="depth_bin", columns="model", values="rmse").loc[["0-5", "5-10", "10-20", "0-20"]]
    piv = piv[["stumpf", "stumpf_switch", "lyzenga", "xgb", "rf"]]
    counts = metrics[metrics.model == "rf"].set_index("depth_bin").n
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(piv))
    colors = ["#bdbdbd", "#969696", "#737373", "#f4a261", "#2a9d8f"]
    for i, c in enumerate(piv.columns):
        bars = ax.bar(x + (i - 2) * 0.16, piv[c], 0.16, label=NAMES[c], color=colors[i])
        if c == "rf":
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, [f"{b} m\n(n={counts[b]:,})" for b in piv.index])
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Error by depth bin, held-out ground tracks")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(FIGS / "rmse_by_depth_bin.png")
    plt.close(fig)


def fig_leakage():
    a = pd.concat([pd.read_csv(RESULTS / f) for f in ["metrics_composite_a.csv", "metrics_composite_b.csv", "metrics_primary.csv"]])
    a = a[(a.depth_bin == "0-20") & a.model.isin(["stumpf", "rf"])]
    a["label"] = a.scheme.map({"random": "Random split", "track": "By track", "track_buffer": "By track\n+500 m buffer",
                               "rgt_buffer": "By ground track\n+buffer"})
    a.loc[a.scheme == "rgt_buffer", "label"] = a.loc[a.scheme == "rgt_buffer", "buffer_m"].map(
        lambda b: f"By ground track\n+{int(b)} m buffer")
    order = ["Random split", "By track", "By track\n+500 m buffer", "By ground track\n+500 m buffer", "By ground track\n+1000 m buffer"]
    bs = pd.read_csv(RESULTS / "buffer_sensitivity.csv").sort_values("buffer_m")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for m, c in [("stumpf", "#969696"), ("rf", "#2a9d8f")]:
        s = a[a.model == m].drop_duplicates("label").set_index("label").reindex(order)
        axes[0].plot(range(len(order)), s.rmse, "o-", color=c, label=NAMES[m])
        for i, v in enumerate(s.rmse):
            axes[0].annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    axes[0].set_xticks(range(len(order)), order, fontsize=8)
    axes[0].set_ylabel("RMSE (m), 0-20 m")
    axes[0].set_title("Same model, different split: how much a leaky split flatters")
    axes[0].legend()
    for m, c in [("stumpf_switch", "#969696"), ("rf", "#2a9d8f")]:
        s = bs[bs.model == m]
        axes[1].plot(s.buffer_m, s.rmse, "o-", color=c, label=NAMES[m])
    ax2 = axes[1].twinx()
    s = bs[bs.model == "rf"]
    ax2.bar(s.buffer_m, s.mean_train_size, width=80, alpha=0.15, color="k")
    ax2.set_ylabel("training samples per fold (bars)")
    axes[1].axvline(1000, ls="--", c="r", lw=1)
    axes[1].set_xlabel("buffer around held-out ground track (m)")
    axes[1].set_ylabel("RMSE (m)")
    axes[1].set_title("Buffer sensitivity: RF error plateaus from ~1 km (chosen)")
    axes[1].legend(loc="center right")
    fig.savefig(FIGS / "leakage.png")
    plt.close(fig)
    SUMMARY["leakage_rf_rmse"] = {l.replace("\n", " "): round(float(v), 3) for l, v in
                                  a[a.model == "rf"].drop_duplicates("label").set_index("label").rmse.items()}


def fig_error_map(bands, oof, step=4):
    oof = oof.dropna(subset=["pred_rf"])
    e = (oof.pred_rf - oof.depth).to_numpy()
    fig, ax = plt.subplots(figsize=(14, 10.5))
    ax.imshow(rgb_background(bands, step) * 0.7)
    o = np.argsort(np.abs(e))
    sc = ax.scatter(oof.col.to_numpy()[o] / step, oof.row.to_numpy()[o] / step, c=e[o], s=0.6, cmap="RdBu_r", vmin=-3, vmax=3)
    plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.01, label="RF error = predicted - ATL24 (m); red = predicted too deep")
    ax.set_title("Error map along held-out ICESat-2 tracks (Random Forest, leave-one-RGT-out + 1 km)")
    ax.axis("off")
    fig.savefig(FIGS / "error_map.png", dpi=120)
    plt.close(fig)


def fig_profile(bands, oof):
    """Along-track profile of one held-out pass with a wide depth range."""
    g = oof.groupby("pass_id").depth.agg(["size", "max"])
    pid = g[(g["size"] > 800)].sort_values("max").index[-1]
    t = oof[oof.pass_id == pid].sort_values("y")
    dist = np.hypot(t.x - t.x.iloc[0], t.y - t.y.iloc[0]) / 1000
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.scatter(dist, t.depth, s=3, c="k", label="ICESat-2 ATL24 (truth)")
    ax.scatter(dist, t.pred_stumpf, s=2, c="#969696", label="Stumpf")
    ax.scatter(dist, t.pred_rf, s=2, c="#2a9d8f", label="Random Forest")
    ax.invert_yaxis()
    ax.set_xlabel("distance along track (km, south to north)")
    ax.set_ylabel("depth (m)")
    ax.set_title(f"Along-track profile, held-out pass {pid} (RGT {pid.split('_')[0]}, cycle {pid.split('_')[1]}), never seen in training")
    ax.legend(markerscale=4)
    fig.savefig(FIGS / "track_profile.png")
    plt.close(fig)
    SUMMARY["profile_pass"] = pid


def fig_feature_importance():
    import joblib
    from sklearn.inspection import permutation_importance
    d = joblib.load(RESULTS / "models" / "rf.joblib")
    s = pd.read_parquet(RESULTS.parent / "data" / "processed" / "samples_composite.parquet").sample(8000, random_state=0)
    r = permutation_importance(d["model"], s[d["cols"]].to_numpy(), s.depth.to_numpy(), n_repeats=5, random_state=0, n_jobs=1)
    imp = pd.Series(r.importances_mean, index=d["cols"]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color="#2a9d8f")
    ax.set_xlabel("permutation importance (drop in R² when shuffled)")
    ax.set_title("What the Random Forest relies on")
    fig.savefig(FIGS / "feature_importance.png")
    plt.close(fig)
    SUMMARY["feature_importance_top5"] = imp.sort_values(ascending=False).head(5).round(3).to_dict()


def pass_bias(oof):
    """How much of the error is a constant offset per ICESat-2 pass, and what drives it?"""
    from common import PROCESSED
    o = oof.dropna(subset=["pred_rf"]).merge(
        pd.read_parquet(PROCESSED / "samples_composite.parquet")[["pass_id", "row", "col", "tide", "night"]],
        on=["pass_id", "row", "col"])
    o["err"] = o.pred_rf - o.depth
    p = o.groupby("pass_id").agg(n=("err", "size"), bias=("err", "mean"), tide=("tide", "first"), depth=("depth", "median"))
    p = p[p.n >= 200]
    SUMMARY["per_pass_bias"] = dict(
        passes=len(p), std_of_pass_bias_m=round(float(p.bias.std()), 3),
        share_of_error_variance=round(float(o.pass_id.map(o.groupby("pass_id").err.mean()).var() / o.err.var()), 3),
        corr_bias_vs_pass_median_depth=round(float(np.corrcoef(p.bias, p.depth)[0, 1]), 3),
        corr_bias_vs_tide=round(float(np.corrcoef(p.bias, p.tide)[0, 1]), 3))


def composite_vs_single():
    c = pd.read_parquet(RESULTS / "oof_primary.parquet")
    s = pd.read_parquet(RESULTS / "oof_single.parquet")
    j = c.merge(s, on=["pass_id", "row", "col"], suffixes=("_c", "_s"))
    out = {}
    for m in ["stumpf", "stumpf_switch", "lyzenga", "rf", "xgb"]:
        out[m] = {k: round(float(np.sqrt(np.mean((j[f"pred_{m}_{k}"] - j.depth_c) ** 2))), 3) for k in ("c", "s")}
    SUMMARY["composite_vs_single_rmse_common_pixels"] = dict(n=len(j), **{m: {"composite": v["c"], "single": v["s"]} for m, v in out.items()})


def gebco_check(depth, transform, crs):
    """Aggregate the 10 m SDB to the GMRT/GEBCO grid and compare (regional sanity check only)."""
    with rasterio.open(RAW / "gebco" / "gmrt_abu_dhabi.tif") as src:
        g = src.read(1).astype("float32")
        dst = np.full(g.shape, np.nan, dtype="float32")
        reproject(depth, dst, src_transform=transform, src_crs=crs, dst_transform=src.transform, dst_crs=src.crs,
                  resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    gd = -g
    ok = np.isfinite(dst) & (gd > 0) & (gd < 30)
    r = np.corrcoef(dst[ok], gd[ok])[0, 1]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.hexbin(gd[ok], dst[ok], gridsize=50, extent=(0, 25, 0, 25), bins="log", mincnt=1)
    ax.plot([0, 25], [0, 25], "r--", lw=1)
    ax.set_xlabel("GMRT / GEBCO depth (m), ~460 m native")
    ax.set_ylabel("SDB (RF) averaged to the same cells (m)")
    ax.set_title(f"Regional sanity check vs GEBCO-based grid\nr = {r:.2f}, n = {ok.sum():,} cells")
    fig.savefig(FIGS / "gebco_check.png")
    plt.close(fig)
    SUMMARY["gebco_check"] = dict(n_cells=int(ok.sum()), pearson_r=round(float(r), 3),
                                  median_abs_diff_m=round(float(np.median(np.abs(dst[ok] - gd[ok]))), 2))


def failure_cases(bands, depth, oof, n=6, half=150):
    """Crops (3 x 3 km) around the worst held-out errors: RGB | SDB | ATL24 truth vs RF along the track."""
    o = oof.dropna(subset=["pred_rf"]).copy()
    o["err"] = o.pred_rf - o.depth
    o["cell"] = (o.row // (2 * half)).astype(str) + "_" + (o.col // (2 * half)).astype(str)
    worst = (o.assign(ae=o.err.abs()).groupby("cell").agg(ae=("ae", "mean"), n=("ae", "size"), row=("row", "median"),
             col=("col", "median"), bias=("err", "mean"), depth=("depth", "mean"))
             .query("n >= 50").sort_values("ae", ascending=False).head(n))
    H, W = depth.shape
    rows = []
    for k, (cell, w) in enumerate(worst.iterrows()):
        r0, c0 = int(max(0, w.row - half)), int(max(0, w.col - half))
        r1, c1 = min(H, r0 + 2 * half), min(W, c0 + 2 * half)
        sub = o[(o.row >= r0) & (o.row < r1) & (o.col >= c0) & (o.col < c1)]
        rgb = np.clip(np.nan_to_num(np.dstack([bands[b][r0:r1, c0:c1] for b in ("B04", "B03", "B02")]) + 0.003) / 0.25, 0, 1) ** 0.6
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        ax[0].imshow(rgb)
        ax[0].scatter(sub.col - c0, sub.row - r0, s=1, c="yellow")
        ax[0].set_title("Sentinel-2 composite + ATL24 track (yellow)")
        im = ax[1].imshow(depth[r0:r1, c0:c1], cmap=DEPTH_CMAP, vmin=0, vmax=20)
        plt.colorbar(im, ax=ax[1], shrink=0.8, pad=0.02).set_label("SDB depth (m)")
        ax[2].yaxis.set_label_position("right")
        ax[1].set_title("RF depth map")
        ax[2].scatter(sub.depth, sub.pred_rf, s=3, alpha=0.5, c="#2a9d8f")
        lim = max(5, float(sub.depth.max()) + 1)
        ax[2].plot([0, lim], [0, lim], "r--", lw=1)
        ax[2].set_xlabel("ATL24 depth (m)")
        ax[2].set_ylabel("RF predicted (m)")
        ax[2].set_title(f"MAE {w.ae:.2f} m, bias {w.bias:+.2f} m, n={int(w.n)}")
        for a in ax[:2]:
            a.axis("off")
        fig.suptitle(f"Failure case {k + 1}: 3 x 3 km crop at pixel row {r0}, col {c0}")
        fig.savefig(FAILS / f"failure_{k + 1}.png")
        plt.close(fig)
        rows.append(dict(case=k + 1, row=r0, col=c0, mae=round(w.ae, 2), bias=round(w.bias, 2), mean_depth=round(w.depth, 2), n=int(w.n)))
    pd.DataFrame(rows).to_csv(FAILS / "failure_cases.csv", index=False)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    FAILS.mkdir(parents=True, exist_ok=True)
    bands, _ = F.load_composite("composite")
    depth, transform, crs = load_depth("rf")
    oof = pd.read_parquet(RESULTS / "oof_primary.parquet")
    metrics = pd.read_csv(RESULTS / "metrics_primary.csv")
    for f in (lambda: fig_depth_map(bands, depth), lambda: fig_scatter(oof), lambda: fig_error_by_bin(metrics),
              fig_leakage, lambda: fig_error_map(bands, oof), lambda: fig_profile(bands, oof), fig_feature_importance,
              composite_vs_single, lambda: pass_bias(oof), lambda: gebco_check(depth, transform, crs), lambda: failure_cases(bands, depth, oof)):
        f()
        print("done", getattr(f, "__name__", ""), flush=True)
    SUMMARY["primary_metrics_0_20"] = metrics[metrics.depth_bin == "0-20"].set_index("model")[["rmse", "mae", "bias", "r2"]].round(3).to_dict("index")
    (RESULTS / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=float))
    print(json.dumps(SUMMARY, indent=2, default=float))


if __name__ == "__main__":
    main()
