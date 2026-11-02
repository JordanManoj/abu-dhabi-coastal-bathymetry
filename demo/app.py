"""Abu Dhabi Coastal Bathymetry: interactive demo.

Run from the repo root:   streamlit run demo/app.py
Needs results/depth_rf.tif and data/processed/composite.tif (see README "Run end to end").

Tabs
  Depth map      : colour-coded SDB map + "can my vessel pass?" draft-clearance view
  Point lookup   : depth at a latitude/longitude
  Held-out track : ICESat-2 profile vs predictions on a track the model never saw
  Results        : metrics tables and the evaluation figures
  Failure cases  : where and why the model is wrong
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
STEP = 3  # display downsampling (10 m -> 30 m) to keep the app light

st.set_page_config(page_title="Abu Dhabi SDB", layout="wide")


@st.cache_data
def load_depth():
    with rasterio.open(RES / "depth_rf.tif") as src:
        d = src.read(1)
        return d, src.transform, src.crs.to_string()


@st.cache_data
def load_rgb():
    with rasterio.open(ROOT / "data" / "processed" / "composite.tif") as src:
        names = list(src.descriptions)
        b = {n: src.read(i + 1)[::STEP, ::STEP] for i, n in enumerate(names) if n in ("B02", "B03", "B04", "B11")}
    rgb = np.dstack([b[k] + b["B11"] for k in ("B04", "B03", "B02")])
    return (np.clip(np.nan_to_num(rgb + 0.003) / 0.25, 0, 1) ** 0.6 * 255).astype("uint8")


def colourise(d, vmax=20):
    """Depth -> RGBA with the same light-to-dark-blue ramp as the report figures."""
    stops = np.array([[232, 246, 243], [127, 209, 199], [42, 157, 143], [33, 103, 143], [29, 53, 87], [11, 26, 51]], float)
    t = np.clip(np.nan_to_num(d) / vmax, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    a = np.where(np.isfinite(d), 255, 0)[..., None]
    return np.concatenate([rgb, a], -1).astype("uint8")


def blend(base, overlay):
    a = overlay[..., 3:4] / 255.0
    return (base * (1 - a) + overlay[..., :3] * a).astype("uint8")


st.title("Abu Dhabi Coastal Bathymetry from space")
st.caption("Sentinel-2 imagery + ICESat-2 ATL24 laser depths → Random Forest depth map at 10 m. "
           "All numbers are from held-out ICESat-2 ground tracks (leave-one-RGT-out + 1 km buffer).")

depth, transform, crs = load_depth()
summary = json.loads((RES / "summary.json").read_text())
pm = summary["primary_metrics_0_20"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("RF RMSE (0-20 m)", f"{pm['rf']['rmse']:.2f} m", f"{pm['rf']['rmse'] - pm['stumpf']['rmse']:+.2f} m vs Stumpf",
          delta_color="inverse")
c2.metric("RF MAE", f"{pm['rf']['mae']:.2f} m")
c3.metric("RF R²", f"{pm['rf']['r2']:.2f}", f"Stumpf: {pm['stumpf']['r2']:.2f}", delta_color="off")
c4.metric("Water pixels mapped", f"{np.isfinite(depth).sum() / 1e6:.1f} M")

tabs = st.tabs(["Depth map", "Point lookup", "Held-out track", "Results", "Failure cases"])

with tabs[0]:
    rgb = load_rgb()
    d = depth[::STEP, ::STEP][: rgb.shape[0], : rgb.shape[1]]
    mode = st.radio("View", ["Depth", "Draft clearance (navigation)"], horizontal=True)
    if mode == "Depth":
        st.image(blend(rgb, colourise(d)), caption="Depth below mean sea level, 0 (light) to 20 m (dark). "
                 "Reliable to ~10 m; deeper water is under-estimated.", width="stretch")
    else:
        col_a, col_b = st.columns(2)
        draft = col_a.slider("Vessel draft (m)", 0.5, 15.0, 3.0, 0.5)
        margin = col_b.slider("Safety margin (m), should cover the ~1 m typical error", 0.0, 3.0, 1.0, 0.5)
        need = draft + margin
        ov = np.zeros(d.shape + (4,), "uint8")
        ov[np.isfinite(d) & (d < need)] = [230, 57, 70, 170]
        ov[np.isfinite(d) & (d >= need)] = [42, 157, 143, 150]
        st.image(blend(rgb, ov), caption=f"Red: shallower than {need:.1f} m (draft + margin). Green: enough water "
                 "according to the SDB map. Screening only, never a substitute for official charts.",
                 width="stretch")
        w = np.isfinite(d)
        st.write(f"{(d[w] >= need).mean():.0%} of mapped water has ≥ {need:.1f} m.")

with tabs[1]:
    lat = st.number_input("Latitude", 24.30, 24.70, 24.52, 0.001, format="%.4f")
    lon = st.number_input("Longitude", 54.20, 54.80, 54.33, 0.001, format="%.4f")
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    col, row = ~transform * (x, y)
    r, c = int(row), int(col)
    if 0 <= r < depth.shape[0] and 0 <= c < depth.shape[1] and np.isfinite(depth[r, c]):
        win = depth[max(0, r - 1): r + 2, max(0, c - 1): c + 2]
        st.success(f"Predicted depth: **{depth[r, c]:.1f} m** below mean sea level "
                   f"(3x3 neighbourhood {np.nanmin(win):.1f}-{np.nanmax(win):.1f} m). Typical error in this "
                   f"depth range: see the Results tab.")
    else:
        st.warning("Land, cloud-masked or outside the map.")

with tabs[2]:
    oof = pd.read_parquet(RES / "oof_primary.parquet").dropna(subset=["pred_rf"])
    counts = oof.pass_id.value_counts()
    passes = counts[counts > 300].index.tolist()
    default = summary.get("profile_pass", passes[0])
    pid = st.selectbox("ICESat-2 pass (RGT_cycle), predicted by a model trained without its ground track",
                       passes, index=passes.index(default) if default in passes else 0)
    t = oof[oof.pass_id == pid].sort_values("y")
    dist = np.hypot(t.x - t.x.iloc[0], t.y - t.y.iloc[0]) / 1000
    chart = pd.DataFrame({"km": dist.round(3), "ICESat-2 (truth)": -t.depth, "Random Forest": -t.pred_rf,
                          "Stumpf": -t.pred_stumpf}).set_index("km")
    st.scatter_chart(chart, height=380)
    e = t.pred_rf - t.depth
    st.write(f"This pass: {len(t):,} pixels, RF RMSE {np.sqrt((e**2).mean()):.2f} m, "
             f"Stumpf RMSE {np.sqrt(((t.pred_stumpf - t.depth)**2).mean()):.2f} m. (Depth shown negative = below sea level.)")

with tabs[3]:
    m = pd.read_csv(RES / "metrics_primary.csv")
    tab = m.pivot_table(index="model", columns="depth_bin", values="rmse")[["0-5", "5-10", "10-20", "0-20"]].round(2)
    st.subheader("RMSE (m) by depth bin, held-out ground tracks")
    st.dataframe(tab.sort_values("0-20"))
    for f, cap in [("pred_vs_measured.png", "Predicted vs measured"), ("rmse_by_depth_bin.png", "Error by depth"),
                   ("leakage.png", "Why the split matters"), ("error_map.png", "Where the errors are"),
                   ("feature_importance.png", "What the model uses"), ("gebco_check.png", "Independent sanity check")]:
        st.image(str(RES / "figures" / f), caption=cap, width="stretch")

with tabs[4]:
    fc = pd.read_csv(RES / "failures" / "failure_cases.csv")
    st.write("The six 3 × 3 km areas with the largest held-out errors. All are 9-11 m deep: the bottom barely "
             "reflects any light back, so every optical method regresses towards the middle (under-estimates). "
             "Some ATL24 points in uniformly deep water read 2-4 m: likely water-column returns mislabelled as seafloor.")
    st.dataframe(fc)
    for i in fc.case:
        st.image(str(RES / "failures" / f"failure_{i}.png"), width="stretch")
