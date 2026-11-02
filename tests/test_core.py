"""Unit tests for the core logic (no data download needed).  Run:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as F  # noqa: E402
from train_eval import StumpfSwitch, apply_buffer, folds, metrics  # noqa: E402


def test_stumpf_ratio_is_undefined_below_validity_limit():
    r = F.stumpf_ratio(np.array([0.05, 0.0009, 0.05]), np.array([0.04, 0.04, 0.0005]))
    assert np.isfinite(r[0]) and np.isnan(r[1]) and np.isnan(r[2])


def test_stumpf_ratio_rises_when_green_drops_faster_than_blue():
    shallow = F.stumpf_ratio(np.array([0.10]), np.array([0.12]))
    deep = F.stumpf_ratio(np.array([0.04]), np.array([0.03]))
    assert deep[0] > shallow[0]


def test_valid_pixels_requires_blue_and_green():
    refl = {"B02": np.array([0.05, 0.0005]), "B03": np.array([0.05, 0.05])}
    assert F.valid_pixels(refl).tolist() == [True, False]


def test_compute_returns_all_features():
    refl = {b: np.array([0.05]) for b in F.BANDS}
    feats = F.compute(refl, {b: 0.01 for b in F.BANDS})
    assert set(F.FEATURES) <= set(feats)


def test_water_mask_drops_small_components():
    H, W = 60, 60
    mndwi = np.full((H, W), -0.5)
    mndwi[:40, :40] = 0.5          # 1,600 px water body: kept
    mndwi[50:53, 50:53] = 0.5      # 9 px speckle: removed
    bands = {"MNDWI": mndwi, "n_valid": np.full((H, W), 10.0), "B03": np.full((H, W), 0.05)}
    w = F.water_mask(bands, min_valid=3, min_component_px=1000)
    assert w[:40, :40].all() and not w[50:53, 50:53].any()


def test_switch_model_falls_back_to_green_where_red_is_dark():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 15, 500)
    bg = 0.94 + 0.01 * y
    br = 1.0 + 0.15 * np.minimum(y, 5)
    red = np.where(y < 5, 0.05, 0.002)
    X = np.column_stack([bg, br, red])
    m = StumpfSwitch().fit(X, y)
    deep = np.array([[0.94 + 0.01 * 12, np.nan, 0.002]])
    assert abs(m.predict(deep)[0] - m.green.predict([[deep[0, 0]]])[0]) < 1e-9


def test_buffer_removes_close_training_points():
    df = pd.DataFrame({"x": [0, 50, 500, 2000], "y": [0, 0, 0, 0]})
    kept = apply_buffer(df, np.array([1, 2, 3]), np.array([0]), 100)
    assert kept.tolist() == [2, 3]


def test_rgt_folds_hold_out_whole_ground_tracks():
    df = pd.DataFrame({"rgt": [1] * 150 + [2] * 150 + [3] * 5, "pass_id": ["a"] * 305})
    for tr, te in folds(df, "rgt_buffer"):
        assert set(df.rgt.iloc[te]).isdisjoint(set(df.rgt.iloc[tr]))
        assert 3 not in set(df.rgt.iloc[te])  # tiny RGTs stay in training only


def test_metrics_signs():
    m = metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
    assert m["bias"] == 1.0 and m["rmse"] == 1.0 and m["mae"] == 1.0
