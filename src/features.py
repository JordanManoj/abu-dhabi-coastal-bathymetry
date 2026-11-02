"""Feature engineering shared by training (sample extraction) and map prediction.

Inputs are the composite bands (glint-corrected reflectance B01, B02, B03, B04, B08).
Features:
  - the 5 reflectances themselves
  - Stumpf log-ratios  ln(n*Ri)/ln(n*Rj), n = 1000:  blue/green, blue/red, coastal/green
  - Lyzenga log-transforms  ln(Ri - Rdeep_i) for B01-B04
No coordinates on purpose: with track-shaped labels, lat/lon would let a model memorise
the depth of a neighbourhood instead of learning the optics (spatial leakage).
"""
import numpy as np
import rasterio

from common import PROCESSED

BANDS = ["B01", "B02", "B03", "B04", "B08"]
N_STUMPF = 1000.0
EPS = 1e-4  # reflectance floor after glint subtraction (keeps logs finite)

FEATURES = BANDS + ["pSDB_bg", "pSDB_br", "pSDB_cg"] + [f"lyz_{b}" for b in BANDS[:4]]


VALID_MIN = 1.0 / N_STUMPF  # ln(n*R) must be > 0, i.e. R > 0.001 (definition of the method)


def stumpf_ratio(ri, rj):
    """ln(n Ri) / ln(n Rj); NaN where either reflectance is <= 1/n (log-ratio undefined).

    Without this rule a few near-zero pixels give ratios of -16 or +19 and wreck the linear
    Stumpf fit (R^2 ~ 0 although Spearman rho with depth is 0.69 / 0.84).
    """
    ok = (ri > VALID_MIN) & (rj > VALID_MIN)
    with np.errstate(all="ignore"):
        r = np.log(N_STUMPF * ri) / np.log(N_STUMPF * rj)
    return np.where(ok, r, np.nan)


def valid_pixels(refl):
    """Pixels where the blue/green log-ratio is defined. Used for ALL models, so every
    method is scored on exactly the same samples."""
    return (refl["B02"] > VALID_MIN) & (refl["B03"] > VALID_MIN)


def compute(refl, r_deep):
    """refl: dict band -> array of glint-corrected reflectance. r_deep: dict band -> float.
    pSDB_br / pSDB_cg can be NaN (red or coastal too dark); tree models handle NaN natively."""
    f = {b: refl[b] for b in BANDS}
    f["pSDB_bg"] = stumpf_ratio(refl["B02"], refl["B03"])
    f["pSDB_br"] = stumpf_ratio(refl["B02"], refl["B04"])
    f["pSDB_cg"] = stumpf_ratio(refl["B01"], refl["B03"])
    for b in BANDS[:4]:
        f[f"lyz_{b}"] = np.log(np.maximum(refl[b] - r_deep[b], EPS))
    return f


def load_composite(name="composite"):
    """Return (bands dict, meta dict) for data/processed/<name>.tif."""
    with rasterio.open(PROCESSED / f"{name}.tif") as src:
        arr = src.read()
        names = list(src.descriptions)
        meta = dict(transform=src.transform, crs=src.crs, profile=src.profile)
    return {n: arr[i] for i, n in enumerate(names)}, meta


def water_mask(bands, min_valid=3, min_component_px=1000):
    """Water = MNDWI > 0, at least `min_valid` clear scenes in the median, and part of a
    connected water body of >= `min_component_px` pixels (0.1 km^2).

    The size filter removes ~60,000 speckles of 'water' on wet sabkha (salt flats) inland,
    which MNDWI alone calls water; it keeps the sea, channels and lagoons (18 bodies).
    """
    from scipy import ndimage
    w = (bands["MNDWI"] > 0) & (bands["n_valid"] >= min_valid) & np.isfinite(bands["B03"])
    lab, _ = ndimage.label(w)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_component_px
    keep[0] = False
    return keep[lab]
