#!/usr/bin/env python
# coding: utf-8

import math
import warnings
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

"""
Capstone Week 10 — Student-Friendly Notes (same writing style as Weeks 5–9)

What this script is for
- Build Week 10 query proposals for F1..F8 using Weeks 1–9 observations.
- Produce one portal-ready point per function.

How to read this file
1) DATA section: historical points X and scores y.
2) Helper section: clipping, formatting, uniqueness, EI math.
3) GP section: local surrogate fit (Matérn Gaussian Process).
4) Strategy section: trust-region micro-search (TuRBO-lite style).
5) PLAN/output section: final Week 10 queries + diagnostics.

Key ideas
- Local BO around the incumbent instead of large global jumps.
- Expected Improvement (EI) for candidate ranking.
- Nearest-neighbor local training set for a stable local GP.
- Deterministic seeds for reproducible proposals.

Safety and constraints
- Coordinates are clipped to [0, 0.999999] for portal compatibility.
- Duplicate submissions are avoided after 6-decimal rounding.
- Candidate-set fallbacks are used if filtering becomes too strict.

How to run
- python /home/nish/anaconda3/capstoneweek10.py
- Or run cells in notebook order.

Tips for classmates
- Keep DATA shape and final print format unchanged.
- Tune local_sigma / n_local only when comparing exploration strength.
- Keep architecture fixed if you want fair week-to-week comparison.
"""

np.set_printoptions(suppress=True, precision=6)

# ============================================================
# DATA: Week 1–9 history used for Week 10 planning
# ============================================================
DATA = {
    "F1": {
        "X": np.array([
            [0.145000, 0.515000],  # Week 1 (baseline)
            [0.725000, 0.285000],  # Week 2 (best so far, up vs Week 1)
            [0.515000, 0.515000],  # Week 3 (best so far, up vs Week 2)
            [0.750000, 0.750000],  # Week 4 (down vs Week 3)
            [0.990000, 0.010000],  # Week 5 (down vs Week 4)
            [0.000029, 0.001417],  # Week 6 (up vs Week 5)
            [0.305976, 0.997403],  # Week 7 (down vs Week 6)
            [0.422868, 0.002773],  # Week 8 (down vs Week 7)
            [0.005626, 0.830621],  # Week 9 (up vs Week 8)
        ], float),
        "y": np.array([
            -3.353165630322361e-61,
            6.743225602289377e-78,
            4.714509345171323e-13,
            1.3319145509281447e-22,
            0.0,
            1.825040909472812e-247,
            -1.5662072753465034e-167,
            -7.806084086345555e-123,
            1.5539262084660508e-237,
        ], float),
    },

    "F2": {
        "X": np.array([
            [0.755000, 0.275000],  # Week 1 (baseline)
            [0.785000, 0.305000],  # Week 2 (worst so far, down vs Week 1)
            [0.740000, 0.260000],  # Week 3 (best so far, up vs Week 2)
            [0.730000, 0.270000],  # Week 4 (best so far, up vs Week 3)
            [0.718763, 0.261649],  # Week 5 (down vs Week 4)
            [0.722018, 0.263976],  # Week 6 (up vs Week 5)
            [0.721323, 0.261711],  # Week 7 (up vs Week 6)
            [0.724285, 0.264402],  # Week 8 (best so far, up vs Week 7)
            [0.722773, 0.263092],  # Week 9 (down vs Week 8)
        ], float),
        "y": np.array([
            0.42044085041824825,
            -0.0456643112924181,
            0.46274019045813003,
            0.6060955609811236,
            0.5195146975906033,
            0.5794253005452772,
            0.5796694237276565,
            0.6272586156230583,
            0.45895771213691383,
        ], float),
    },

    "F3": {
        "X": np.array([
            [0.395000, 0.875000, 0.635000],  # Week 1 (baseline)
            [0.145000, 0.395000, 0.915000],  # Week 2 (best so far, up vs Week 1)
            [0.120000, 0.347000, 0.943000],  # Week 3 (worst so far, down vs Week 2)
            [0.155000, 0.385000, 0.905000],  # Week 4 (best so far, up vs Week 3)
            [0.165000, 0.375000, 0.895000],  # Week 5 (best so far, up vs Week 4)
            [0.178771, 0.372140, 0.880781],  # Week 6 (best so far, up vs Week 5)
            [0.184441, 0.353663, 0.875638],  # Week 7 (down vs Week 6)
            [0.181867, 0.354586, 0.878279],  # Week 8 (up vs Week 7)
            [0.179573, 0.365715, 0.880210],  # Week 9 (best so far, up vs Week 8)
        ], float),
        "y": np.array([
            -0.12080733985523133,
            -0.11535196594300248,
            -0.20076336857175398,
            -0.07852077254038155,
            -0.06033571734237718,
            -0.04739292498526722,
            -0.05056402944032541,
            -0.0478844185459012,
            -0.04178386362696305,
        ], float),
    },

    "F4": {
        "X": np.array([
            [0.275000, 0.955000, 0.515000, 0.145000],  # Week 1 (baseline)
            [0.815000, 0.245000, 0.355000, 0.695000],  # Week 2 (best so far, up vs Week 1)
            [0.869000, 0.174000, 0.339000, 0.750000],  # Week 3 (worst so far, down vs Week 2)
            [0.795000, 0.265000, 0.365000, 0.665000],  # Week 4 (best so far, up vs Week 3)
            [0.785000, 0.275000, 0.370000, 0.650000],  # Week 5 (best so far, up vs Week 4)
            [0.792676, 0.264502, 0.367988, 0.657198],  # Week 6 (down vs Week 5)
            [0.791656, 0.265832, 0.368297, 0.656143],  # Week 7 (up vs Week 6)
            [0.789610, 0.268623, 0.368839, 0.654212],  # Week 8 (up vs Week 7)
            [0.787305, 0.271802, 0.369425, 0.652092],  # Week 9 (up vs Week 8)
        ], float),
        "y": np.array([
            -18.59723490448631,
            -14.395540985679897,
            -18.67377341401988,
            -13.169944884454413,
            -12.699964227491282,
            -12.987699814058924,
            -12.94099410856025,
            -12.85705507882481,
            -12.773487280801856,
        ], float),
    },

    "F5": {
        "X": np.array([
            [0.635000, 0.395000, 0.755000, 0.875000],  # Week 1 (baseline)
            [0.665000, 0.365000, 0.785000, 0.845000],  # Week 2 (best so far, up vs Week 1)
            [0.680000, 0.350000, 0.800000, 0.830000],  # Week 3 (best so far, up vs Week 2)
            [0.695000, 0.335000, 0.815000, 0.815000],  # Week 4 (best so far, up vs Week 3)
            [0.707000, 0.323000, 0.827000, 0.803000],  # Week 5 (best so far, up vs Week 4)
            [0.728000, 0.302000, 0.848000, 0.782000],  # Week 6 (best so far, up vs Week 5)
            [0.591139, 0.057257, 0.976087, 0.523586],  # Week 7 (worst so far, down vs Week 6)
            [0.745989, 0.305287, 0.849251, 0.788893],  # Week 8 (best so far, up vs Week 7)
            [0.756823, 0.306295, 0.854453, 0.799818],  # Week 9 (best so far, up vs Week 8)
        ], float),
        "y": np.array([
            287.4343816627659,
            292.2593658119571,
            301.5311905557768,
            315.65049985154724,
            330.66611638919255,
            365.66328225833024,
            283.75880106841055,
            413.12789189378645,
            472.01213995906096,
        ], float),
    },

    "F6": {
        "X": np.array([
            [0.515000, 0.145000, 0.955000, 0.395000, 0.755000],  # Week 1 (baseline)
            [0.185000, 0.745000, 0.315000, 0.865000, 0.455000],  # Week 2 (best so far, up vs Week 1)
            [0.152000, 0.805000, 0.251000, 0.912000, 0.425000],  # Week 3 (worst so far, down vs Week 2)
            [0.170000, 0.760000, 0.300000, 0.890000, 0.470000],  # Week 4 (up vs Week 3)
            [0.200000, 0.730000, 0.330000, 0.840000, 0.455000],  # Week 5 (best so far, up vs Week 4)
            [0.218000, 0.712000, 0.348000, 0.810000, 0.446000],  # Week 6 (best so far, up vs Week 5)
            [0.228800, 0.701200, 0.358800, 0.792000, 0.440600],  # Week 7 (best so far, up vs Week 6)
            [0.235280, 0.694720, 0.365280, 0.781200, 0.437360],  # Week 8 (best so far, up vs Week 7)
            [0.238844, 0.691156, 0.368844, 0.775260, 0.435578],  # Week 9 (down vs Week 8)
        ], float),
        "y": np.array([
            -1.6304531811460896,
            -1.4347679755670883,
            -1.6451191179236977,
            -1.6022183821509282,
            -1.3295280103104827,
            -1.2429202946292475,
            -1.2012624047628697,
            -1.1222958408999941,
            -1.1753434539633536,
        ], float),
    },

    "F7": {
        "X": np.array([
            [0.875000, 0.275000, 0.635000, 0.515000, 0.145000, 0.955000],  # Week 1 (baseline)
            [0.845000, 0.305000, 0.665000, 0.485000, 0.175000, 0.925000],  # Week 2 (best so far, up vs Week 1)
            [0.830000, 0.320000, 0.680000, 0.470000, 0.190000, 0.910000],  # Week 3 (best so far, up vs Week 2)
            [0.815000, 0.335000, 0.695000, 0.455000, 0.205000, 0.895000],  # Week 4 (best so far, up vs Week 3)
            [0.805202, 0.344798, 0.704798, 0.445202, 0.214798, 0.885202],  # Week 5 (best so far, up vs Week 4)
            [0.791730, 0.358270, 0.718270, 0.431730, 0.228270, 0.871730],  # Week 6 (best so far, up vs Week 5)
            [0.013373, 0.928169, 0.299072, 0.839656, 0.777563, 0.029987],  # Week 7 (worst so far, down vs Week 6)
            [0.789513, 0.359715, 0.716440, 0.416791, 0.231308, 0.873588],  # Week 8 (best so far, up vs Week 7)
            [0.790638, 0.361606, 0.715460, 0.410503, 0.238615, 0.878998],  # Week 9 (best so far, up vs Week 8)
        ], float),
        "y": np.array([
            0.6267064847700778,
            0.8069621926499697,
            0.8919314248129555,
            0.969339703275594,
            1.0144420450032012,
            1.0679017392374972,
            0.10868500160826922,
            1.0862632473367084,
            1.0911035659970207,
        ], float),
    },

    "F8": {
        "X": np.array([
            [0.145000, 0.275000, 0.395000, 0.515000, 0.635000, 0.755000, 0.875000, 0.955000],  # Week 1 (baseline)
            [0.175000, 0.305000, 0.425000, 0.545000, 0.665000, 0.785000, 0.905000, 0.945000],  # Week 2 (worst so far, down vs Week 1)
            [0.130000, 0.260000, 0.380000, 0.500000, 0.620000, 0.740000, 0.860000, 0.960000],  # Week 3 (best so far, up vs Week 2)
            [0.140000, 0.270000, 0.390000, 0.500000, 0.620000, 0.740000, 0.860000, 0.960000],  # Week 4 (down vs Week 3)
            [0.120000, 0.250000, 0.370000, 0.500000, 0.620000, 0.740000, 0.860000, 0.960000],  # Week 5 (best so far, up vs Week 4)
            [0.114226, 0.244226, 0.364226, 0.500000, 0.620000, 0.740000, 0.860000, 0.960000],  # Week 6 (best so far, up vs Week 5)
            [0.109607, 0.239607, 0.359607, 0.500000, 0.620000, 0.740000, 0.860000, 0.960000],  # Week 7 (best so far, up vs Week 6)
            [0.104988, 0.234988, 0.354988, 0.500000, 0.620000, 0.740000, 0.860000, 0.960000],  # Week 8 (best so far, up vs Week 7)
            [0.103545, 0.233545, 0.353545, 0.500000, 0.620000, 0.740000, 0.860000, 0.960000],  # Week 9 (best so far, up vs Week 8)
        ], float),
        "y": np.array([
            8.633935,
            8.451335,
            8.71814,
            8.69914,
            8.73594,
            8.745671245544,
            8.753167873306,
            8.760408479136,
            8.76261799785,
        ], float),
    },

}

# ============================================================
# Helper utilities
# ============================================================

def clip_01(x: np.ndarray) -> np.ndarray:
    """Clip to portal-safe range [0, 0.999999]."""
    return np.clip(np.asarray(x, float), 0.0, 0.999999)

def format_query(x: np.ndarray) -> str:
    """Format vector as portal string: 0.xxxxxx-0.xxxxxx-..."""
    return "-".join(f"{v:.6f}" for v in np.asarray(x, float))

def ensure_unique_after_rounding(
    x: np.ndarray,
    X_existing: np.ndarray,
    seed: int = 0,
    tries: int = 200,
    jitter: float = 2e-6,
) -> np.ndarray:
    """Add tiny jitter until rounded query is unique versus existing X."""
    rng = np.random.default_rng(seed)
    existing_str = {format_query(row) for row in np.asarray(X_existing, float)}
    x = np.asarray(x, float).copy()
    for _ in range(tries):
        if format_query(x) not in existing_str:
            return clip_01(x)
        x = clip_01(x + rng.normal(0.0, jitter, size=x.shape))
    return clip_01(x)

def stdnorm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

def stdnorm_cdf(z: np.ndarray) -> np.ndarray:
    # vectorized via erf
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 1e-8) -> np.ndarray:
    """EI for maximization, with numerical guards."""
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    ei = np.zeros_like(mu)

    mask = sigma > 1e-12
    imp = mu[mask] - y_best - xi
    Z = imp / sigma[mask]

    Phi = stdnorm_cdf(Z)
    phi = stdnorm_pdf(Z)

    ei[mask] = imp * Phi + sigma[mask] * phi
    ei[ei < 0.0] = 0.0
    return ei

def fit_gp_matern(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    nu: float = 2.5,
    lengthscale_bounds=(1e-2, 1e2),
    noise_bounds=(1e-12, 1e-2),
    constant_bounds=(1e-3, 1e3),
    n_restarts: int = 6,
    normalize_y: bool = True,
) -> GaussianProcessRegressor:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    d = X.shape[1]

    kernel = (
        C(1.0, constant_bounds)
        * Matern(length_scale=np.ones(d), length_scale_bounds=lengthscale_bounds, nu=nu)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=noise_bounds)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=normalize_y,
        n_restarts_optimizer=n_restarts,
        random_state=seed,
        alpha=1e-10,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)
    return gp

def propose_trust_region_micro(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    xi: float = 1e-8,
    n_local: int = 80000,
    local_sigma: float = 0.003,
    radius_mult: float = 2.5,
    k_local_train: int = 8,
    mask=None,
    fixed_values=None,
):
    """Single-step TuRBO-lite / local BO.

    1) Center trust region at the best observed x.
    2) Fit a GP on the K nearest points to that incumbent (local surrogate).
    3) Sample many candidates inside the trust region.
    4) Pick argmax EI (maximization).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)

    best_idx = int(np.argmax(y))
    x_best = X[best_idx].copy()
    y_best = float(y[best_idx])

    # Local training set: K nearest points to x_best
    dists = np.sqrt(((X - x_best) ** 2).sum(axis=1))
    k = min(int(k_local_train), X.shape[0])
    idxs = np.argsort(dists)[:k]
    X_train = X[idxs]
    y_train = y[idxs]

    gp = fit_gp_matern(X_train, y_train, seed=seed, n_restarts=6, normalize_y=True)

    d = X.shape[1]
    radius = radius_mult * local_sigma

    # Local candidate set
    CANDS = x_best + rng.normal(0.0, local_sigma, size=(n_local, d))
    CANDS = clip_01(CANDS)

    # Hard trust-region (per-coordinate)
    diff = np.abs(CANDS - x_best[None, :])
    CANDS = CANDS[(diff <= radius).all(axis=1)]
    if CANDS.shape[0] == 0:
        CANDS = clip_01(x_best + rng.normal(0.0, local_sigma, size=(n_local, d)))

    # Effective-dimension mask (SAAS-style heuristic)
    if mask is not None:
        mask = list(mask)
        for j in range(d):
            if j not in mask:
                CANDS[:, j] = fixed_values[j] if fixed_values and j in fixed_values else x_best[j]

    if fixed_values is not None:
        for j, val in fixed_values.items():
            CANDS[:, j] = val

    CANDS = clip_01(CANDS)

    # Remove duplicates (after rounding) by distance to existing points
    dist2 = ((CANDS[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    min_d = np.sqrt(dist2.min(axis=1))
    CANDS = CANDS[min_d > 1e-6]
    if CANDS.shape[0] == 0:
        CANDS = clip_01(x_best + rng.normal(0.0, local_sigma, size=(max(2000, n_local // 10), d)))

    mu, sigma = gp.predict(CANDS, return_std=True)
    ei = expected_improvement(mu, sigma, y_best=y_best, xi=xi)

    idx = int(np.argmax(ei))
    x_next = ensure_unique_after_rounding(CANDS[idx], X, seed=seed + 999, jitter=2e-6)

    info = {
        "x_next": x_next,
        "x_best": x_best,
        "y_best": y_best,
        "kernel": str(gp.kernel_),
        "best_ei": float(ei[idx]),
        "n_candidates": int(CANDS.shape[0]),
        "radius": radius,
        "local_sigma": local_sigma,
        "k_local_train": k,
    }
    return info

# === Week 10 strategy per function ===
# (feel free to tweak sigmas if you want “more exploration”)

PARAMS = {
    "F1": dict(method="INCUMBENT_NEARBY"),
    "F2": dict(method="TR_BO_NOISY_2D", local_sigma=0.0009, n_local=80000),
    "F3": dict(method="TR_BO_3D",       local_sigma=0.0045, n_local=90000),
    "F4": dict(method="TuRBO_LITE_4D",  local_sigma=0.0035, n_local=110000),
    "F5": dict(method="TuRBO_LITE_4D",  local_sigma=0.0040, n_local=120000),
    "F6": dict(method="TuRBO_LITE_5D",  local_sigma=0.0030, n_local=130000),
    "F7": dict(method="TuRBO_LITE_6D",  local_sigma=0.0030, n_local=150000),
    "F8": dict(method="LOW_EFF_DIM_TR_BO", local_sigma=0.0015, n_local=120000, mask=[0,1,2],
               fixed_values={3:0.5,4:0.62,5:0.74,6:0.86,7:0.96}),
}

PLAN = {}

# F1: incumbent / near-incumbent (jittered to avoid duplicate rounding)
X1, y1 = DATA["F1"]["X"], DATA["F1"]["y"]
best_idx = int(np.argmax(y1))
x_best = X1[best_idx].copy()
PLAN["F1"] = ensure_unique_after_rounding(x_best, X1, seed=201, jitter=5e-6)

# F2–F8: TuRBO-lite single step
INFO = {}
for f in ["F2","F3","F4","F5","F6","F7","F8"]:
    p = PARAMS[f].copy()
    method = p.pop("method")
    INFO[f] = propose_trust_region_micro(
        DATA[f]["X"],
        DATA[f]["y"],
        seed=200 + int(f[1:]),   # deterministic per-function
        xi=1e-8,
        n_local=p.pop("n_local"),
        local_sigma=p.pop("local_sigma"),
        radius_mult=2.5,
        k_local_train=8,
        mask=p.pop("mask", None),
        fixed_values=p.pop("fixed_values", None),
    )
    PLAN[f] = INFO[f]["x_next"]

print("==== WEEK 10 QUERY PLAN (PORTAL FORMAT) ====")
for f in ["F1","F2","F3","F4","F5","F6","F7","F8"]:
    print(f"{f}: {format_query(PLAN[f])}")

# Optional: diagnostics (incumbent, best y, GP kernel, EI at chosen point)
for f in ["F2","F3","F4","F5","F6","F7","F8"]:
    info = INFO[f]
    print(f"\n{f}  [{PARAMS[f]['method']}]")
    print("  x_best:", format_query(info["x_best"]), "| y_best:", info["y_best"])
    print("  x_next:", format_query(info["x_next"]), "| EI:", f"{info['best_ei']:.3e}")
    print("  kernel:", info["kernel"])



