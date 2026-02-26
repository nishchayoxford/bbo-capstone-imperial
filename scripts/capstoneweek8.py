#!/usr/bin/env python
# coding: utf-8
import numpy as np
import math
import warnings



"""
Capstone Week 8 — Student-Friendly Notes (consistent format)

Goal
- Propose Week 8 queries using Weeks 1–7 data.

Method mix
- Minimal effort: F1.
- Ridge local tuning: F2, F3, F4, F8.
- Trust-region BO (local-only GP+EI): F5, F7.
- Manual direction continuation: F6.

Why this mapping?
- F5/F7 had evidence that large/global jumps can hurt, so search is restricted locally.
- F2/F3/F4/F8 are refined with small local Ridge-guided moves.
- F1 is deprioritized; lightweight exploration is enough.
- F6 continues to improve under controlled directional updates.

Output
- Prints portal-ready Week 8 queries and concise method diagnostics.
"""

"""
============================================================
Student-Friendly Guide (applies to this script)
============================================================
What this script does
- Uses historical weekly data for black-box optimization functions (F1..F8).
- Builds a next-week query proposal for each function based on the chosen strategy.
- Prints final answers in portal-ready format: `0.xxxxxx-0.xxxxxx-...`

How to read this file
1) DATA section
   - Stores past points X and scores y for each function.
2) Helper section
   - Reusable utilities (clipping, formatting, model helpers).
3) Strategy section(s)
   - Function-specific logic (Ridge / BO / manual / exploration).
4) PLAN section
   - Combines all strategy outputs into one dictionary.
5) Print section
   - Outputs final query strings to submit.

Key ideas (simple)
- Ridge regression: linear surrogate for local improvement direction.
- BO (Bayesian Optimization): GP + EI to balance confidence and upside.
- Manual rule: deterministic heuristic when model-based search is not preferred.

Safety / constraints in all scripts
- Values are clipped to [0, 0.999999] for portal compatibility.
- Rounding/uniqueness checks help avoid duplicate submissions.

How to run
- In notebook: run all cells in order.
- As script: `python capstoneweekX.py`

Tip for classmates
- If you change strategy, keep DATA and output format unchanged so submissions stay valid.
============================================================
"""


try:
    from sklearn.linear_model import Ridge
    SKLEARN_RIDGE_AVAILABLE = True
except ImportError:
    SKLEARN_RIDGE_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
    from sklearn.exceptions import ConvergenceWarning
    SKLEARN_GP_AVAILABLE = True
except ImportError:
    SKLEARN_GP_AVAILABLE = False

if SKLEARN_GP_AVAILABLE:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA = {
    "F1": {
        "X": np.array([
            [0.145000, 0.515000],  # Week 1 (baseline)
            [0.725000, 0.285000],  # Week 2 (best so far, up vs Week 1)
            [0.515000, 0.515000],  # Week 3 (best so far, up vs Week 2)
            [0.750000, 0.750000],  # Week 4 (worst so far, down vs Week 3)
            [0.990000, 0.010000],  # Week 5 (worst so far, down vs Week 4)
            [0.000029, 0.001417],  # Week 6 (worst so far, up vs Week 5)
            [0.305976, 0.997403],  # Week 7 (worst so far, down vs Week 6)
        ], float),
        "y": np.array([
            -3.353165630322361e-61,
            6.743225602289377e-78,
            4.714509345171323e-13,
            1.3319145509281447e-22,
            0.0,
            1.825040909472812e-247,
            -1.5662072753465034e-167,
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
        ], float),
        "y": np.array([
            0.42044085041824825,
            -0.0456643112924181,
            0.46274019045813003,
            0.6060955609811236,
            0.5195146975906033,
            0.5794253005452772,
            0.5796694237276565,
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
        ], float),
        "y": np.array([
            -0.12080733985523133,
            -0.11535196594300248,
            -0.20076336857175398,
            -0.07852077254038155,
            -0.06033571734237718,
            -0.04739292498526722,
            -0.05056402944032541,
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
        ], float),
        "y": np.array([
            -18.59723490448631,
            -14.395540985679897,
            -18.67377341401988,
            -13.169944884454413,
            -12.699964227491282,
            -12.987699814058924,
            -12.94099410856025,
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
        ], float),
        "y": np.array([
            287.4343816627659,
            292.2593658119571,
            301.5311905557768,
            315.65049985154724,
            330.66611638919255,
            365.66328225833024,
            283.75880106841055,
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
        ], float),
        "y": np.array([
            -1.6304531811460896,
            -1.4347679755670883,
            -1.6451191179236977,
            -1.6022183821509282,
            -1.3295280103104827,
            -1.2429202946292475,
            -1.2012624047628697,
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
        ], float),
        "y": np.array([
            0.6267064847700778,
            0.8069621926499697,
            0.8919314248129555,
            0.969339703275594,
            1.0144420450032012,
            1.0679017392374972,
            0.10868500160826922,
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
        ], float),
        "y": np.array([
            8.633935,
            8.451335,
            8.71814,
            8.69914,
            8.73594,
            8.745671245544,
            8.753167873306,
        ], float),
    },
}


def clip_01(x):
    """Clip to [0, 0.999999] so portal always shows values starting with '0.'"""
    return np.clip(np.asarray(x, float), 0.0, 0.999999)

def format_query(x):
    """Portal query string: 6 decimals, hyphen-separated."""
    return "-".join(f"{v:.6f}" for v in np.asarray(x, float))

def min_dist_to_existing(x, X_existing):
    X_existing = np.asarray(X_existing, float)
    d = np.sqrt(((X_existing - x)**2).sum(axis=1))
    return float(d.min())

def ensure_unique_after_rounding(x, X_existing, seed=0, tries=200, jitter=1e-5):
    """
    Because the portal rounds to 6 decimals, two different floats can become identical strings.
    This nudges x slightly if needed to avoid an accidental duplicate submission.
    """
    rng = np.random.default_rng(seed)
    existing_str = {format_query(row) for row in np.asarray(X_existing, float)}
    x = np.asarray(x, float).copy()

    for _ in range(tries):
        if format_query(x) not in existing_str:
            return clip_01(x)
        x = x + rng.normal(0.0, jitter, size=x.shape)
        x = clip_01(x)

    return clip_01(x)


def propose_F1_minimal(X_existing, n_rand=8000, seed=8):
    rng = np.random.default_rng(seed)
    d = X_existing.shape[1]
    R = rng.random((n_rand, d)) * 0.999999
    dists = np.sqrt(((R[:, None, :] - X_existing[None, :, :]) ** 2).sum(axis=2))
    min_d = dists.min(axis=1)
    x = R[int(np.argmax(min_d))]
    return ensure_unique_after_rounding(x, X_existing, seed=seed+100)


def propose_F6_continue_direction(X, y, gamma=0.60):
    X = np.asarray(X, float); y = np.asarray(y, float)
    x_prev, x_last = X[-2].copy(), X[-1].copy()
    y_prev, y_last = float(y[-2]), float(y[-1])
    delta = x_last - x_prev
    x_new = x_last + gamma * delta if y_last >= y_prev else x_last - gamma * delta
    return ensure_unique_after_rounding(clip_01(x_new), X, seed=606)


ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

STEP_GRID = {
    "F2": [0.002, 0.004, 0.006, 0.008],
    "F3": [0.006, 0.010, 0.014, 0.018],
    "F4": [0.003, 0.005, 0.007, 0.009],
    "F8": [0.002, 0.004, 0.006, 0.008],
}

MASK = {"F8": [0, 1, 2]}

def ridge_fit(X, y, alpha):
    X = np.asarray(X, float); y = np.asarray(y, float)
    if not SKLEARN_RIDGE_AVAILABLE:
        n, d = X.shape
        Z = np.hstack([np.ones((n, 1)), X])
        I = np.eye(d + 1); I[0, 0] = 0.0
        beta = np.linalg.solve(Z.T @ Z + alpha * I, Z.T @ y)
        return float(beta[0]), beta[1:].astype(float)

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    return float(model.intercept_), model.coef_.astype(float)

def ridge_predict(b0, b, X):
    return b0 + np.asarray(X, float) @ np.asarray(b, float)

def choose_alpha_loocv(X, y, alphas):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = X.shape[0]
    best_a, best_mse = None, np.inf
    for a in alphas:
        errs = []
        for i in range(n):
            m = np.ones(n, dtype=bool); m[i] = False
            b0, b = ridge_fit(X[m], y[m], alpha=a)
            yhat = ridge_predict(b0, b, X[~m])[0]
            errs.append((y[~m][0] - yhat) ** 2)
        mse = float(np.mean(errs))
        if mse < best_mse:
            best_mse, best_a = mse, a
    return best_a, best_mse

def apply_mask(v, mask):
    v = np.asarray(v, float)
    if mask is None:
        return v
    out = np.zeros_like(v)
    out[mask] = v[mask]
    return out

def ridge_candidates(x_best, b, step, mask=None):
    """
    Generate a small candidate set around x_best:
      - move along +b direction
      - coordinate nudges using sign(b_j)
    """
    x_best = np.asarray(x_best, float)
    b_eff = apply_mask(b, mask)
    norm = np.linalg.norm(b_eff)

    cands = []
    if norm < 1e-12:
        for j in range(x_best.size):
            if mask is not None and j not in mask:
                continue
            for s in [step, 0.5 * step]:
                x1 = x_best.copy(); x1[j] += s; cands.append(x1)
                x2 = x_best.copy(); x2[j] -= s; cands.append(x2)
        return [clip_01(c) for c in cands]

    direction = b_eff / norm

    for m in [0.5, 1.0]:
        cands.append(x_best + (m * step) * direction)

    for j in range(x_best.size):
        if mask is not None and j not in mask:
            continue
        sgn = np.sign(b_eff[j])
        if sgn == 0:
            continue
        xj = x_best.copy()
        xj[j] += 0.8 * step * sgn
        cands.append(xj)

    return [clip_01(c) for c in cands]

def propose_ridge_next(fname, X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)

    best_idx = int(np.argmax(y))
    x_best = X[best_idx].copy()
    y_best = float(y[best_idx])

    y_mean, y_std = float(np.mean(y)), float(np.std(y))
    if y_std < 1e-12:
        y_std = 1.0
    y_z = (y - y_mean) / y_std

    alpha, loocv_mse = choose_alpha_loocv(X, y_z, ALPHAS)

    b0, b = ridge_fit(X, y_z, alpha=alpha)

    mask = MASK.get(fname, None)

    best_score = -np.inf
    best_step = None
    best_x = None

    for step in STEP_GRID[fname]:
        cands = ridge_candidates(x_best, b, step, mask=mask)
        preds = ridge_predict(b0, b, np.array(cands))
        idx = int(np.argmax(preds))
        if float(preds[idx]) > best_score:
            best_score = float(preds[idx])
            best_step = step
            best_x = cands[idx]

    best_x = ensure_unique_after_rounding(best_x, X, seed=hash(fname) % 10000)

    return {
        "x_next": best_x,
        "alpha": alpha,
        "loocv_mse": loocv_mse,
        "step": best_step,
        "x_best": x_best,
        "y_best": y_best,
    }


def stdnorm_pdf(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

def stdnorm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def expected_improvement(mu, sigma, y_best, xi=1e-6):
    """
    EI for maximisation (vectorised).
    Smaller xi => more exploitation.
    """
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)

    ei = np.zeros_like(mu)
    m = sigma > 1e-12
    imp = mu[m] - y_best - xi
    Z = imp / sigma[m]

    Phi = np.array([stdnorm_cdf(z) for z in Z])
    phi = np.array([stdnorm_pdf(z) for z in Z])

    ei[m] = imp * Phi + sigma[m] * phi
    ei[ei < 0] = 0.0
    return ei

def fit_gp(X, y, seed=0):
    if not SKLEARN_GP_AVAILABLE:
        raise ImportError("GaussianProcessRegressor not available. Install scikit-learn to use BO.")
    X = np.asarray(X, float); y = np.asarray(y, float)
    d = X.shape[1]

    kernel = (
        C(1.0, (1e-3, 1e3)) *
        Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=6,
        random_state=seed,
    )
    gp.fit(X, y)
    return gp

def propose_bo_trust_region(fname, X, y, seed=0, xi=1e-6, n_local=50000, local_sigma=0.006, radius_mult=3.0):
    """
    Trust-region BO (LOCAL ONLY):
      1) Fit GP on all (X,y)
      2) Generate ONLY local candidates around best observed point
      3) Enforce hard trust-region: max |x_i - x_best_i| <= radius_mult*local_sigma
      4) Select argmax EI
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    rng = np.random.default_rng(seed)

    best_idx = int(np.argmax(y))
    x_best = X[best_idx].copy()
    y_best = float(y[best_idx])

    gp = fit_gp(X, y, seed=seed)

    d = X.shape[1]
    radius = radius_mult * local_sigma

    CANDS = x_best + rng.normal(0.0, local_sigma, size=(n_local, d))
    CANDS = clip_01(CANDS)

    diff = np.abs(CANDS - x_best[None, :])
    CANDS = CANDS[(diff <= radius).all(axis=1)]

    if CANDS.shape[0] == 0:
        CANDS = clip_01(x_best + rng.normal(0.0, local_sigma, size=(n_local, d)))

    keep = []
    for i in range(CANDS.shape[0]):
        keep.append(min_dist_to_existing(CANDS[i], X) > 1e-6)
    CANDS = CANDS[np.array(keep, dtype=bool)]
    if CANDS.shape[0] == 0:
        CANDS = clip_01(x_best + rng.normal(0.0, local_sigma, size=(n_local, d)))

    mu, sigma = gp.predict(CANDS, return_std=True)
    ei = expected_improvement(mu, sigma, y_best=y_best, xi=xi)

    idx = int(np.argmax(ei))
    x_next = ensure_unique_after_rounding(CANDS[idx], X, seed=seed+999)

    return {
        "x_next": x_next,
        "x_best": x_best,
        "y_best": y_best,
        "kernel_": str(gp.kernel_),
        "xi": xi,
        "local_sigma": local_sigma,
        "radius": radius,
        "best_ei": float(ei[idx]),
        "n_candidates": int(CANDS.shape[0]),
    }


PLAN = {}

PLAN["F1"] = ("MINIMAL_MAXIMIN", {"x_next": propose_F1_minimal(DATA["F1"]["X"], n_rand=8000, seed=8)})

for f in ["F2", "F3", "F4", "F8"]:
    PLAN[f] = ("RIDGE_LOCAL_TUNED", propose_ridge_next(f, DATA[f]["X"], DATA[f]["y"]))

PLAN["F5"] = ("TRUST_REGION_BO_GP_EI", propose_bo_trust_region(
    "F5", DATA["F5"]["X"], DATA["F5"]["y"],
    seed=105, xi=1e-6, n_local=50000, local_sigma=0.006, radius_mult=3.0
))
PLAN["F7"] = ("TRUST_REGION_BO_GP_EI", propose_bo_trust_region(
    "F7", DATA["F7"]["X"], DATA["F7"]["y"],
    seed=107, xi=1e-6, n_local=60000, local_sigma=0.005, radius_mult=3.0
))

PLAN["F6"] = ("MANUAL_CONTINUE_DIR", {"x_next": propose_F6_continue_direction(
    DATA["F6"]["X"], DATA["F6"]["y"], gamma=0.60
)})

print("==== WEEK 8 QUERY PLAN (PORTAL FORMAT) ====\n")

for f in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]:
    method, info = PLAN[f]
    x_next = info["x_next"]

    print(f"{f}  [{method}]")
    print(f"  Query: {format_query(x_next)}")

    if method == "RIDGE_LOCAL_TUNED":
        print(f"  tuned alpha: {info['alpha']} | LOOCV MSE: {info['loocv_mse']:.6f} | tuned step: {info['step']}")
        print(f"  best observed y: {info['y_best']:.6f} at x_best={format_query(info['x_best'])}")

    if method == "TRUST_REGION_BO_GP_EI":
        print(f"  GP kernel: {info['kernel_']}")
        print(f"  EI xi: {info['xi']} | local_sigma: {info['local_sigma']} | radius: {info['radius']}")
        print(f"  candidates kept: {info['n_candidates']} | best EI: {info['best_ei']:.6e}")
        print(f"  best observed y: {info['y_best']:.6f} at x_best={format_query(info['x_best'])}")

    print()

print("Submit each query string to the matching function field in the portal.")






