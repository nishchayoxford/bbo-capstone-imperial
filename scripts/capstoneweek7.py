#!/usr/bin/env python
# coding: utf-8
import numpy as np



"""
Capstone Week 7 — Student-Friendly Notes (consistent format)

Goal
- Propose Week 7 queries using Weeks 1–6 data.

Method mix
- Exploration: F1.
- Basic ML (Ridge): F2, F3, F4, F8.
- Advanced ML (GP + EI / BO-style): F5, F7.
- Manual direction rule: F6.

Why this mapping?
- F5/F7 benefited most from nonlinear local modelling, so GP+EI was used.
- F2/F3/F4/F8 were handled with robust linear local surrogates (Ridge).
- F1 kept as exploration to maintain coverage.
- F6 remained a stable manual continuation case.

Output
- Prints portal-ready Week 7 queries and diagnostic details.
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
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
    SKLEARN_GP_AVAILABLE = True
except ImportError:
    SKLEARN_GP_AVAILABLE = False

import math


DATA = {
    "F1": {
        "X": np.array([
            [0.145000, 0.515000],  # Week 1 (baseline)
            [0.725000, 0.285000],  # Week 2 (best so far, up vs Week 1)
            [0.515000, 0.515000],  # Week 3 (best so far, up vs Week 2)
            [0.750000, 0.750000],  # Week 4 (worst so far, down vs Week 3)
            [0.990000, 0.010000],  # Week 5 (worst so far, down vs Week 4)
            [0.000029, 0.001417],  # Week 6 (worst so far, up vs Week 5)
        ], float),
        "y": np.array([
            -3.353165630322361e-61,
            6.743225602289377e-78,
            4.714509345171323e-13,
            1.3319145509281447e-22,
            0.0,
            1.825040909472812e-247,
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
        ], float),
        "y": np.array([
            0.42044085041824825,
            -0.0456643112924181,
            0.46274019045813003,
            0.6060955609811236,
            0.5195146975906033,
            0.5794253005452772,
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
        ], float),
        "y": np.array([
            -0.12080733985523133,
            -0.11535196594300248,
            -0.20076336857175398,
            -0.07852077254038155,
            -0.06033571734237718,
            -0.04739292498526722,
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
        ], float),
        "y": np.array([
            -18.59723490448631,
            -14.395540985679897,
            -18.67377341401988,
            -13.169944884454413,
            -12.699964227491282,
            -12.987699814058924,
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
        ], float),
        "y": np.array([
            287.4343816627659,
            292.2593658119571,
            301.5311905557768,
            315.65049985154724,
            330.66611638919255,
            365.66328225833024,
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
        ], float),
        "y": np.array([
            -1.6304531811460896,
            -1.4347679755670883,
            -1.6451191179236977,
            -1.6022183821509282,
            -1.3295280103104827,
            -1.2429202946292475,
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
        ], float),
        "y": np.array([
            0.6267064847700778,
            0.8069621926499697,
            0.8919314248129555,
            0.969339703275594,
            1.0144420450032012,
            1.0679017392374972,
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
        ], float),
        "y": np.array([
            8.633935,
            8.451335,
            8.71814,
            8.69914,
            8.73594,
            8.745671245544,
        ], float),
    },
}


def clip_01(x):
    return np.clip(np.asarray(x, float), 0.0, 0.999999)

def format_query(x):
    return "-".join(f"{v:.6f}" for v in np.asarray(x, float))

def min_dist_to_existing(x, X_existing):
    X_existing = np.asarray(X_existing, float)
    d = np.sqrt(((X_existing - x)**2).sum(axis=1))
    return float(d.min())


def propose_F1_maximin(X_existing, n_rand=60000, seed=11):
    rng = np.random.default_rng(seed)
    d = X_existing.shape[1]
    R = rng.random((n_rand, d)) * 0.999999
    dists = np.sqrt(((R[:, None, :] - X_existing[None, :, :]) ** 2).sum(axis=2))
    min_d = dists.min(axis=1)
    return clip_01(R[int(np.argmax(min_d))])


def propose_F6_continue_direction(X, y, gamma=0.6):
    X = np.asarray(X, float); y = np.asarray(y, float)
    x_prev, x_last = X[-2].copy(), X[-1].copy()
    y_prev, y_last = float(y[-2]), float(y[-1])
    delta = x_last - x_prev
    x_new = x_last + gamma * delta if y_last >= y_prev else x_last - gamma * delta
    return clip_01(x_new)


ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

STEP_GRID = {
    "F2": [0.003, 0.006, 0.009, 0.012],   # sensitive near optimum
    "F3": [0.008, 0.012, 0.016, 0.020],
    "F4": [0.004, 0.007, 0.010, 0.013],   # narrow basin
    "F8": [0.002, 0.004, 0.006, 0.008],   # ridge-tracking, tiny steps
}

MASK = {"F8": [0, 1, 2]}

def ridge_fit(X, y, alpha):
    X = np.asarray(X, float); y = np.asarray(y, float)

    if not SKLEARN_AVAILABLE:
        n, d = X.shape
        Z = np.hstack([np.ones((n, 1)), X])
        I = np.eye(d + 1); I[0, 0] = 0.0
        beta = np.linalg.solve(Z.T @ Z + alpha * I, Z.T @ y)
        b0 = float(beta[0]); b = beta[1:].astype(float)
        return b0, b

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
            errs.append((y[~m][0] - yhat)**2)
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

def ridge_candidates(x_best, b, step, mask=None, mode="local"):
    x_best = np.asarray(x_best, float)
    b_eff = apply_mask(b, mask)
    norm = np.linalg.norm(b_eff)

    cands = []

    if norm < 1e-12:
        for j in range(x_best.size):
            if mask is not None and j not in mask:
                continue
            for s in [step, 0.5*step]:
                x1 = x_best.copy(); x1[j] += s; cands.append(x1)
                x2 = x_best.copy(); x2[j] -= s; cands.append(x2)
        return [clip_01(c) for c in cands]

    direction = b_eff / norm
    mults = [0.5, 1.0] if mode == "local" else [0.5, 1.0, 1.5]

    for m in mults:
        cands.append(x_best + (m*step)*direction)

    for j in range(x_best.size):
        if mask is not None and j not in mask:
            continue
        sgn = np.sign(b_eff[j])
        if sgn == 0:
            continue
        xj = x_best.copy()
        xj[j] += 0.8*step*sgn
        cands.append(xj)

    return [clip_01(c) for c in cands]

def propose_ridge_next(fname, X, y, step_grid):
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

    for step in step_grid:
        cands = ridge_candidates(x_best, b, step, mask=mask, mode="local")
        preds = ridge_predict(b0, b, np.array(cands))
        idx = int(np.argmax(preds))
        if float(preds[idx]) > best_score:
            best_score = float(preds[idx])
            best_step = step
            best_x = cands[idx]

    return {
        "x_next": best_x,
        "alpha": alpha,
        "loocv_mse": loocv_mse,
        "step": best_step,
        "x_best": x_best,
        "y_best": y_best,
    }


def stdnorm_pdf(z):
    return np.exp(-0.5*z*z) / math.sqrt(2.0*math.pi)

def stdnorm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def expected_improvement(mu, sigma, y_best, xi=1e-6):
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    ei = np.zeros_like(mu)

    mask = sigma > 1e-12
    imp = mu[mask] - y_best - xi
    Z = imp / sigma[mask]

    Phi = np.array([stdnorm_cdf(float(z)) for z in Z])
    phi = stdnorm_pdf(Z)

    ei[mask] = imp * Phi + sigma[mask] * phi
    ei[ei < 0] = 0.0
    return ei

def fit_gp_stable(X, y, seed=0):
    if not SKLEARN_GP_AVAILABLE:
        raise ImportError("scikit-learn GaussianProcessRegressor is not available.")

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    d = X.shape[1]

    kernel = (
        C(1.0, (1e-2, 1e3)) *
        Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 10.0), nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,           # we standardise y ourselves
        n_restarts_optimizer=12,
        random_state=seed,
        optimizer="fmin_l_bfgs_b",
    )

    gp._optimizer_kwargs = {"maxiter": 5000}

    gp.fit(X, y)
    return gp

def propose_bo_trust_region(fname, X, y, seed=0, xi=1e-6, n_local=30000, n_global=2000, local_sigma=0.02):
    """
    Trust-region BO:
      - Fit GP on standardised y
      - Sample mostly locally around x_best (strong exploitation)
      - Keep small global sample set to avoid pathological traps
      - Select point that maximises EI
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    rng = np.random.default_rng(seed)

    best_idx = int(np.argmax(y))
    x_best = X[best_idx].copy()
    y_best = float(y[best_idx])

    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std < 1e-12:
        y_std = 1.0
    y_z = (y - y_mean) / y_std
    y_best_z = float(np.max(y_z))

    gp = fit_gp_stable(X, y_z, seed=seed)

    d = X.shape[1]

    L = x_best + rng.normal(0.0, local_sigma, size=(n_local, d))
    L = clip_01(L)

    G = rng.random((n_global, d)) * 0.999999

    CANDS = np.vstack([L, G])

    keep = []
    for i in range(CANDS.shape[0]):
        keep.append(min_dist_to_existing(CANDS[i], X) > 1e-6)
    CANDS = CANDS[np.array(keep, dtype=bool)]
    if CANDS.shape[0] == 0:
        CANDS = rng.random((3000, d)) * 0.999999

    mu, sigma = gp.predict(CANDS, return_std=True)
    ei = expected_improvement(mu, sigma, y_best=y_best_z, xi=xi)

    idx = int(np.argmax(ei))
    x_next = clip_01(CANDS[idx])

    return {
        "x_next": x_next,
        "x_best": x_best,
        "y_best": y_best,
        "kernel_": str(gp.kernel_),
        "xi": xi,
        "local_sigma": local_sigma,
        "n_candidates": int(CANDS.shape[0]),
        "best_ei": float(ei[idx]),
    }


PLAN = {}

PLAN["F1"] = ("EXPLORATION_MAXIMIN", {"x_next": propose_F1_maximin(DATA["F1"]["X"], n_rand=60000, seed=11)})

for f in ["F2", "F3", "F4", "F8"]:
    PLAN[f] = ("BASIC_ML_RIDGE_TUNED", propose_ridge_next(f, DATA[f]["X"], DATA[f]["y"], step_grid=STEP_GRID[f]))

PLAN["F5"] = ("ADV_ML_BO_GP_EI_TR", propose_bo_trust_region("F5", DATA["F5"]["X"], DATA["F5"]["y"], seed=21, xi=1e-6, local_sigma=0.02))
PLAN["F7"] = ("ADV_ML_BO_GP_EI_TR", propose_bo_trust_region("F7", DATA["F7"]["X"], DATA["F7"]["y"], seed=22, xi=1e-6, local_sigma=0.02))

PLAN["F6"] = ("MANUAL_CONTINUE_DIR", {"x_next": propose_F6_continue_direction(DATA["F6"]["X"], DATA["F6"]["y"], gamma=0.60)})


print("==== WEEK 7 QUERY PLAN (PORTAL FORMAT) ====\n")

for f in ["F1","F2","F3","F4","F5","F6","F7","F8"]:
    method, info = PLAN[f]
    x_next = info["x_next"]

    print(f"{f}  [{method}]")
    print(f"  Query: {format_query(x_next)}")

    if method == "BASIC_ML_RIDGE_TUNED":
        print(f"  tuned alpha: {info['alpha']} | LOOCV MSE: {info['loocv_mse']:.6f} | tuned step: {info['step']}")
        print(f"  best observed y: {info['y_best']:.6f} at x_best={format_query(info['x_best'])}")

    if method == "ADV_ML_BO_GP_EI_TR":
        print(f"  GP kernel: {info['kernel_']}")
        print(f"  EI xi: {info['xi']} | local_sigma: {info['local_sigma']} | candidates: {info['n_candidates']} | best EI: {info['best_ei']:.6e}")
        print(f"  best observed y: {info['y_best']:.6f} at x_best={format_query(info['x_best'])}")

    print()

print("Submit each query string to the matching function field in the portal.")






