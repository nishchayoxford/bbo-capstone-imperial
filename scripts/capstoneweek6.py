#!/usr/bin/env python
# coding: utf-8
import numpy as np



"""
Capstone Week 6 — Student-Friendly Notes (consistent format)

Goal
- Propose Week 6 queries using Weeks 1–5 data.

Method mix
- Strong ML exploitation: F5, F7 (Ridge + richer candidate search).
- Local ML refinement: F2, F3, F4, F8 (Ridge + smaller moves).
- Manual/exploration rules: F1, F6.

Why this mapping?
- F5/F7 had cleaner momentum, so stronger model-based exploitation is justified.
- F2/F3/F4/F8 need safer local updates to avoid overshooting.
- F1 benefits from space-filling exploration (maximin) when signal is weak.
- F6 follows directional heuristic because local trend was reliable.

Output
- Prints one portal-ready query per function (F1..F8).
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


DATA = {
    "F1": {
        "X": np.array([
            [0.145000, 0.515000],  # Week 1 (baseline)
            [0.725000, 0.285000],  # Week 2 (best so far, up vs Week 1)
            [0.515000, 0.515000],  # Week 3 (best so far, up vs Week 2)
            [0.750000, 0.750000],  # Week 4 (worst so far, down vs Week 3)
            [0.990000, 0.010000],  # Week 5 (worst so far, down vs Week 4)
        ], dtype=float),
        "y": np.array([
            -3.353165630322361e-61,
            6.743225602289377e-78,
            4.714509345171323e-13,
            1.3319145509281447e-22,
            0.0,
        ], dtype=float),
    },

    "F2": {
        "X": np.array([
            [0.755000, 0.275000],  # Week 1 (baseline)
            [0.785000, 0.305000],  # Week 2 (worst so far, down vs Week 1)
            [0.740000, 0.260000],  # Week 3 (best so far, up vs Week 2)
            [0.730000, 0.270000],  # Week 4 (best so far, up vs Week 3)
            [0.718763, 0.261649],  # Week 5 (down vs Week 4)
        ], dtype=float),
        "y": np.array([
            0.42044085041824825,
            -0.0456643112924181,
            0.46274019045813003,
            0.6060955609811236,
            0.5195146975906033,
        ], dtype=float),
    },

    "F3": {
        "X": np.array([
            [0.395000, 0.875000, 0.635000],  # Week 1 (baseline)
            [0.145000, 0.395000, 0.915000],  # Week 2 (best so far, up vs Week 1)
            [0.120000, 0.347000, 0.943000],  # Week 3 (worst so far, down vs Week 2)
            [0.155000, 0.385000, 0.905000],  # Week 4 (best so far, up vs Week 3)
            [0.165000, 0.375000, 0.895000],  # Week 5 (best so far, up vs Week 4)
        ], dtype=float),
        "y": np.array([
            -0.12080733985523133,
            -0.11535196594300248,
            -0.20076336857175398,
            -0.07852077254038155,
            -0.06033571734237718,
        ], dtype=float),
    },

    "F4": {
        "X": np.array([
            [0.275000, 0.955000, 0.515000, 0.145000],  # Week 1 (baseline)
            [0.815000, 0.245000, 0.355000, 0.695000],  # Week 2 (best so far, up vs Week 1)
            [0.869000, 0.174000, 0.339000, 0.750000],  # Week 3 (worst so far, down vs Week 2)
            [0.795000, 0.265000, 0.365000, 0.665000],  # Week 4 (best so far, up vs Week 3)
            [0.785000, 0.275000, 0.370000, 0.650000],  # Week 5 (best so far, up vs Week 4)
        ], dtype=float),
        "y": np.array([
            -18.59723490448631,
            -14.395540985679897,
            -18.67377341401988,
            -13.169944884454413,
            -12.699964227491282,
        ], dtype=float),
    },

    "F5": {
        "X": np.array([
            [0.635000, 0.395000, 0.755000, 0.875000],  # Week 1 (baseline)
            [0.665000, 0.365000, 0.785000, 0.845000],  # Week 2 (best so far, up vs Week 1)
            [0.680000, 0.350000, 0.800000, 0.830000],  # Week 3 (best so far, up vs Week 2)
            [0.695000, 0.335000, 0.815000, 0.815000],  # Week 4 (best so far, up vs Week 3)
            [0.707000, 0.323000, 0.827000, 0.803000],  # Week 5 (best so far, up vs Week 4)
        ], dtype=float),
        "y": np.array([
            287.4343816627659,
            292.2593658119571,
            301.5311905557768,
            315.65049985154724,
            330.66611638919255,
        ], dtype=float),
    },

    "F6": {
        "X": np.array([
            [0.515000, 0.145000, 0.955000, 0.395000, 0.755000],  # Week 1 (baseline)
            [0.185000, 0.745000, 0.315000, 0.865000, 0.455000],  # Week 2 (best so far, up vs Week 1)
            [0.152000, 0.805000, 0.251000, 0.912000, 0.425000],  # Week 3 (worst so far, down vs Week 2)
            [0.170000, 0.760000, 0.300000, 0.890000, 0.470000],  # Week 4 (up vs Week 3)
            [0.200000, 0.730000, 0.330000, 0.840000, 0.455000],  # Week 5 (best so far, up vs Week 4)
        ], dtype=float),
        "y": np.array([
            -1.6304531811460896,
            -1.4347679755670883,
            -1.6451191179236977,
            -1.6022183821509282,
            -1.3295280103104827,
        ], dtype=float),
    },

    "F7": {
        "X": np.array([
            [0.875000, 0.275000, 0.635000, 0.515000, 0.145000, 0.955000],  # Week 1 (baseline)
            [0.845000, 0.305000, 0.665000, 0.485000, 0.175000, 0.925000],  # Week 2 (best so far, up vs Week 1)
            [0.830000, 0.320000, 0.680000, 0.470000, 0.190000, 0.910000],  # Week 3 (best so far, up vs Week 2)
            [0.815000, 0.335000, 0.695000, 0.455000, 0.205000, 0.895000],  # Week 4 (best so far, up vs Week 3)
            [0.805202, 0.344798, 0.704798, 0.445202, 0.214798, 0.885202],  # Week 5 (best so far, up vs Week 4)
        ], dtype=float),
        "y": np.array([
            0.6267064847700778,
            0.8069621926499697,
            0.8919314248129555,
            0.969339703275594,
            1.0144420450032012,
        ], dtype=float),
    },

    "F8": {
        "X": np.array([
            [0.145000, 0.275000, 0.395000, 0.515000, 0.635000, 0.755000, 0.875000, 0.955000],  # Week 1 (baseline)
            [0.175000, 0.305000, 0.425000, 0.545000, 0.665000, 0.785000, 0.905000, 0.945000],  # Week 2 (worst so far, down vs Week 1)
            [0.130000, 0.260000, 0.380000, 0.500000, 0.620000, 0.740000, 0.860000, 0.960000],  # Week 3 (best so far, up vs Week 2)
            [0.140000, 0.270000, 0.390000, 0.500000, 0.620000, 0.740000, 0.860000, 0.960000],  # Week 4 (down vs Week 3)
            [0.120000, 0.250000, 0.370000, 0.500000, 0.620000, 0.740000, 0.860000, 0.960000],  # Week 5 (best so far, up vs Week 4)
        ], dtype=float),
        "y": np.array([
            8.633935,
            8.451335,
            8.71814,
            8.69914,
            8.73594,
        ], dtype=float),
    },
}


ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
USE_LOOCV_ALPHA = True

CONFIG = {
    "strong_ml": {
        "F5": {"step": 0.028, "mask": None},
        "F7": {"step": 0.022, "mask": None},
    },
    "local_ml": {
        "F2": {"step": 0.010, "mask": None},          # reduce step (F2 is sensitive)
        "F3": {"step": 0.020, "mask": None},          # moderate step in 3D
        "F4": {"step": 0.015, "mask": None},          # cautious (narrow basin)
        "F8": {"step": 0.010, "mask": [0, 1, 2]},     # only adjust first 3 dims (ridge tracking)
    },
    "manual": {
        "F1": {"mode": "maximin_random", "n_rand": 20000, "seed": 7},
        "F6": {"mode": "continue_last_direction", "gamma": 0.6},
    }
}


def clip_01(x: np.ndarray) -> np.ndarray:
    """Keep values in [0, 0.999999] so they always start with 0."""
    return np.clip(x, 0.0, 0.999999)

def format_query(x: np.ndarray) -> str:
    """Return x as '0.xxxxxx-0.xxxxxx-...' with 6 decimals."""
    return "-".join(f"{v:.6f}" for v in x)

def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float):
    """
    Fit ridge: y_hat = b0 + b^T x.
    Returns (b0, b).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    if SKLEARN_AVAILABLE:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X, y)
        return float(model.intercept_), model.coef_.astype(float)

    n, d = X.shape
    Z = np.hstack([np.ones((n, 1)), X])
    I = np.eye(d + 1)
    I[0, 0] = 0.0
    beta = np.linalg.solve(Z.T @ Z + alpha * I, Z.T @ y)
    return float(beta[0]), beta[1:].astype(float)

def ridge_predict(b0: float, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return b0 + X @ b

def choose_alpha_loocv(X: np.ndarray, y: np.ndarray, alphas):
    """
    Leave-one-out CV for ridge alpha selection.
    For tiny datasets, this is a reasonable heuristic.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = X.shape[0]

    best_alpha = None
    best_mse = np.inf

    for a in alphas:
        errs = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            Xtr, ytr = X[mask], y[mask]
            Xte, yte = X[~mask], y[~mask]

            b0, b = ridge_fit(Xtr, ytr, alpha=a)
            yhat = ridge_predict(b0, b, Xte)[0]
            errs.append((yte[0] - yhat) ** 2)

        mse = float(np.mean(errs))
        if mse < best_mse:
            best_mse = mse
            best_alpha = a

    return best_alpha, best_mse

def masked_direction(b: np.ndarray, mask):
    """
    If mask is provided, only allow movement in those dimensions.
    """
    b = np.asarray(b, float).copy()
    if mask is None:
        return b
    keep = np.zeros_like(b)
    keep[mask] = b[mask]
    return keep

def generate_candidates(x_best, b, step, mask=None, mode="local"):
    """
    Create a set of candidate points around x_best.
    We score them using the ridge surrogate and pick the best predicted.

    mode:
      - "strong": more candidates, slightly more aggressive
      - "local": smaller moves, safer set
    """
    x_best = np.asarray(x_best, float)
    d = x_best.size

    b_eff = masked_direction(b, mask)
    norm = np.linalg.norm(b_eff)

    candidates = []

    if norm < 1e-12:
        base_steps = [step, step * 0.5]
        for s in base_steps:
            for j in range(d):
                if (mask is not None) and (j not in mask):
                    continue
                x1 = x_best.copy()
                x1[j] = x1[j] + s
                candidates.append(x1)
                x2 = x_best.copy()
                x2[j] = x2[j] - s
                candidates.append(x2)
        return [clip_01(c) for c in candidates]

    direction = b_eff / norm

    if mode == "strong":
        step_multipliers = [0.5, 1.0, 1.5]
    else:
        step_multipliers = [0.5, 1.0]

    for m in step_multipliers:
        x_new = x_best + (m * step) * direction
        candidates.append(x_new)

    coord_mult = 1.0 if mode == "strong" else 0.8
    for j in range(d):
        if (mask is not None) and (j not in mask):
            continue
        sgn = np.sign(b_eff[j])
        if sgn == 0:
            continue
        x_new = x_best.copy()
        x_new[j] = x_new[j] + coord_mult * step * sgn
        candidates.append(x_new)

    return [clip_01(c) for c in candidates]

def propose_by_ridge(X, y, step, mask=None, mode="local"):
    """
    Fit ridge, pick x_best, generate candidates, select candidate with max predicted y_hat.
    Returns a dict with everything useful for debugging.
    """
    if USE_LOOCV_ALPHA:
        alpha, mse = choose_alpha_loocv(X, y, ALPHAS)
    else:
        alpha, mse = 1e-2, None

    b0, b = ridge_fit(X, y, alpha=alpha)

    best_idx = int(np.argmax(y))
    x_best = X[best_idx].copy()
    y_best = float(y[best_idx])

    cands = generate_candidates(x_best, b, step=step, mask=mask, mode=mode)
    preds = ridge_predict(b0, b, np.array(cands))

    best_cand_idx = int(np.argmax(preds))
    x_next = cands[best_cand_idx]
    yhat_next = float(preds[best_cand_idx])

    return {
        "alpha": alpha,
        "loocv_mse": mse,
        "b0": b0,
        "b": b,
        "x_best": x_best,
        "y_best": y_best,
        "x_next": x_next,
        "yhat_next": yhat_next,
        "n_candidates": len(cands),
    }


def manual_maximin_random(X_existing, n_rand=20000, seed=0):
    """
    F1 manual exploration:
    - sample many random points
    - pick the point that maximises min distance to existing points (space-filling)
    """
    rng = np.random.default_rng(seed)
    X_existing = np.asarray(X_existing, float)

    R = rng.random((n_rand, X_existing.shape[1])) * 0.999999  # in [0, 0.999999)

    dists = np.sqrt(((R[:, None, :] - X_existing[None, :, :]) ** 2).sum(axis=2))
    min_dist = dists.min(axis=1)

    idx = int(np.argmax(min_dist))
    return clip_01(R[idx])

def manual_continue_last_direction(X, y, gamma=0.6):
    """
    F6 manual:
    - Use the direction from last step.
    - If last move improved, continue in same direction (scaled by gamma).
    - If last move worsened, reverse direction (scaled by gamma).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    x_prev = X[-2].copy()
    x_last = X[-1].copy()
    y_prev = float(y[-2])
    y_last = float(y[-1])

    delta = x_last - x_prev
    if y_last >= y_prev:
        x_new = x_last + gamma * delta
    else:
        x_new = x_last - gamma * delta

    return clip_01(x_new)


week6_plan = {}

for fname, params in CONFIG["strong_ml"].items():
    X, y = DATA[fname]["X"], DATA[fname]["y"]
    res = propose_by_ridge(X, y, step=params["step"], mask=params["mask"], mode="strong")
    week6_plan[fname] = ("STRONG_ML_RIDGE", res)

for fname, params in CONFIG["local_ml"].items():
    X, y = DATA[fname]["X"], DATA[fname]["y"]
    res = propose_by_ridge(X, y, step=params["step"], mask=params["mask"], mode="local")
    week6_plan[fname] = ("LOCAL_ML_RIDGE", res)

f1_params = CONFIG["manual"]["F1"]
x_f1 = manual_maximin_random(DATA["F1"]["X"], n_rand=f1_params["n_rand"], seed=f1_params["seed"])
week6_plan["F1"] = ("MANUAL_MAXIMIN", {"x_next": x_f1})

f6_params = CONFIG["manual"]["F6"]
x_f6 = manual_continue_last_direction(DATA["F6"]["X"], DATA["F6"]["y"], gamma=f6_params["gamma"])
week6_plan["F6"] = ("MANUAL_CONTINUE_DIR", {"x_next": x_f6})


print("==== WEEK 6 QUERY PLAN (PORTAL FORMAT) ====\n")

for key in ["F1","F2","F3","F4","F5","F6","F7","F8"]:
    method, info = week6_plan[key]
    x_next = info["x_next"]

    print(f"{key}  ({method})")
    print(f"  Query: {format_query(x_next)}")

    if "alpha" in info:
        print(f"  alpha: {info['alpha']} | candidates: {info['n_candidates']} | y_best(obs): {info['y_best']:.6f} | yhat(next): {info['yhat_next']:.6f}")
    print()

print("Submit each query string to the matching function field in the portal.")






