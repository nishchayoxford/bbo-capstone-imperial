#!/usr/bin/env python
# coding: utf-8
import numpy as np



"""
Capstone Week 5 — Student-Friendly Notes (consistent format)

Goal
- Propose Week 5 query points using only Weeks 1–4 observations.

Method used
- Ridge regression surrogate for selected functions: F2, F5, F7.

Why this method here?
- At Week 5, each function has very few data points.
- Ridge is more stable than plain linear regression on tiny datasets.
- The fitted coefficient vector gives a practical local direction for improvement.

Function coverage
- F2: Ridge (small step because function is sensitive).
- F5: Ridge (moderate step; trend was smooth/improving).
- F7: Ridge (moderate-cautious step due to higher dimension).

Output
- Prints portal-ready query strings with 6 decimals.
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
    "F2": {
        "X": np.array([
            [0.755000, 0.275000],  # Week 1 (baseline)
            [0.785000, 0.305000],  # Week 2 (worst so far, down vs Week 1)
            [0.740000, 0.260000],  # Week 3 (best so far, up vs Week 2)
            [0.730000, 0.270000],  # Week 4 (best so far, up vs Week 3)
        ], dtype=float),
        "y": np.array([
            0.42044085041824825,
            -0.0456643112924181,
            0.46274019045813003,
            0.6060955609811236,
        ], dtype=float),
        "step": 0.014,
    },

    "F5": {
        "X": np.array([
            [0.635000, 0.395000, 0.755000, 0.875000],  # Week 1 (baseline)
            [0.665000, 0.365000, 0.785000, 0.845000],  # Week 2 (best so far, up vs Week 1)
            [0.680000, 0.350000, 0.800000, 0.830000],  # Week 3 (best so far, up vs Week 2)
            [0.695000, 0.335000, 0.815000, 0.815000],  # Week 4 (best so far, up vs Week 3)
        ], dtype=float),
        "y": np.array([
            287.4343816627659,
            292.2593658119571,
            301.5311905557768,
            315.65049985154724,
        ], dtype=float),
        "step": 0.024,
    },

    "F7": {
        "X": np.array([
            [0.875000, 0.275000, 0.635000, 0.515000, 0.145000, 0.955000],  # Week 1 (baseline)
            [0.845000, 0.305000, 0.665000, 0.485000, 0.175000, 0.925000],  # Week 2 (best so far, up vs Week 1)
            [0.830000, 0.320000, 0.680000, 0.470000, 0.190000, 0.910000],  # Week 3 (best so far, up vs Week 2)
            [0.815000, 0.335000, 0.695000, 0.455000, 0.205000, 0.895000],  # Week 4 (best so far, up vs Week 3)
        ], dtype=float),
        "y": np.array([
            0.6267064847700778,
            0.8069621926499697,
            0.8919314248129555,
            0.969339703275594,
        ], dtype=float),
        "step": 0.024,
    },
}


def format_query(x: np.ndarray) -> str:
    """
    Portal format: '0.xxxxxx-0.xxxxxx-...'
    """
    return "-".join(f"{v:.6f}" for v in x)


def clip_01(x: np.ndarray) -> np.ndarray:
    """
    Keep values within [0, 0.999999] so they always start with '0.' and remain valid.
    (The portal expects values starting with 0 and 6 decimals.)
    """
    return np.clip(x, 0.0, 0.999999)


def fit_ridge_sklearn(X: np.ndarray, y: np.ndarray, alpha: float):
    """
    Fit Ridge regression using scikit-learn.

    The model is:
        y_hat(x) = b0 + b^T x

    scikit-learn solves:
        min Σ (y_i - y_hat(x_i))^2 + alpha * ||b||^2

    Returns:
        b0: intercept
        b : coefficient vector
    """
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    return float(model.intercept_), model.coef_.astype(float)


def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float):
    """
    Closed-form Ridge regression (no sklearn required).

    We rewrite the linear model with an intercept using an augmented matrix:
        Z = [1, X]   (prepend a column of ones)

    Then solve:
        beta = (Z^T Z + alpha * I)^(-1) Z^T y

    Important detail:
        We do NOT regularise the intercept.
        So we set I[0,0] = 0 (no penalty on intercept term).

    Returns:
        b0: intercept
        b : coefficient vector
    """
    n, d = X.shape
    Z = np.hstack([np.ones((n, 1)), X])  # add intercept column
    I = np.eye(d + 1)
    I[0, 0] = 0.0  # do NOT regularise intercept

    beta = np.linalg.solve(Z.T @ Z + alpha * I, Z.T @ y)
    b0 = float(beta[0])
    b = beta[1:].astype(float)
    return b0, b


def propose_week5(X: np.ndarray, y: np.ndarray, alpha: float, step: float):
    """
    Core algorithm to propose Week 5 query:

    Step 1) Fit Ridge regression surrogate:
        y_hat(x) = b0 + b^T x

    Step 2) Pick the best observed point so far:
        x_best = argmax_y observed (x_i)

    Step 3) Move a small step in the direction that increases y_hat:
        For a linear model, gradient wrt x is:
            ∇_x y_hat = b
        So an "uphill" move is along +b.

        We normalise b so 'step' has consistent meaning:
            direction = b / ||b||
            x_new = x_best + step * direction

    Why normalise?
        b's magnitude depends on scaling and regularisation.
        Normalising makes step size comparable across functions.

    Safety:
        If ||b|| is ~0, the model is essentially flat -> no reliable direction.
        Then we keep x_new = x_best.

    Returns:
        x_best, y_best, b0, b, x_new
    """
    if SKLEARN_AVAILABLE:
        b0, b = fit_ridge_sklearn(X, y, alpha=alpha)
    else:
        b0, b = fit_ridge_closed_form(X, y, alpha=alpha)

    best_idx = int(np.argmax(y))
    x_best = X[best_idx].copy()
    y_best = float(y[best_idx])

    norm = np.linalg.norm(b)
    if norm < 1e-12:
        x_new = x_best
        direction = np.zeros_like(b)
    else:
        direction = b / norm
        x_new = x_best + step * direction

    x_new = clip_01(x_new)
    return x_best, y_best, b0, b, direction, x_new


def main():
    alpha = 1e-2

    print("==== WEEK 5 QUERY PLAN (PORTAL FORMAT) ====\n")
    print(f"Using sklearn: {SKLEARN_AVAILABLE}")
    print(f"Ridge alpha = {alpha}\n")

    for fname, d in DATA.items():
        X, y, step = d["X"], d["y"], d["step"]

        x_best, y_best, b0, b, direction, x_new = propose_week5(X, y, alpha=alpha, step=step)

        print(f"{fname}")
        print(f"  Best-so-far x : {format_query(x_best)}   y: {y_best:.6f}")
        print(f"  Ridge intercept b0: {b0:.6f}")
        print(f"  Ridge coeffs b (≈ local gradient direction): {np.array2string(b, precision=6, suppress_small=False)}")
        print(f"  Unit direction b/||b||: {np.array2string(direction, precision=6, suppress_small=False)}")
        print(f"  Step size: {step}")
        print(f"  Query: {format_query(x_new)}")
        print()

    print("Submit each query string to the matching function field in the portal.")

main()






