#!/usr/bin/env python
# coding: utf-8


# %% [cell 1]
import math
import warnings
import itertools
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import GradientBoostingRegressor

np.set_printoptions(suppress=True, precision=6)

"""
WEEK 13 FINAL NOTEBOOK

Core method
- GP + GradientBoosting ensemble surrogate
- UCB / EI hybrid acquisition
- Multi-center trust-region search
- Function-specific exploration vs exploitation

High-level policy
- F1: incumbent-safe fallback
- F2: best-basin anchoring (historical best region)
- F3: revert-and-refine around W10 best
- F4: UCB-heavy basin escape
- F5: slightly farther-reaching exploitation
- F6: UCB-heavy plateau escape
- F7: disciplined local exploitation
- F8: all 8 dims live again, UCB-heavy + blockwise search
"""

# %% [cell 2]
# ============================================================
# DATA: Weeks 1–12 history
# ============================================================

X_hist_weeks = [
    [np.array([0.145, 0.515]), np.array([0.755, 0.275]), np.array([0.395, 0.875, 0.635]), np.array([0.275, 0.955, 0.515, 0.145]), np.array([0.635, 0.395, 0.755, 0.875]), np.array([0.515, 0.145, 0.955, 0.395, 0.755]), np.array([0.875, 0.275, 0.635, 0.515, 0.145, 0.955]), np.array([0.145, 0.275, 0.395, 0.515, 0.635, 0.755, 0.875, 0.955])],
    [np.array([0.725, 0.285]), np.array([0.785, 0.305]), np.array([0.145, 0.395, 0.915]), np.array([0.815, 0.245, 0.355, 0.695]), np.array([0.665, 0.365, 0.785, 0.845]), np.array([0.185, 0.745, 0.315, 0.865, 0.455]), np.array([0.845, 0.305, 0.665, 0.485, 0.175, 0.925]), np.array([0.175, 0.305, 0.425, 0.545, 0.665, 0.785, 0.905, 0.945])],
    [np.array([0.515, 0.515]), np.array([0.74, 0.26]), np.array([0.12, 0.347, 0.943]), np.array([0.869, 0.174, 0.339, 0.75]), np.array([0.68, 0.35, 0.8, 0.83]), np.array([0.152, 0.805, 0.251, 0.912, 0.425]), np.array([0.83, 0.32, 0.68, 0.47, 0.19, 0.91]), np.array([0.13, 0.26, 0.38, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.75, 0.75]), np.array([0.73, 0.27]), np.array([0.155, 0.385, 0.905]), np.array([0.795, 0.265, 0.365, 0.665]), np.array([0.695, 0.335, 0.815, 0.815]), np.array([0.17, 0.76, 0.3, 0.89, 0.47]), np.array([0.815, 0.335, 0.695, 0.455, 0.205, 0.895]), np.array([0.14, 0.27, 0.39, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.99, 0.01]), np.array([0.718763, 0.261649]), np.array([0.165, 0.375, 0.895]), np.array([0.785, 0.275, 0.37, 0.65]), np.array([0.707, 0.323, 0.827, 0.803]), np.array([0.2, 0.73, 0.33, 0.84, 0.455]), np.array([0.805202, 0.344798, 0.704798, 0.445202, 0.214798, 0.885202]), np.array([0.12, 0.25, 0.37, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([2.9e-05, 0.001417]), np.array([0.722018, 0.263976]), np.array([0.178771, 0.37214, 0.880781]), np.array([0.792676, 0.264502, 0.367988, 0.657198]), np.array([0.728, 0.302, 0.848, 0.782]), np.array([0.218, 0.712, 0.348, 0.81, 0.446]), np.array([0.79173, 0.35827, 0.71827, 0.43173, 0.22827, 0.87173]), np.array([0.114226, 0.244226, 0.364226, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.305976, 0.997403]), np.array([0.721323, 0.261711]), np.array([0.184441, 0.353663, 0.875638]), np.array([0.791656, 0.265832, 0.368297, 0.656143]), np.array([0.591139, 0.057257, 0.976087, 0.523586]), np.array([0.2288, 0.7012, 0.3588, 0.792, 0.4406]), np.array([0.013373, 0.928169, 0.299072, 0.839656, 0.777563, 0.029987]), np.array([0.109607, 0.239607, 0.359607, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.422868, 0.002773]), np.array([0.724285, 0.264402]), np.array([0.181867, 0.354586, 0.878279]), np.array([0.78961, 0.268623, 0.368839, 0.654212]), np.array([0.745989, 0.305287, 0.849251, 0.788893]), np.array([0.23528, 0.69472, 0.36528, 0.7812, 0.43736]), np.array([0.789513, 0.359715, 0.71644, 0.416791, 0.231308, 0.873588]), np.array([0.104988, 0.234988, 0.354988, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.005626, 0.830621]), np.array([0.722773, 0.263092]), np.array([0.179573, 0.365715, 0.88021]), np.array([0.787305, 0.271802, 0.369425, 0.652092]), np.array([0.756823, 0.306295, 0.854453, 0.799818]), np.array([0.238844, 0.691156, 0.368844, 0.77526, 0.435578]), np.array([0.790638, 0.361606, 0.71546, 0.410503, 0.238615, 0.878998]), np.array([0.103545, 0.233545, 0.353545, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.51501, 0.514997]), np.array([0.723998, 0.266649]), np.array([0.177793, 0.357336, 0.86896]), np.array([0.784639, 0.281273, 0.362394, 0.641259]), np.array([0.764969, 0.307245, 0.863995, 0.808149]), np.array([0.232427, 0.691496, 0.365569, 0.773719, 0.430499]), np.array([0.790313, 0.359857, 0.72289, 0.403231, 0.239694, 0.876697]), np.array([0.100814, 0.229796, 0.355035, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.999972, 0.999842]), np.array([0.727609, 0.264271]), np.array([0.170478, 0.360459, 0.860882]), np.array([0.783515, 0.282458, 0.3596, 0.638557]), np.array([0.770105, 0.31664, 0.87353, 0.817254]), np.array([0.227438, 0.691141, 0.365093, 0.772713, 0.427418]), np.array([0.7908, 0.359965, 0.728517, 0.397789, 0.237068, 0.874439]), np.array([0.101631, 0.232396, 0.351414, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.515075, 0.514919]), np.array([0.724875, 0.264121]), np.array([0.176001, 0.357398, 0.867368]), np.array([0.78206, 0.283883, 0.358083, 0.636839]), np.array([0.776519, 0.316606, 0.880967, 0.825098]), np.array([0.233818, 0.69188, 0.366212, 0.773982, 0.429499]), np.array([0.790463, 0.361297, 0.732293, 0.393909, 0.23915, 0.872327]), np.array([0.101699, 0.231828, 0.350766, 0.5, 0.62, 0.74, 0.86, 0.96])]
]

Y_hist_weeks = np.array([
    [-3.353165630322361e-61, 0.42044085041824825, -0.12080733985523133, -18.59723490448631, 287.4343816627659, -1.6304531811460896, 0.6267064847700778, 8.633935],
    [6.743225602289377e-78, -0.0456643112924181, -0.11535196594300248, -14.395540985679897, 292.2593658119571, -1.4347679755670883, 0.8069621926499697, 8.451335],
    [4.714509345171323e-13, 0.46274019045813003, -0.20076336857175398, -18.67377341401988, 301.5311905557768, -1.6451191179236977, 0.8919314248129555, 8.71814],
    [1.3319145509281447e-22, 0.6060955609811236, -0.07852077254038155, -13.169944884454413, 315.65049985154724, -1.6022183821509282, 0.969339703275594, 8.69914],
    [0.0, 0.5195146975906033, -0.06033571734237718, -12.699964227491282, 330.66611638919255, -1.3295280101304827, 1.0144420450032012, 8.73594],
    [1.825040909472812e-247, 0.5794253005452772, -0.04739292498526722, -12.987699814058924, 365.66328225833024, -1.2429202946292475, 1.0679017392374972, 8.745671245544],
    [-1.5662072753465034e-167, 0.5796694237276565, -0.05056402944032541, -12.94099410856025, 283.75880106841055, -1.2012624047628697, 0.10868500160826922, 8.753167873306],
    [-7.806084086345555e-123, 0.6272586156230583, -0.0478844185459012, -12.85705507882481, 413.12789189378645, -1.122295840899941, 1.0862632473367084, 8.760408479136],
    [1.5539262084660508e-237, 0.45895771213691383, -0.04178386362696305, -12.773487280801856, 472.01213995906096, -1.1753434539633536, 1.0911035659970207, 8.76261799785],
    [4.703738868832531e-13, 0.3874579841475896, -0.034103125669950204, -12.329975985137398, 538.0130710513863, -1.1220282300796462, 1.1115031887444646, 8.761249019517],
    [1.872770193805514e-192, 0.47431047359278533, -0.043208142756616864, -12.184664944867055, 607.1300596362602, -1.138471254079614, 1.1257593200884997, 8.765673100674],
    [4.723599890726132e-13, 0.4976822234044719, -0.04474602512711549, -12.086855405433607, 672.5061969450038, -1.112747932286696, 1.1339304757057649, 8.766625524946],
], dtype=float)

FNAMES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]

# %% [cell 3]
# ============================================================
# Utilities
# ============================================================

def clip01(x):
    return np.clip(np.asarray(x, float), 0.0, 0.999999)

def fmt(x):
    return "-".join(f"{float(v):.6f}" for v in np.asarray(x, float))

def split_by_function(X_weeks, Y_weeks):
    X_by_f, y_by_f = {}, {}
    for j, f in enumerate(FNAMES):
        X_by_f[f] = np.stack([X_weeks[t][j] for t in range(len(X_weeks))], axis=0)
        y_by_f[f] = Y_weeks[:, j].copy()
    return X_by_f, y_by_f

def unique_after_rounding(x, X_existing, seed=0, tries=5000, jitter=5e-7):
    rng = np.random.default_rng(seed)
    existing_strings = {fmt(row) for row in np.asarray(X_existing, float)}
    x = clip01(np.asarray(x, float).copy())
    for _ in range(tries):
        if fmt(x) not in existing_strings:
            return clip01(x)
        x = clip01(x + rng.normal(0.0, jitter, size=x.shape))
    return clip01(x)

def sigma_last_steps(X_hist, mult=1.0):
    d = X_hist.shape[1]
    norms = [
        float(np.linalg.norm(X_hist[-1] - X_hist[-2])),
        float(np.linalg.norm(X_hist[-2] - X_hist[-3])),
        float(np.linalg.norm(X_hist[-3] - X_hist[-4])),
    ]
    base = np.median(norms) / (2.0 * math.sqrt(d))
    return float(base * mult)

def elite_center(X, y, k=4):
    idx = np.argsort(-y)[: min(k, len(y))]
    ys = y[idx]
    w = ys - ys.min() + 1e-9
    w = w / w.sum()
    return np.sum(X[idx] * w[:, None], axis=0)

def select_training_indices(X, y, anchor, k_near=8, top_k=4):
    dists = np.linalg.norm(X - anchor[None, :], axis=1)
    near = list(np.argsort(dists)[: min(k_near, len(y))])
    top = list(np.argsort(-y)[: min(top_k, len(y))])
    return np.array(sorted(set(near + top)))

def fit_gp(X, y, seed=0):
    d = X.shape[1]
    kernel = (
        C(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-3, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-12, 1e-2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=1e-10,
        n_restarts_optimizer=8,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)
    return gp

def fit_gbr_loocv(X, y, seed=0):
    """
    Small but strong LOOCV grid.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    grid = list(itertools.product(
        [80, 160],
        [0.03, 0.07],
        [2, 3],
        [0.7, 1.0],
        [1, 2]
    ))

    best_mse = np.inf
    best_params = None

    n = len(y)
    for n_estimators, lr, depth, subsample, min_leaf in grid:
        errs = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=lr,
                max_depth=depth,
                subsample=subsample,
                min_samples_leaf=min_leaf,
                random_state=seed,
            )
            model.fit(X[mask], y[mask])
            pred = float(model.predict(X[~mask])[0])
            errs.append((y[~mask][0] - pred) ** 2)

        mse = float(np.mean(errs))
        if mse < best_mse:
            best_mse = mse
            best_params = {
                "n_estimators": n_estimators,
                "learning_rate": lr,
                "max_depth": depth,
                "subsample": subsample,
                "min_samples_leaf": min_leaf,
            }

    model = GradientBoostingRegressor(
        **best_params,
        random_state=seed
    )
    model.fit(X, y)

    rel_mse = best_mse / (float(np.var(y)) + 1e-12)
    return model, best_params, best_mse, rel_mse

erf_vec = np.vectorize(math.erf, otypes=[float])

def norm_cdf(z):
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))

def norm_pdf(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

def expected_improvement(mu, std, y_best, xi=1e-8):
    out = np.zeros_like(mu)
    mask = std > 1e-12
    if np.any(mask):
        improvement = mu[mask] - y_best - xi
        Z = improvement / std[mask]
        out[mask] = improvement * norm_cdf(Z) + std[mask] * norm_pdf(Z)
    out[out < 0.0] = 0.0
    return out

def zscore(v):
    v = np.asarray(v, float)
    return (v - np.median(v)) / (np.std(v) + 1e-12)

def elite_box(X, y, k=4, pad=0.25):
    idx = np.argsort(-y)[: min(k, len(y))]
    lo = np.min(X[idx], axis=0)
    hi = np.max(X[idx], axis=0)
    span = np.maximum(hi - lo, 0.05)
    lo = np.clip(lo - pad * span, 0.0, 0.999999)
    hi = np.clip(hi + pad * span, 0.0, 0.999999)
    return lo, hi

# %% [cell 4]
# ============================================================
# Candidate generation and scoring
# ============================================================

def build_candidate_pool(
    X_hist,
    y_hist,
    centers,
    sigma,
    radius_mult,
    seed=0,
    n_local_per_center=12000,
    n_global_elite=0,
    n_global_uniform=0,
    box_pad=0.25,
    masks=None,
    trend_dirs=None,
    fixed_dims=None,
):
    rng = np.random.default_rng(seed)
    X_hist = np.asarray(X_hist, float)
    d = X_hist.shape[1]
    radius = sigma * radius_mult
    pools = []

    # Local Gaussian clouds around centers
    for center in centers:
        center = clip01(center)
        C = center[None, :] + rng.normal(0.0, sigma, size=(n_local_per_center, d))

        if fixed_dims:
            for j, val in fixed_dims.items():
                C[:, j] = val

        C = clip01(C)

        diff = np.abs(C - center[None, :])
        if fixed_dims:
            free = [j for j in range(d) if j not in fixed_dims]
            keep = (diff[:, free] <= radius).all(axis=1)
        else:
            keep = (diff <= radius).all(axis=1)

        C = C[keep]
        if len(C):
            pools.append(C)

    # Masked local clouds (useful especially for F8)
    if masks is not None and len(masks) > 0:
        base_center = clip01(centers[0])
        n_mask = max(5000, n_local_per_center // 2)

        for mask in masks:
            mask = list(mask)
            C = np.tile(base_center, (n_mask, 1))
            C[:, mask] += rng.normal(0.0, sigma * 1.25, size=(n_mask, len(mask)))

            if fixed_dims:
                for j, val in fixed_dims.items():
                    C[:, j] = val

            C = clip01(C)
            diff = np.abs(C - base_center[None, :])
            keep = (diff[:, mask] <= radius * 1.2).all(axis=1)
            C = C[keep]

            if len(C):
                pools.append(C)

    # Line candidates along successful trend directions
    if trend_dirs is not None and len(trend_dirs) > 0:
        base = clip01(centers[0])
        line_pts = []
        step = sigma * math.sqrt(d)

        for direction in trend_dirs:
            v = np.asarray(direction, float).copy()
            if fixed_dims:
                for j in fixed_dims:
                    v[j] = 0.0

            norm = float(np.linalg.norm(v))
            if norm < 1e-12:
                continue

            u = v / norm
            for mult in [0.5, 1.0, 1.5]:
                line_pts.append(base + mult * step * u)
                line_pts.append(base - 0.25 * mult * step * u)

        if line_pts:
            L = clip01(np.array(line_pts, float))
            if fixed_dims:
                for j, val in fixed_dims.items():
                    L[:, j] = val
            pools.append(L)

    # Elite-box global samples
    if n_global_elite > 0:
        lo, hi = elite_box(X_hist, y_hist, k=4, pad=box_pad)
        G = lo + rng.random((n_global_elite, d)) * (hi - lo)
        if fixed_dims:
            for j, val in fixed_dims.items():
                G[:, j] = val
        pools.append(clip01(G))

    # Uniform global samples
    if n_global_uniform > 0:
        U = rng.random((n_global_uniform, d)) * 0.999999
        if fixed_dims:
            for j, val in fixed_dims.items():
                U[:, j] = val
        pools.append(clip01(U))

    CANDS = np.vstack(pools)

    # Remove historical duplicates after portal formatting
    hist_strings = {fmt(row) for row in X_hist}
    keep = np.array([fmt(row) not in hist_strings for row in CANDS], dtype=bool)
    CANDS = CANDS[keep]

    return clip01(CANDS)


def score_candidates(
    gp,
    gbr,
    y_best,
    CANDS,
    w_ucb=0.5,
    kappa=2.0,
    gbr_weight=0.35,
    disagreement_penalty=0.15,
):
    gp_mu, gp_std = gp.predict(CANDS, return_std=True)
    gbr_mu = gbr.predict(CANDS)

    ensemble_mu = (1.0 - gbr_weight) * gp_mu + gbr_weight * gbr_mu
    ucb = ensemble_mu + kappa * gp_std
    ei = expected_improvement(gp_mu, gp_std, y_best, xi=1e-8)
    disagreement = np.abs(gp_mu - gbr_mu)

    score = (
        w_ucb * zscore(ucb)
        + (1.0 - w_ucb) * zscore(ei)
        - disagreement_penalty * zscore(disagreement)
    )

    med_std = float(np.median(gp_std))
    safe_mask = gp_std <= (3.0 * med_std + 1e-12)

    if np.any(safe_mask):
        safe_idx = np.where(safe_mask)[0]
        idx = int(safe_idx[np.argmax(score[safe_mask])])
    else:
        idx = int(np.argmax(score))

    info = {
        "best_score": float(score[idx]),
        "pred_gp_mu": float(gp_mu[idx]),
        "pred_gp_std": float(gp_std[idx]),
        "pred_gbr_mu": float(gbr_mu[idx]),
        "pred_ucb": float(ucb[idx]),
        "pred_ei": float(ei[idx]),
        "pred_disagreement": float(disagreement[idx]),
    }
    return CANDS[idx], info

# %% [cell 5]
# ============================================================
# Function-specific strategy
# ============================================================

X_by_f, y_by_f = split_by_function(X_hist_weeks, Y_hist_weeks)

def propose_f1(X, y, seed=101):
    # Weak / degenerate signal -> stay near best-known interior basin
    center = np.mean(np.vstack([X[2], X[9], X[11]]), axis=0)  # W3, W10, W12 region
    x = center + np.array([5e-5, -6e-5])
    return unique_after_rounding(x, X, seed=seed), {
        "mode": "INCUMBENT_SAFE",
        "center": center
    }

def get_centers_and_dirs(f, X, y):
    best_idx = int(np.argmax(y))

    if f == "F2":
        # Historical best basin at Week 8 (index 7)
        best_idx = 7
        centers = [
            X[7],
            elite_center(X, y, k=3),
            0.75 * X[7] + 0.25 * X[11],
            0.70 * X[7] + 0.30 * X[3],
        ]
        trend_dirs = [
            X[7] - X[6],
            X[7] - X[11],
            X[3] - X[4],
        ]
        return best_idx, centers, trend_dirs

    if f == "F3":
        # Historical best at Week 10 (index 9)
        best_idx = 9
        centers = [
            X[9],
            elite_center(X, y, k=3),
            0.80 * X[9] + 0.20 * X[8],
            0.85 * X[9] + 0.15 * X[11],
        ]
        trend_dirs = [
            X[9] - X[8],
            X[9] - X[10],
            X[9] - X[11],
        ]
        return best_idx, centers, trend_dirs

    if f == "F4":
        centers = [
            X[best_idx],
            elite_center(X, y, k=4),
            X[best_idx] + 0.60 * (X[-1] - X[-2]),
            X[best_idx] + 0.30 * (X[-1] - X[-3]),
        ]
        trend_dirs = [
            X[-1] - X[-2],
            X[-2] - X[-3],
            X[-1] - X[-3],
        ]
        return best_idx, centers, trend_dirs

    if f == "F5":
        centers = [
            X[best_idx],
            elite_center(X, y, k=4),
            X[best_idx] + 0.80 * (X[-1] - X[-2]),
            X[best_idx] + 0.40 * (X[-1] - X[-3]),
        ]
        trend_dirs = [
            X[-1] - X[-2],
            X[-2] - X[-3],
            X[-1] - X[-3],
        ]
        return best_idx, centers, trend_dirs

    if f == "F6":
        centers = [
            X[best_idx],
            elite_center(X, y, k=4),
            X[best_idx] + 0.70 * (X[-1] - X[-2]),
            0.60 * X[best_idx] + 0.40 * X[9],
        ]
        trend_dirs = [
            X[-1] - X[-2],
            X[9] - X[10],
            X[11] - X[10],
        ]
        return best_idx, centers, trend_dirs

    if f == "F7":
        centers = [
            X[best_idx],
            elite_center(X, y, k=4),
            X[best_idx] + 0.60 * (X[-1] - X[-2]),
        ]
        trend_dirs = [
            X[-1] - X[-2],
            X[-2] - X[-3],
            X[-1] - X[-3],
        ]
        return best_idx, centers, trend_dirs

    if f == "F8":
        centers = [
            X[best_idx],
            elite_center(X, y, k=4),
            X[best_idx] + 0.50 * (X[-1] - X[-2]),
            0.70 * X[best_idx] + 0.30 * X[10],
        ]
        trend_dirs = [
            X[-1] - X[-2],
            X[-2] - X[-3],
            X[-1] - X[-3],
        ]
        return best_idx, centers, trend_dirs

    return best_idx, [X[best_idx]], []


CONFIG = {
    "F2": {
        "sigma_mult": 0.18,
        "radius_mult": 1.6,
        "w_ucb": 0.20,
        "kappa": 0.80,
        "gbr_weight": 0.15,
        "disagreement_penalty": 0.10,
        "n_local_per_center": 12000,
        "n_global_elite": 4000,
        "n_global_uniform": 0,
        "box_pad": 0.5,
        "k_near": 6,
        "top_k": 4,
        "restarts": 3,
        "masks": None,
    },
    "F3": {
        "sigma_mult": 0.20,
        "radius_mult": 1.6,
        "w_ucb": 0.10,
        "kappa": 0.60,
        "gbr_weight": 0.15,
        "disagreement_penalty": 0.10,
        "n_local_per_center": 12000,
        "n_global_elite": 3000,
        "n_global_uniform": 0,
        "box_pad": 0.4,
        "k_near": 7,
        "top_k": 4,
        "restarts": 3,
        "masks": None,
    },
    "F4": {
        "sigma_mult": 1.40,
        "radius_mult": 2.8,
        "w_ucb": 0.75,
        "kappa": 4.50,
        "gbr_weight": 0.35,
        "disagreement_penalty": 0.18,
        "n_local_per_center": 15000,
        "n_global_elite": 6000,
        "n_global_uniform": 25000,
        "box_pad": 1.0,
        "k_near": 9,
        "top_k": 5,
        "restarts": 3,
        "masks": None,
    },
    "F5": {
        "sigma_mult": 1.10,
        "radius_mult": 2.5,
        "w_ucb": 0.45,
        "kappa": 2.00,
        "gbr_weight": 0.45,
        "disagreement_penalty": 0.15,
        "n_local_per_center": 15000,
        "n_global_elite": 8000,
        "n_global_uniform": 4000,
        "box_pad": 0.7,
        "k_near": 9,
        "top_k": 5,
        "restarts": 3,
        "masks": None,
    },
    "F6": {
        "sigma_mult": 1.60,
        "radius_mult": 3.0,
        "w_ucb": 0.65,
        "kappa": 3.50,
        "gbr_weight": 0.35,
        "disagreement_penalty": 0.18,
        "n_local_per_center": 15000,
        "n_global_elite": 7000,
        "n_global_uniform": 25000,
        "box_pad": 1.0,
        "k_near": 9,
        "top_k": 5,
        "restarts": 3,
        "masks": None,
    },
    "F7": {
        "sigma_mult": 0.85,
        "radius_mult": 2.0,
        "w_ucb": 0.25,
        "kappa": 1.00,
        "gbr_weight": 0.20,
        "disagreement_penalty": 0.12,
        "n_local_per_center": 15000,
        "n_global_elite": 4000,
        "n_global_uniform": 0,
        "box_pad": 0.5,
        "k_near": 9,
        "top_k": 5,
        "restarts": 3,
        "masks": None,
    },
    "F8": {
        "sigma_mult": 1.80,
        "radius_mult": 3.5,
        "w_ucb": 0.80,
        "kappa": 6.00,
        "gbr_weight": 0.45,
        "disagreement_penalty": 0.18,
        "n_local_per_center": 14000,
        "n_global_elite": 8000,
        "n_global_uniform": 20000,
        "box_pad": 1.2,
        "k_near": 10,
        "top_k": 5,
        "restarts": 3,
        "masks": [
            [0, 1, 2],
            [3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4],
            [2, 3, 4, 5, 6, 7],
        ],
    },
}


def propose_with_ensemble_bo(f, X, y, seed_base=100):
    cfg = CONFIG[f]
    best_idx, centers, trend_dirs = get_centers_and_dirs(f, X, y)
    anchor = X[best_idx].copy()

    train_idxs = select_training_indices(
        X, y, anchor,
        k_near=cfg["k_near"],
        top_k=cfg["top_k"]
    )

    X_train = X[train_idxs]
    y_train = y[train_idxs]

    gp = fit_gp(X_train, y_train, seed=seed_base)
    gbr, gbr_params, loocv_mse, rel_mse = fit_gbr_loocv(X_train, y_train, seed=seed_base)

    gbr_weight = cfg["gbr_weight"]
    if rel_mse > 3.0:
        gbr_weight = 0.0
    elif rel_mse > 1.5:
        gbr_weight = min(gbr_weight, 0.20)

    sigma = sigma_last_steps(X, mult=cfg["sigma_mult"])

    best_candidate = None
    best_info = None
    best_score = -np.inf

    for r in range(cfg["restarts"]):
        seed = seed_base + 1000 * r

        CANDS = build_candidate_pool(
            X_hist=X,
            y_hist=y,
            centers=centers,
            sigma=sigma,
            radius_mult=cfg["radius_mult"],
            seed=seed,
            n_local_per_center=cfg["n_local_per_center"],
            n_global_elite=cfg["n_global_elite"],
            n_global_uniform=cfg["n_global_uniform"],
            box_pad=cfg["box_pad"],
            masks=cfg["masks"],
            trend_dirs=trend_dirs,
            fixed_dims=None,
        )

        cand, info = score_candidates(
            gp=gp,
            gbr=gbr,
            y_best=float(np.max(y)),
            CANDS=CANDS,
            w_ucb=cfg["w_ucb"],
            kappa=cfg["kappa"],
            gbr_weight=gbr_weight,
            disagreement_penalty=cfg["disagreement_penalty"],
        )

        if info["best_score"] > best_score:
            best_score = info["best_score"]
            best_candidate = cand.copy()
            best_info = info.copy()

    best_candidate = unique_after_rounding(best_candidate, X, seed=seed_base + 999)

    full_info = {
        "anchor_idx": int(best_idx),
        "anchor": anchor,
        "train_idxs": train_idxs,
        "gp_kernel": str(gp.kernel_),
        "gbr_params": gbr_params,
        "gbr_loocv_mse": loocv_mse,
        "gbr_rel_mse": rel_mse,
        "gbr_weight_used": gbr_weight,
        "sigma_used": sigma,
        "config": cfg,
    }
    full_info.update(best_info)
    return best_candidate, full_info

# %% [cell 6]
# ============================================================
# Build final Week 13 plan
# ============================================================

FINAL_PLAN = {}
FINAL_INFO = {}

# F1 special case
FINAL_PLAN["F1"], FINAL_INFO["F1"] = propose_f1(X_by_f["F1"], y_by_f["F1"], seed=101)

# Others
for f, seed in zip(["F2", "F3", "F4", "F5", "F6", "F7", "F8"], [102, 103, 104, 105, 106, 107, 108]):
    FINAL_PLAN[f], FINAL_INFO[f] = propose_with_ensemble_bo(f, X_by_f[f], y_by_f[f], seed_base=seed)

print("==== WEEK 13 FINAL QUERY PLAN (PORTAL FORMAT) ====\n")
for f in FNAMES:
    print(f"{f}: {fmt(FINAL_PLAN[f])}")

# %% [cell 7]
# ============================================================
# Diagnostics
# ============================================================

for f in FNAMES:
    print(f"\n{f}")
    for k, v in FINAL_INFO[f].items():
        print(f"  {k}: {v}")

# %% [cell 8]
# ============================================================
# Optional: compact summary table
# ============================================================

summary_rows = []
for f in FNAMES:
    info = FINAL_INFO[f]
    summary_rows.append({
        "Function": f,
        "Query": fmt(FINAL_PLAN[f]),
        "AnchorIdx": info.get("anchor_idx", None),
        "Sigma": info.get("sigma_used", None),
        "BestScore": info.get("best_score", None),
        "PredUCB": info.get("pred_ucb", None),
        "PredEI": info.get("pred_ei", None),
    })

summary_rows

# %% [cell 9]
