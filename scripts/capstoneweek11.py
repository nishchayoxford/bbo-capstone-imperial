#!/usr/bin/env python
# coding: utf-8

import math
import warnings
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

"""
Capstone Week 11 — Student-Friendly Notes (consistent with Weeks 5–10)

What this script is for
- Build Week 11 query proposals for F1..F8 using observed Weeks 1–10 history.
- Print one portal-ready point per function.

How to read this file
1) DATA section: full historical X (queries) and y (scores) up to Week 10.
2) Helper section: clipping, formatting, splitting history by function.
3) GP section: local surrogate fit (Matérn Gaussian Process + EI scoring).
4) Strategy section: function-wise Week 11 policy.
5) Output section: final Week 11 query strings.

Week 11 strategy summary
- F1: maximin exploration (signal is noisy/near-zero, so coverage is useful).
- F2–F7: local trust-region BO (TuRBO-lite style) with adaptive sigma from recent step sizes.
- F8: same local BO, but keep dimensions that were fixed historically fixed.

Safety / portal constraints
- Coordinates are clipped to [0, 0.999999] for portal compatibility.
- Duplicate submissions are avoided after 6-decimal formatting.

How to run
- python /home/nish/anaconda3/capstoneweek11.py
- Or run in notebook cells (capstoneweek11.ipynb).
"""

np.set_printoptions(suppress=True, precision=6)

# ============================================================
# DATA: Week 1–10 history used for Week 11 planning
# ============================================================

# Each week row has 8 entries: [F1, F2, ..., F8]
X_hist_weeks = [
    [np.array([0.145, 0.515]), np.array([0.755, 0.275]), np.array([0.395, 0.875, 0.635]), np.array([0.275, 0.955, 0.515, 0.145]), np.array([0.635, 0.395, 0.755, 0.875]), np.array([0.515, 0.145, 0.955, 0.395, 0.755]), np.array([0.875, 0.275, 0.635, 0.515, 0.145, 0.955]), np.array([0.145, 0.275, 0.395, 0.515, 0.635, 0.755, 0.875, 0.955])],
    [np.array([0.725, 0.285]), np.array([0.785, 0.305]), np.array([0.145, 0.395, 0.915]), np.array([0.815, 0.245, 0.355, 0.695]), np.array([0.665, 0.365, 0.785, 0.845]), np.array([0.185, 0.745, 0.315, 0.865, 0.455]), np.array([0.845, 0.305, 0.665, 0.485, 0.175, 0.925]), np.array([0.175, 0.305, 0.425, 0.545, 0.665, 0.785, 0.905, 0.945])],
    [np.array([0.515, 0.515]), np.array([0.74, 0.26]), np.array([0.12, 0.347, 0.943]), np.array([0.869, 0.174, 0.339, 0.75]), np.array([0.68, 0.35, 0.8, 0.83]), np.array([0.152, 0.805, 0.251, 0.912, 0.425]), np.array([0.83, 0.32, 0.68, 0.47, 0.19, 0.91]), np.array([0.13, 0.26, 0.38, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.75, 0.75]), np.array([0.73, 0.27]), np.array([0.155, 0.385, 0.905]), np.array([0.795, 0.265, 0.365, 0.665]), np.array([0.695, 0.335, 0.815, 0.815]), np.array([0.17, 0.76, 0.3, 0.89, 0.47]), np.array([0.815, 0.335, 0.695, 0.455, 0.205, 0.895]), np.array([0.14, 0.27, 0.39, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.99, 0.01]), np.array([0.718763, 0.261649]), np.array([0.165, 0.375, 0.895]), np.array([0.785, 0.275, 0.37, 0.65]), np.array([0.707, 0.323, 0.827, 0.803]), np.array([0.2, 0.73, 0.33, 0.84, 0.455]), np.array([0.805202, 0.344798, 0.704798, 0.445202, 0.214798, 0.885202]), np.array([0.12, 0.25, 0.37, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([2.900e-05, 1.417e-03]), np.array([0.722018, 0.263976]), np.array([0.178771, 0.37214, 0.880781]), np.array([0.792676, 0.264502, 0.367988, 0.657198]), np.array([0.728, 0.302, 0.848, 0.782]), np.array([0.218, 0.712, 0.348, 0.81, 0.446]), np.array([0.79173, 0.35827, 0.71827, 0.43173, 0.22827, 0.87173]), np.array([0.114226, 0.244226, 0.364226, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.305976, 0.997403]), np.array([0.721323, 0.261711]), np.array([0.184441, 0.353663, 0.875638]), np.array([0.791656, 0.265832, 0.368297, 0.656143]), np.array([0.591139, 0.057257, 0.976087, 0.523586]), np.array([0.2288, 0.7012, 0.3588, 0.792, 0.4406]), np.array([0.013373, 0.928169, 0.299072, 0.839656, 0.777563, 0.029987]), np.array([0.109607, 0.239607, 0.359607, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.422868, 0.002773]), np.array([0.724285, 0.264402]), np.array([0.181867, 0.354586, 0.878279]), np.array([0.78961, 0.268623, 0.368839, 0.654212]), np.array([0.745989, 0.305287, 0.849251, 0.788893]), np.array([0.23528, 0.69472, 0.36528, 0.7812, 0.43736]), np.array([0.789513, 0.359715, 0.71644, 0.416791, 0.231308, 0.873588]), np.array([0.104988, 0.234988, 0.354988, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.005626, 0.830621]), np.array([0.722773, 0.263092]), np.array([0.179573, 0.365715, 0.88021]), np.array([0.787305, 0.271802, 0.369425, 0.652092]), np.array([0.756823, 0.306295, 0.854453, 0.799818]), np.array([0.238844, 0.691156, 0.368844, 0.77526, 0.435578]), np.array([0.790638, 0.361606, 0.71546, 0.410503, 0.238615, 0.878998]), np.array([0.103545, 0.233545, 0.353545, 0.5, 0.62, 0.74, 0.86, 0.96])],
    [np.array([0.51501, 0.514997]), np.array([0.723998, 0.266649]), np.array([0.177793, 0.357336, 0.86896]), np.array([0.784639, 0.281273, 0.362394, 0.641259]), np.array([0.764969, 0.307245, 0.863995, 0.808149]), np.array([0.232427, 0.691496, 0.365569, 0.773719, 0.430499]), np.array([0.790313, 0.359857, 0.72289, 0.403231, 0.239694, 0.876697]), np.array([0.100814, 0.229796, 0.355035, 0.5, 0.62, 0.74, 0.86, 0.96])],
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
], dtype=float)

FNAMES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]

# ============================================================
# Helper utilities
# ============================================================

def clip01(x):
    """Clip to portal-safe range."""
    return np.clip(np.asarray(x, float), 0.0, 0.999999)


def fmt(x):
    """Format vector as portal string: 0.xxxxxx-0.xxxxxx-..."""
    return "-".join(f"{float(v):.6f}" for v in np.asarray(x, float))


def split_by_function(X_weeks, Y_weeks):
    """Convert week-major history to function-major history."""
    X_by_f = {}
    y_by_f = {}
    for j, f in enumerate(FNAMES):
        X_by_f[f] = np.stack([X_weeks[t][j] for t in range(len(X_weeks))], axis=0)
        y_by_f[f] = Y_weeks[:, j].copy()
    return X_by_f, y_by_f


def sigma_from_last_three_steps(X_hist):
    """Estimate coordinate scale from recent move sizes."""
    d = X_hist.shape[1]
    norms = [
        float(np.linalg.norm(X_hist[-3] - X_hist[-4])),
        float(np.linalg.norm(X_hist[-2] - X_hist[-3])),
        float(np.linalg.norm(X_hist[-1] - X_hist[-2])),
    ]
    s = float(np.median(norms))
    return s / (2.0 * math.sqrt(d))


def fit_local_gp(X, y, seed=0):
    """Fit a local Matérn GP surrogate with tiny noise prior."""
    d = X.shape[1]
    kernel = (
        C(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-12, 1e-2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=1e-10,
        n_restarts_optimizer=5,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)
    return gp


erf_vec = np.vectorize(math.erf, otypes=[float])


def norm_cdf(z):
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def propose_turbo_lite(
    X_hist,
    y_hist,
    sigma_coord,
    seed=0,
    n_cand=120000,
    k_local=8,
    radius_mult=2.5,
    xi=1e-8,
    fixed_dims=None,
):
    """Single-step local BO with EI in a trust region."""
    best_idx = int(np.argmax(y_hist))
    x_best = X_hist[best_idx].copy()
    y_best = float(y_hist[best_idx])

    dists = np.linalg.norm(X_hist - x_best[None, :], axis=1)
    idxs = np.argsort(dists)[: min(k_local, len(X_hist))]
    X_train = X_hist[idxs]
    y_train = y_hist[idxs]

    gp = fit_local_gp(X_train, y_train, seed=seed)
    rng = np.random.default_rng(seed)
    d = X_hist.shape[1]

    CANDS = x_best[None, :] + rng.normal(0.0, sigma_coord, size=(n_cand, d))
    if fixed_dims:
        for j, val in fixed_dims.items():
            CANDS[:, j] = val
    CANDS = clip01(CANDS)

    radius = radius_mult * sigma_coord
    diff = np.abs(CANDS - x_best[None, :])

    if fixed_dims:
        free = [j for j in range(d) if j not in fixed_dims]
        mask = (diff[:, free] <= radius).all(axis=1)
    else:
        mask = (diff <= radius).all(axis=1)

    CANDS = CANDS[mask]
    if CANDS.shape[0] == 0:
        CANDS = rng.uniform(0.0, 0.999999, size=(max(2000, n_cand // 20), d))
        if fixed_dims:
            for j, val in fixed_dims.items():
                CANDS[:, j] = val

    mu, std = gp.predict(CANDS, return_std=True)

    # Expected Improvement (maximization)
    ei = np.zeros_like(mu)
    m = std > 1e-12
    if np.any(m):
        imp = mu[m] - y_best - xi
        Z = imp / std[m]
        Phi = norm_cdf(Z)
        phi = np.exp(-0.5 * Z * Z) / math.sqrt(2.0 * math.pi)
        ei[m] = imp * Phi + std[m] * phi
    ei[ei < 0.0] = 0.0

    idx = int(np.argmax(mu)) if np.all(ei == 0.0) else int(np.argmax(ei))
    x_next = CANDS[idx].copy()

    existing = {fmt(row) for row in X_hist}
    jitter_rng = np.random.default_rng(seed + 999)
    for _ in range(2000):
        if fmt(x_next) not in existing:
            break
        x_next = clip01(x_next + jitter_rng.normal(0.0, 1e-6, size=x_next.shape))

    if fixed_dims:
        for j, val in fixed_dims.items():
            x_next[j] = val

    return clip01(x_next)


def propose_maximin(X_hist, seed=42, n=200000):
    """Pick the point farthest from observed points (space-filling exploration)."""
    rng = np.random.default_rng(seed)
    d = X_hist.shape[1]
    CANDS = rng.uniform(0.0, 0.999999, size=(n, d))
    d2 = ((CANDS[:, None, :] - X_hist[None, :, :]) ** 2).sum(axis=2)
    min_dist = np.sqrt(d2.min(axis=1))
    return CANDS[int(np.argmax(min_dist))].copy()


def infer_fixed_dims_from_late_history(X_hist, start_week_index=2, atol=0.0):
    """Detect dimensions that stayed constant from late history onward."""
    X_sub = X_hist[start_week_index:]
    d = X_hist.shape[1]
    fixed = {}
    for j in range(d):
        if np.allclose(X_sub[:, j], X_sub[0, j], atol=atol, rtol=0.0):
            fixed[j] = float(X_sub[0, j])
    return fixed


# ============================================================
# Build Week 11 plan
# ============================================================

X_by_f, y_by_f = split_by_function(X_hist_weeks, Y_hist_weeks)
week11 = {}

# F1: exploration-first
week11["F1"] = propose_maximin(X_by_f["F1"], seed=42, n=200000)

# F2..F7: local BO with adaptive sigma
for f in ["F2", "F3", "F4", "F5", "F6", "F7"]:
    sigma = sigma_from_last_three_steps(X_by_f[f])
    week11[f] = propose_turbo_lite(
        X_by_f[f],
        y_by_f[f],
        sigma_coord=sigma,
        seed=100 + int(f[1:]),
    )

# F8: local BO with fixed historical dimensions preserved
fixed_f8 = infer_fixed_dims_from_late_history(X_by_f["F8"], start_week_index=2, atol=0.0)
sigma_f8 = sigma_from_last_three_steps(X_by_f["F8"])
week11["F8"] = propose_turbo_lite(
    X_by_f["F8"],
    y_by_f["F8"],
    sigma_coord=sigma_f8,
    seed=108,
    fixed_dims=fixed_f8,
)

print("==== WEEK 11 QUERY PLAN (PORTAL FORMAT) ====")
for f in FNAMES:
    print(f"{f}: {fmt(clip01(week11[f]))}")
