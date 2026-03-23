#!/usr/bin/env python3
"""
Generate professional plots for capstone documentation:
1) Convergence lines for F1..F8 across weeks.
2) 3D GP-surrogate surface slice for each function (F1..F8).

Outputs:
- results/plots/convergence_all_functions.png
- results/plots/F1_surface_slice.png ... F8_surface_slice.png
- results/plots/F1_contour_slice.png ... F8_contour_slice.png
"""

from pathlib import Path
import ast
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO = Path('/home/nish/.openclaw/workspace-aiml-course/github-bbo-capstone')
SOURCE_SCRIPT = REPO / 'scripts/capstoneweek13.py'
PLOTS = REPO / 'results/plots'
PLOTS.mkdir(parents=True, exist_ok=True)

FNAMES = [f'F{i}' for i in range(1, 9)]


def _extract_between(text: str, start_key: str, end_key: str) -> str:
    s = text.index(start_key) + len(start_key)
    e = text.index(end_key, s)
    return text[s:e]


def load_weekly_data():
    text = SOURCE_SCRIPT.read_text(encoding='utf-8')

    x_block = _extract_between(text, 'X_hist_weeks = [', ']\n\nY_hist_weeks')
    y_block = _extract_between(text, 'Y_hist_weeks = np.array([', '], dtype=float)')

    # Rebuild complete literal strings for parsing
    x_literal = '[' + x_block + ']'
    y_literal = '[' + y_block + ']'

    X_weeks = eval(x_literal, {'__builtins__': {}}, {'np': np, 'array': np.array})
    Y_weeks = np.array(ast.literal_eval(y_literal), dtype=float)

    # function-major
    X_by_f, y_by_f = {}, {}
    for j, f in enumerate(FNAMES):
        X_by_f[f] = np.stack([np.asarray(X_weeks[t][j], float) for t in range(len(X_weeks))], axis=0)
        y_by_f[f] = Y_weeks[:, j].copy()
    return X_by_f, y_by_f


def idw_predict(X_train, y_train, X_query, power=2.0, eps=1e-12):
    """Inverse-distance weighting surrogate (dependency-free)."""
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train, float)
    X_query = np.asarray(X_query, float)

    # squared euclidean distances: (n_query, n_train)
    d2 = ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)
    d = np.sqrt(d2) + eps

    w = 1.0 / (d ** power)
    w = w / (w.sum(axis=1, keepdims=True) + eps)
    return (w * y_train[None, :]).sum(axis=1)


def plot_convergence(y_by_f):
    weeks = np.arange(1, len(next(iter(y_by_f.values()))) + 1)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for i, f in enumerate(FNAMES):
        ax = axes[i // 4, i % 4]
        y = y_by_f[f]
        best = np.maximum.accumulate(y)
        ax.plot(weeks, y, marker='o', lw=1.7, label='Observed y')
        ax.plot(weeks, best, lw=1.7, linestyle='--', label='Best-so-far')
        ax.set_title(f)
        ax.set_xlabel('Week')
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle('BBO Capstone: Weekly performance by function (F1–F8)', fontsize=14)
    fig.savefig(PLOTS / 'convergence_all_functions.png', dpi=180)
    plt.close(fig)


def plot_surface_slices(X_by_f, y_by_f, grid_n=55):
    u = np.linspace(0.0, 0.999999, grid_n)
    U, V = np.meshgrid(u, u)

    for idx, f in enumerate(FNAMES, start=1):
        X = X_by_f[f]
        y = y_by_f[f]
        d = X.shape[1]

        best_idx = int(np.argmax(y))
        base = X[best_idx].copy()

        G = np.tile(base, (grid_n * grid_n, 1))
        G[:, 0] = U.ravel()
        G[:, 1] = V.ravel() if d >= 2 else U.ravel()

        Z = idw_predict(X, y, G, power=2.0).reshape(U.shape)

        # 3D surface
        fig = plt.figure(figsize=(9, 6.5))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(U, V, Z, cmap='viridis', edgecolor='none', alpha=0.95)
        ax.scatter(X[:, 0], X[:, 1] if d >= 2 else X[:, 0], y, c='r', s=24, label='Observed points')
        ax.set_title(f'{f} surrogate surface slice (dims 1 & 2)')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Predicted y')
        ax.view_init(elev=28, azim=-60)
        fig.colorbar(surf, shrink=0.65, aspect=16, label='Predicted y')
        ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS / f'{f}_surface_slice.png', dpi=170)
        plt.close(fig)

        # contour
        fig2, ax2 = plt.subplots(figsize=(7.2, 5.8))
        c = ax2.contourf(U, V, Z, levels=20, cmap='viridis')
        ax2.scatter(X[:, 0], X[:, 1] if d >= 2 else X[:, 0], c='white', edgecolors='black', s=28)
        ax2.set_title(f'{f} surrogate contour slice (dims 1 & 2)')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.grid(alpha=0.2)
        fig2.colorbar(c, ax=ax2, label='Predicted y')
        fig2.tight_layout()
        fig2.savefig(PLOTS / f'{f}_contour_slice.png', dpi=170)
        plt.close(fig2)


def main():
    X_by_f, y_by_f = load_weekly_data()
    plot_convergence(y_by_f)
    plot_surface_slices(X_by_f, y_by_f)
    print(f'Plots saved in: {PLOTS}')


if __name__ == '__main__':
    main()
