#!/usr/bin/env python3
"""
Generate one combined IDW-style visualization for all functions F1-F8.

For F1-F2 (2D functions), the plot is a direct IDW surface over x1/x2.
For F3-F8 (higher-dimensional functions), the plot is an IDW slice over x1/x2
while fixing the remaining dimensions at the best observed point.

Output:
- results/plots/idw_all_functions.png
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
PLOTS = REPO / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
FNAMES = [f"F{i}" for i in range(1, 9)]


def parse_outputs(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"np\.float64\(([^\)]*)\)", r"\1", line)
        rows.append(eval(line, {"__builtins__": {}}, {}))
    return np.asarray(rows, dtype=float)


def parse_inputs(path: Path):
    rows = []
    safe_globals = {"np": np, "array": np.array}
    text = path.read_text(encoding="utf-8")

    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start is not None:
                expr = text[start : i + 1]
                rows.append(eval(expr, {"__builtins__": {}}, safe_globals))
                start = None
    return rows


def idw_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, power: float = 2.0):
    d = np.sqrt(((query_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2))
    d = np.maximum(d, 1e-9)
    w = 1.0 / (d ** power)
    w = w / np.sum(w, axis=1, keepdims=True)
    return (w * train_y[None, :]).sum(axis=1)


def build_slice_grid(X: np.ndarray, y: np.ndarray, grid_n: int = 60):
    d = X.shape[1]
    best_idx = int(np.argmax(y))
    base = X[best_idx].copy()

    gx = np.linspace(0.0, 0.999999, grid_n)
    gy = np.linspace(0.0, 0.999999, grid_n)
    XX, YY = np.meshgrid(gx, gy)

    G = np.tile(base, (grid_n * grid_n, 1))
    G[:, 0] = XX.ravel()
    G[:, 1] = YY.ravel() if d >= 2 else XX.ravel()

    Z = idw_predict(X, y, G, power=2.0).reshape(XX.shape)
    return XX, YY, Z, base


def main():
    outputs_file = REPO / "data" / "weekly_results" / "outputs" / "13.txt"
    inputs_file = REPO / "data" / "weekly_results" / "inputs" / "13.txt"

    Y = parse_outputs(outputs_file)
    X_weeks = parse_inputs(inputs_file)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    for j, f in enumerate(FNAMES):
        ax = axes[j // 4, j % 4]
        X = np.asarray([row[j] for row in X_weeks], dtype=float)
        y = Y[:, j]
        XX, YY, Z, base = build_slice_grid(X, y, grid_n=55)

        c = ax.contourf(XX, YY, Z, levels=20, cmap="viridis")
        ax.scatter(X[:, 0], X[:, 1] if X.shape[1] >= 2 else X[:, 0], c="white", edgecolors="black", s=22)

        if X.shape[1] == 2:
            ax.set_title(f"{f} IDW surface")
        else:
            fixed = ", ".join([f"x{k+1}={base[k]:.2f}" for k in range(2, X.shape[1])])
            ax.set_title(f"{f} IDW slice")
            ax.text(0.02, -0.22, fixed, transform=ax.transAxes, fontsize=7)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.15)
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("IDW visualizations for F1-F8 (Weeks 1-13)", fontsize=16)
    out = PLOTS / "idw_all_functions.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
