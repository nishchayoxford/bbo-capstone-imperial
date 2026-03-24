#!/usr/bin/env python3
"""
Generate professional project plots from weekly_results logs.

Outputs:
- results/plots/function_progress.png
- results/plots/function_progress_normalized.png
- results/plots/f1_surface_gp.png
- results/plots/f2_surface_gp.png
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)


def parse_outputs(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # convert np.float64(...) wrappers to plain values
        line = re.sub(r"np\.float64\(([^\)]*)\)", r"\1", line)
        rows.append(eval(line, {"__builtins__": {}}, {}))  # numeric lists only
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


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_progress(Y: np.ndarray, outdir: Path):
    weeks = np.arange(1, Y.shape[0] + 1)
    fnames = [f"F{i}" for i in range(1, 9)]

    plt.figure(figsize=(12, 7))
    for j, f in enumerate(fnames):
        plt.plot(weeks, Y[:, j], marker="o", linewidth=2, label=f)
    plt.title("BBO Weekly Score Progress (Raw Scale)")
    plt.xlabel("Week")
    plt.ylabel("Score y")
    plt.grid(alpha=0.25)
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(outdir / "function_progress.png", dpi=180)
    plt.close()

    Yn = np.zeros_like(Y)
    for j in range(Y.shape[1]):
        col = Y[:, j]
        cmin, cmax = float(np.min(col)), float(np.max(col))
        Yn[:, j] = 0.0 if cmax - cmin < 1e-12 else (col - cmin) / (cmax - cmin)

    plt.figure(figsize=(12, 7))
    for j, f in enumerate(fnames):
        plt.plot(weeks, Yn[:, j], marker="o", linewidth=2, label=f)
    plt.title("BBO Weekly Score Progress (Per-Function Normalized)")
    plt.xlabel("Week")
    plt.ylabel("Normalized score [0,1]")
    plt.grid(alpha=0.25)
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(outdir / "function_progress_normalized.png", dpi=180)
    plt.close()


def idw_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, power: float = 2.0):
    # Inverse Distance Weighting interpolation (dependency-light surrogate)
    d = np.sqrt(((query_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2))
    d = np.maximum(d, 1e-9)
    w = 1.0 / (d ** power)
    w = w / np.sum(w, axis=1, keepdims=True)
    return (w * train_y[None, :]).sum(axis=1)


def plot_2d_surface(X2: np.ndarray, y: np.ndarray, title: str, out_path: Path):
    grid_n = 70
    gx = np.linspace(0.0, 0.999999, grid_n)
    gy = np.linspace(0.0, 0.999999, grid_n)
    XX, YY = np.meshgrid(gx, gy)
    G = np.column_stack([XX.ravel(), YY.ravel()])
    ZZ = idw_predict(X2, y, G, power=2.0).reshape(XX.shape)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XX, YY, ZZ, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.scatter(X2[:, 0], X2[:, 1], y, color="crimson", s=26, label="Observed points")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Interpolated y")
    ax.legend(loc="upper left")
    fig.colorbar(surf, shrink=0.75, aspect=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    repo = Path(__file__).resolve().parents[1]
    outdir = repo / "results" / "plots"
    ensure_dir(outdir)

    outputs_file = repo / "data" / "weekly_results" / "outputs" / "13.txt"
    inputs_file = repo / "data" / "weekly_results" / "inputs" / "13.txt"

    Y = parse_outputs(outputs_file)
    X_weeks = parse_inputs(inputs_file)

    # Progress plots
    plot_progress(Y, outdir)

    # 2D surface plots for F1 and F2 only (the only 2D functions)
    X_f1 = np.asarray([row[0] for row in X_weeks], dtype=float)
    X_f2 = np.asarray([row[1] for row in X_weeks], dtype=float)
    y_f1 = Y[:, 0]
    y_f2 = Y[:, 1]

    plot_2d_surface(X_f1, y_f1, "F1 Surface (IDW interpolation from weekly observations)", outdir / "f1_surface_idw.png")
    plot_2d_surface(X_f2, y_f2, "F2 Surface (IDW interpolation from weekly observations)", outdir / "f2_surface_idw.png")

    print("Saved plots to:", outdir)


if __name__ == "__main__":
    main()
