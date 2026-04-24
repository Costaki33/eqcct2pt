#!/usr/bin/env python3
"""2x2 error-distribution figure (CDF CPU, CDF GPU, Violin MAE, Box max error).

Reads the NPZ produced by ``eqcct_sb.validation.tf_pt_per_window_errors`` and
writes ``figures/tf_pt_error_distributions.png``.

NPZ key convention: ``<profile>_<branch>_<metric>`` where
``profile in {cpu, gpu0, gpu1}``, ``branch in {p, s}``, and
``metric in {mae, max}``. (``<profile>_dataset`` holds dataset names.)

Usage::

    python scripts/plot_error_distributions.py results/per_window_errors.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _log_decade_ticks(ax, axis: str = "y") -> None:
    """Decade-only major ticks (10^-N), no minor ticks, grid on majors."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(mticker.LogLocator(base=10.0))
    target.set_minor_locator(mticker.NullLocator())
    ax.grid(True, axis=axis, which="major", alpha=0.3)

REPO = Path(__file__).resolve().parents[1]


def _empirical_cdf(values: np.ndarray):
    v = np.sort(values)
    return v, np.arange(1, v.size + 1) / v.size


def _safe(x: np.ndarray, floor: float = 1e-16) -> np.ndarray:
    """Clip non-positive values for log-scale plotting."""
    return np.maximum(x, floor)


def _which_gpu(arrays_keys) -> str:
    """Pick the first available GPU profile present in the NPZ (gpu0 preferred)."""
    for cand in ("gpu0", "gpu1"):
        if any(k.startswith(cand + "_") for k in arrays_keys):
            return cand
    raise SystemExit("NPZ contains no GPU profile; nothing to plot for the GPU panel.")


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "per_window_errors.npz"
    out_dir = REPO / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "tf_pt_error_distributions.png"

    z = dict(np.load(in_path, allow_pickle=True))
    if not any(k.startswith("cpu_") for k in z):
        raise SystemExit("NPZ contains no CPU profile; expected cpu_p_mae etc.")
    gpu = _which_gpu(z.keys())

    cpu_p_max = z["cpu_p_max"].astype(np.float64)
    cpu_s_max = z["cpu_s_max"].astype(np.float64)
    gpu_p_max = z[f"{gpu}_p_max"].astype(np.float64)
    gpu_s_max = z[f"{gpu}_s_max"].astype(np.float64)
    cpu_p_mae = z["cpu_p_mae"].astype(np.float64)
    cpu_s_mae = z["cpu_s_mae"].astype(np.float64)
    gpu_p_mae = z[f"{gpu}_p_mae"].astype(np.float64)
    gpu_s_mae = z[f"{gpu}_s_mae"].astype(np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)

    # (A) CDF — CPU profile
    ax = axes[0, 0]
    for vals, color, label in [
        (cpu_p_max, "#1f77b4", "P branch"),
        (cpu_s_max, "#d62728", "S branch"),
    ]:
        v, c = _empirical_cdf(_safe(vals))
        ax.step(v, c, where="post", color=color, lw=1.6, label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\max |\mathrm{TF}-\mathrm{PT}|$ per window")
    ax.set_ylabel("Cumulative fraction of windows")
    ax.set_title("(A) Empirical CDF — CPU profile")
    ax.grid(True, which="both", alpha=0.3)
    for thr, label in [(1e-6, "1e-6"), (1e-4, "1e-4"), (1e-2, "1e-2")]:
        ax.axvline(thr, color="0.55", ls=":", lw=0.8, alpha=0.7)
    ax.legend(fontsize=9, loc="lower right")
    pct_p = (cpu_p_max < 1e-6).mean() * 100
    pct_s = (cpu_s_max < 1e-6).mean() * 100
    ax.text(0.02, 0.98,
            f"P: {pct_p:.1f}% < 1e-6\nS: {pct_s:.1f}% < 1e-6\nN={cpu_p_max.size} windows",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85, boxstyle="round,pad=0.3"))

    # (B) CDF — GPU profile
    ax = axes[0, 1]
    for vals, color, label in [
        (gpu_p_max, "#1f77b4", "P branch"),
        (gpu_s_max, "#d62728", "S branch"),
    ]:
        v, c = _empirical_cdf(_safe(vals))
        ax.step(v, c, where="post", color=color, lw=1.6, label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\max |\mathrm{TF}-\mathrm{PT}|$ per window")
    ax.set_ylabel("Cumulative fraction of windows")
    ax.set_title(f"(B) Empirical CDF — {gpu.upper()} profile")
    _log_decade_ticks(ax, axis="x")
    ax.grid(True, axis="y", which="major", alpha=0.3)
    for thr, label in [(1e-6, "1e-6"), (1e-4, "1e-4"), (1e-2, "1e-2")]:
        ax.axvline(thr, color="0.55", ls=":", lw=0.8, alpha=0.7)
    ax.legend(fontsize=9, loc="lower right")
    pct_p = (gpu_p_max < 1e-4).mean() * 100
    pct_s = (gpu_s_max < 1e-4).mean() * 100
    ax.text(0.02, 0.98,
            f"P: {pct_p:.1f}% < 1e-4\nS: {pct_s:.1f}% < 1e-4\nN={gpu_p_max.size} windows",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85, boxstyle="round,pad=0.3"))

    # (C) Violin — MAE per window, P/S × CPU/GPU
    ax = axes[1, 0]
    data = [
        np.log10(_safe(cpu_p_mae)),
        np.log10(_safe(cpu_s_mae)),
        np.log10(_safe(gpu_p_mae)),
        np.log10(_safe(gpu_s_mae)),
    ]
    parts = ax.violinplot(data, positions=[1, 2, 3, 4], showmedians=True, showextrema=False, widths=0.85)
    colors = ["#1f77b4", "#d62728", "#1f77b4", "#d62728"]
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor("0.2")
        body.set_alpha(0.55)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([f"P\nCPU", f"S\nCPU", f"P\n{gpu.upper()}", f"S\n{gpu.upper()}"], fontsize=9)
    ax.set_ylabel(r"$\log_{10}(\mathrm{MAE\ per\ window})$")
    ax.set_title("(C) MAE per window — full distribution (violin)")
    ax.grid(True, axis="y", alpha=0.3)

    # (D) Box plot — max |TF-PT| per window with outliers
    ax = axes[1, 1]
    box_data = [
        _safe(cpu_p_max),
        _safe(cpu_s_max),
        _safe(gpu_p_max),
        _safe(gpu_s_max),
    ]
    bp = ax.boxplot(box_data, positions=[1, 2, 3, 4], widths=0.55, patch_artist=True,
                    flierprops=dict(marker="o", markersize=2, markerfacecolor="0.4",
                                    markeredgecolor="0.4", alpha=0.6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor("0.2")
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([f"P\nCPU", f"S\nCPU", f"P\n{gpu.upper()}", f"S\n{gpu.upper()}"], fontsize=9)
    ax.set_ylabel(r"$\max |\mathrm{TF}-\mathrm{PT}|$ per window")
    ax.set_title("(D) Worst-point per-window mismatch (box + outliers)")
    _log_decade_ticks(ax, axis="y")

    fig.suptitle(
        "TF vs PT output discrepancy distributions on real SeisBench windows",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Wrote", out_png)


if __name__ == "__main__":
    main()
