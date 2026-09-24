#!/usr/bin/env python3
"""2x2 error-distribution figure for the TF→PT manuscript.

Reads the NPZ produced by ``validation.tf_pt_per_window_errors`` and
writes ``figures/tf_pt_error_distributions.png``.

Panels (notation matches manuscript Eqs. 1 and 3):
  A: empirical CDFs of ``Δ_max,w`` for the P branch, CPU and GPU overlaid
  B: same for the S branch
  C: violins of ``log10(MAE_w)``
  D: log–log scatter of ``Δ_max,w`` vs ``MAE_w``

Usage::

    python scripts/plot_error_distributions.py results/per_window_errors.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from plot_panel_style import (
    FIG_SAVE_DPI,
    FONT_ANNOTATION,
    apply_manuscript_style,
    panel_letter,
    panel_subtitle,
    SUBPLOTS_ADJUST_2X2,
)

REF_LINES = (1e-6, 1e-4, 1e-2)
REF_COLOR = "#555555"

REPO = Path(__file__).resolve().parents[1]


def _log_decade_ticks(ax, axis: str = "y") -> None:
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(mticker.LogLocator(base=10.0))
    target.set_minor_locator(mticker.NullLocator())
    ax.grid(True, axis=axis, which="major", alpha=0.3)


def _empirical_cdf(values: np.ndarray):
    v = np.sort(values)
    return v, np.arange(1, v.size + 1) / v.size


def _safe(x: np.ndarray, floor: float = 1e-16) -> np.ndarray:
    return np.maximum(x, floor)


def _which_gpu(arrays_keys) -> str:
    for cand in ("gpu0", "gpu1"):
        if any(k.startswith(cand + "_") for k in arrays_keys):
            return cand
    raise SystemExit("NPZ contains no GPU profile; nothing to plot for the GPU series.")


def _cdf_overlay(ax, cpu_max: np.ndarray, gpu_max: np.ndarray, color: str) -> None:
    """Overlay CPU (solid) and GPU (dashed) CDFs for one branch."""
    for vals, ls, label in [
        (cpu_max, "-", "CPU"),
        (gpu_max, "--", "GPU"),
    ]:
        v, c = _empirical_cdf(_safe(vals))
        ax.step(v, c, where="post", color=color, lw=1.8, ls=ls, label=label)
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ax.set_xlim(min(lo, 1e-7), max(hi, 3e-2))
    ax.set_ylim(0.0, 1.0)
    for thr in REF_LINES:
        ax.axvline(thr, color=REF_COLOR, ls=":", lw=1.1, alpha=0.85, zorder=1)
        ax.text(
            thr,
            0.92,
            rf"$10^{{{int(np.log10(thr))}}}$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=FONT_ANNOTATION,
            color=REF_COLOR,
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2),
        )
    ax.set_xlabel(r"$\Delta_{\max,w} = \max_t |\mathrm{TF}-\mathrm{PT}|$ per window")
    ax.set_ylabel("Cumulative fraction of windows")
    _log_decade_ticks(ax, axis="x")
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        fontsize=FONT_ANNOTATION,
        handlelength=2.2,
        borderpad=0.4,
        labelspacing=0.35,
    )


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "per_window_errors.npz"
    out_dir = REPO / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "tf_pt_error_distributions.png"

    apply_manuscript_style()

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

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4), constrained_layout=False)
    fig.subplots_adjust(**SUBPLOTS_ADJUST_2X2)

    _cdf_overlay(axes[0, 0], cpu_p_max, gpu_p_max, "#1f77b4")
    panel_letter(axes[0, 0], "A")
    panel_subtitle(axes[0, 0], "P branch — CPU vs GPU")

    _cdf_overlay(axes[0, 1], cpu_s_max, gpu_s_max, "#d62728")
    panel_letter(axes[0, 1], "B")
    panel_subtitle(axes[0, 1], "S branch — CPU vs GPU")

    group_labels = ["P\nCPU", "S\nCPU", "P\nGPU", "S\nGPU"]
    colors = ["#1f77b4", "#d62728", "#1f77b4", "#d62728"]

    # (C) Violin plot — log10(MAE_w)
    ax = axes[1, 0]
    data = [
        np.log10(_safe(cpu_p_mae)),
        np.log10(_safe(cpu_s_mae)),
        np.log10(_safe(gpu_p_mae)),
        np.log10(_safe(gpu_s_mae)),
    ]
    parts = ax.violinplot(data, positions=[1, 2, 3, 4], showmedians=True, showextrema=False, widths=0.85)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor("0.2")
        body.set_alpha(0.55)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(r"$\log_{10}(\mathrm{MAE}_w)$  (per-window MAE, Eq. 1)")
    ax.grid(True, axis="y", alpha=0.3)
    panel_letter(ax, "C")

    # (D) Scatter — Δ_max,w vs MAE_w
    ax = axes[1, 1]
    series = [
        (cpu_p_mae, cpu_p_max, "#1f77b4", "o", "P CPU"),
        (cpu_s_mae, cpu_s_max, "#d62728", "o", "S CPU"),
        (gpu_p_mae, gpu_p_max, "#1f77b4", "^", "P GPU"),
        (gpu_s_mae, gpu_s_max, "#d62728", "^", "S GPU"),
    ]
    for mae, dmax, color, marker, label in series:
        ax.scatter(
            _safe(mae),
            _safe(dmax),
            s=12,
            alpha=0.35,
            color=color,
            marker=marker,
            edgecolors="none",
            label=label,
            rasterized=True,
        )
    lim_lo = 1e-16
    lim_hi = 1e-1
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="0.5", ls="--", lw=1.0, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-12, 1e-3)
    ax.set_ylim(1e-10, 1e-1)
    ax.set_xlabel(r"$\mathrm{MAE}_w$  (per-window MAE, Eq. 1)")
    ax.set_ylabel(r"$\Delta_{\max,w}$  (per-window max $|$TF$-$PT$|$, Eq. 3)")
    _log_decade_ticks(ax, axis="x")
    _log_decade_ticks(ax, axis="y")
    ax.legend(loc="lower right", fontsize=FONT_ANNOTATION, markerscale=1.4)
    panel_letter(ax, "D")

    fig.savefig(out_png, dpi=FIG_SAVE_DPI, bbox_inches="tight")
    print("Wrote", out_png)


if __name__ == "__main__":
    main()
