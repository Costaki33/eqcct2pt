#!/usr/bin/env python3
"""Plot mean |TF−PT| vs lag relative to catalog arrivals (Figure 4 panel C).

Reads ``results/residual_vs_arrival_*.npz`` from
``validation.tf_pt_residual_vs_arrival``.

Usage::

    python scripts/plot_residual_vs_arrival.py results/residual_vs_arrival_cpu.npz \\
        --output figures/tf_pt_residual_vs_arrival.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from plot_panel_style import FIG_SAVE_DPI, apply_manuscript_style, panel_letter


def load_profile(npz_path: Path, profile: str = "cpu"):
    z = np.load(npz_path, allow_pickle=True)
    prefix = f"{profile}_"
    keys = {k: prefix + k for k in ("lags_seconds", "mean_abs_p", "mean_abs_s", "count_p", "n_windows")}
    if keys["lags_seconds"] not in z.files:
        # bare keys (single-profile saves)
        keys = {k: k for k in keys}
    return {k: z[v] for k, v in keys.items()}


def plot_residual_axes(ax, data: dict, *, xlim_s: float = 30.0, letter: str | None = "C") -> None:
    t = np.asarray(data["lags_seconds"], dtype=float)
    mp = np.asarray(data["mean_abs_p"], dtype=float)
    ms = np.asarray(data["mean_abs_s"], dtype=float)
    mask = (t >= -xlim_s) & (t <= xlim_s)
    ax.plot(t[mask], mp[mask], color="#1f77b4", lw=1.4, label="P branch")
    ax.plot(t[mask], ms[mask], color="#d62728", lw=1.4, ls="--", label="S branch")
    ax.axvline(0.0, color="0.35", ls=":", lw=0.9)
    ax.set_xlabel("Time relative to catalog arrival (s)")
    ax.set_ylabel(r"mean $|$TF $-$ PT$|$")
    ax.set_xlim(-xlim_s, xlim_s)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    if letter:
        panel_letter(ax, letter)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--profile", default="cpu")
    ap.add_argument("--xlim-s", type=float, default=30.0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    apply_manuscript_style()
    data = load_profile(args.npz, args.profile)
    out = args.output or Path("figures") / "tf_pt_residual_vs_arrival.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    plot_residual_axes(ax, data, xlim_s=args.xlim_s, letter=None)
    fig.savefig(out, dpi=FIG_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("[info] Wrote", out)


if __name__ == "__main__":
    main()
