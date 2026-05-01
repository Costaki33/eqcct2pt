#!/usr/bin/env python3
"""Plot layer-by-layer activation diff (TF vs PT, P + S branches, CPU + GPU).

Reads the JSON written by ``validation.tf_pt_layer_activations`` and
writes ``figures/tf_pt_layer_activations.png``.

Usage::

    python scripts/plot_layer_activations.py results/layer_activations.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _clean_log_yaxis(ax) -> None:
    """Decade-only major ticks (10^-N), no minor ticks, grid on majors."""
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(True, axis="y", which="major", alpha=0.3)

REPO = Path(__file__).resolve().parents[1]


def _profile_label(name: str) -> str:
    return {"cpu": "CPU", "gpu0": "GPU 0", "gpu1": "GPU 1"}.get(name, name)


def _palette():
    return {
        ("P", "cpu"): "#1f77b4",
        ("P", "gpu0"): "#9ecae1",
        ("P", "gpu1"): "#3182bd",
        ("S", "cpu"): "#d62728",
        ("S", "gpu0"): "#fdae6b",
        ("S", "gpu1"): "#e6550d",
    }


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "layer_activations.json"
    out_dir = REPO / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "tf_pt_layer_activations.png"

    payload = json.loads(in_path.read_text())
    results = payload["results"]
    palette = _palette()

    # All P branches share the same stage list and short labels; same for S.
    p_short = [r["short"] for r in results[0]["stages_p"]]
    s_short = [r["short"] for r in results[0]["stages_s"]]
    x_p = np.arange(len(p_short))
    x_s = np.arange(len(s_short))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), constrained_layout=True)

    profiles = [r["profile"] for r in results]
    n_groups = max(1, 2 * len(profiles))
    width = 0.8 / n_groups

    def _bar(ax, x, vals, errs, off, color, label):
        ax.bar(x + off, vals, width=width * 0.95, yerr=errs, label=label,
               color=color, capsize=2, error_kw=dict(elinewidth=0.7))

    # (A) max |Δ| — P branch
    ax = axes[0, 0]
    for i, r in enumerate(results):
        prof = r["profile"]
        m = [s["max_abs_diff_mean"] for s in r["stages_p"]]
        e = [s["max_abs_diff_std"] for s in r["stages_p"]]
        off = (i - (len(results) - 1) / 2) * width
        _bar(ax, x_p, m, e, off, palette[("P", prof)], f"{_profile_label(prof)}")
    ax.set_xticks(x_p)
    ax.set_xticklabels(p_short, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$\max |\mathrm{TF}-\mathrm{PT}|$ (per stage)")
    ax.set_yscale("log")
    ax.set_title("(A) P branch — max activation discrepancy at each checkpoint")
    ax.legend(fontsize=8)
    _clean_log_yaxis(ax)

    # (B) max |Δ| — S branch
    ax = axes[0, 1]
    for i, r in enumerate(results):
        prof = r["profile"]
        m = [s["max_abs_diff_mean"] for s in r["stages_s"]]
        e = [s["max_abs_diff_std"] for s in r["stages_s"]]
        off = (i - (len(results) - 1) / 2) * width
        _bar(ax, x_s, m, e, off, palette[("S", prof)], f"{_profile_label(prof)}")
    ax.set_xticks(x_s)
    ax.set_xticklabels(s_short, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$\max |\mathrm{TF}-\mathrm{PT}|$ (per stage)")
    ax.set_yscale("log")
    ax.set_title("(B) S branch — max activation discrepancy at each checkpoint")
    ax.legend(fontsize=8)
    _clean_log_yaxis(ax)

    # (C) mean |Δ| — P branch
    ax = axes[1, 0]
    for i, r in enumerate(results):
        prof = r["profile"]
        m = [s["mean_abs_diff_mean"] for s in r["stages_p"]]
        e = [s["mean_abs_diff_std"] for s in r["stages_p"]]
        off = (i - (len(results) - 1) / 2) * width
        _bar(ax, x_p, m, e, off, palette[("P", prof)], f"{_profile_label(prof)}")
    ax.set_xticks(x_p)
    ax.set_xticklabels(p_short, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$\mathrm{mean}\;|\mathrm{TF}-\mathrm{PT}|$ (per stage)")
    ax.set_yscale("log")
    ax.set_title("(C) P branch — mean activation discrepancy at each checkpoint")
    ax.legend(fontsize=8)
    _clean_log_yaxis(ax)

    # (D) mean |Δ| — S branch
    ax = axes[1, 1]
    for i, r in enumerate(results):
        prof = r["profile"]
        m = [s["mean_abs_diff_mean"] for s in r["stages_s"]]
        e = [s["mean_abs_diff_std"] for s in r["stages_s"]]
        off = (i - (len(results) - 1) / 2) * width
        _bar(ax, x_s, m, e, off, palette[("S", prof)], f"{_profile_label(prof)}")
    ax.set_xticks(x_s)
    ax.set_xticklabels(s_short, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$\mathrm{mean}\;|\mathrm{TF}-\mathrm{PT}|$ (per stage)")
    ax.set_yscale("log")
    ax.set_title("(D) S branch — mean activation discrepancy at each checkpoint")
    ax.legend(fontsize=8)
    _clean_log_yaxis(ax)

    fig.suptitle(
        "Activation discrepancy at each architectural checkpoint\n"
        f"({payload['n_seeds']} random ZNE inputs per profile; bars are mean across inputs, error = std)",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Wrote", out_png)


if __name__ == "__main__":
    main()
