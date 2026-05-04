#!/usr/bin/env python3
"""Plot layer-by-layer activation diff (TF vs PT, P + S branches, CPU + GPU).

Reads the JSON written by ``validation.tf_pt_layer_activations`` and
writes ``figures/tf_pt_layer_activations.png``.

If ``median_*`` keys are absent (legacy JSON), the script warns and plots an
approximate median row using ``√(mean_abs·max_abs)`` per checkpoint; for
publication-grade numbers, regenerate with ``validation.tf_pt_layer_activations``.

Usage::

    python scripts/plot_layer_activations.py results/layer_activations.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _clean_log_yaxis(ax) -> None:
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(True, axis="y", which="major", alpha=0.3)


REPO = Path(__file__).resolve().parents[1]


def _profile_label(name: str) -> str:
    return {"cpu": "CPU", "gpu0": "GPU profile", "gpu1": "GPU profile"}.get(name, name)


def _pretty_stage_label(short_label: str) -> str:
    """Expand legacy abbreviations baked into JSON (TBk = transformer block index k)."""
    if short_label.startswith("TB") and len(short_label) > 2 and short_label[2:].isdigit():
        return f"Transformer blk {short_label[2:]}"
    return short_label


def _palette():
    return {
        ("P", "cpu"): "#1f77b4",
        ("P", "gpu0"): "#9ecae1",
        ("P", "gpu1"): "#3182bd",
        ("S", "cpu"): "#d62728",
        ("S", "gpu0"): "#fdae6b",
        ("S", "gpu1"): "#e6550d",
    }


_MED_FALLBACK_WARNED = False


def _median_mean_std(stage: dict) -> tuple[float, float]:
    """Mean and std across seeds of the per-seed spatial median over |TF−PT|."""
    if "median_abs_diff_mean" in stage:
        return float(stage["median_abs_diff_mean"]), float(stage["median_abs_diff_std"])
    mx = float(stage["max_abs_diff_mean"])
    me = float(stage["mean_abs_diff_mean"])
    sx = float(stage["max_abs_diff_std"])
    sm = float(stage["mean_abs_diff_std"])
    global _MED_FALLBACK_WARNED
    if not _MED_FALLBACK_WARNED:
        warnings.warn(
            "layer_activations.json lacks median_abs_diff_*; using √(mean_abs·max_abs) as a plotting "
            "fallback. Re-run validation.tf_pt_layer_activations for exact medians.",
            stacklevel=2,
        )
        _MED_FALLBACK_WARNED = True
    est = float(np.sqrt(max(mx * me, 1e-320)))
    if mx > 0 and me > 0:
        r = mx / me
        estr = 0.5 * (sx * np.sqrt(me / mx + 1e-320) + sm * np.sqrt(r + 1e-320))
    else:
        estr = max(0.0, (sx + sm) * 0.25)
    return est, float(max(estr, 1e-320))


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "layer_activations.json"
    out_dir = REPO / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "tf_pt_layer_activations.png"

    payload = json.loads(in_path.read_text())
    results = payload["results"]
    palette = _palette()

    p_short = [_pretty_stage_label(r["short"]) for r in results[0]["stages_p"]]
    s_short = [_pretty_stage_label(r["short"]) for r in results[0]["stages_s"]]
    x_p = np.arange(len(p_short))
    x_s = np.arange(len(s_short))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 8.35),
        constrained_layout=False,
    )
    # Extra margins + wide row gap between median row (top) and mean row (bottom).
    fig.subplots_adjust(left=0.09, right=0.96, top=0.90, bottom=0.11, wspace=0.30, hspace=0.50)

    n_groups = max(1, 2 * len(results))
    width = 0.8 / n_groups

    def _bar(ax, x, vals, errs, off, color, label):
        ax.bar(
            x + off,
            vals,
            width=width * 0.95,
            yerr=errs,
            label=label,
            color=color,
            capsize=2,
            error_kw=dict(elinewidth=0.7),
        )

    panel_labels = (
        "(A) P branch — Median activation discrepancy per checkpoint",
        "(B) S branch — Median activation discrepancy per checkpoint",
        "(C) P branch — Mean activation discrepancy per checkpoint",
        "(D) S branch — Mean activation discrepancy per checkpoint",
    )

    # (A) median — P
    ax = axes[0, 0]
    for i, r in enumerate(results):
        prof = r["profile"]
        m = []
        e = []
        for s in r["stages_p"]:
            mm, ss = _median_mean_std(s)
            m.append(mm)
            e.append(ss)
        off = (i - (len(results) - 1) / 2) * width
        _bar(ax, x_p, m, e, off, palette[("P", prof)], f"{_profile_label(prof)}")
    ax.set_xticks(x_p)
    ax.set_xticklabels(p_short, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$\mathrm{median}\;|\mathrm{TF}-\mathrm{PT}|$")
    ax.set_yscale("log")
    ax.set_title(panel_labels[0], fontsize=10, pad=14)
    ax.legend(fontsize=8)
    _clean_log_yaxis(ax)

    # (B) median — S
    ax = axes[0, 1]
    for i, r in enumerate(results):
        prof = r["profile"]
        m = []
        e = []
        for s in r["stages_s"]:
            mm, ss = _median_mean_std(s)
            m.append(mm)
            e.append(ss)
        off = (i - (len(results) - 1) / 2) * width
        _bar(ax, x_s, m, e, off, palette[("S", prof)], f"{_profile_label(prof)}")
    ax.set_xticks(x_s)
    ax.set_xticklabels(s_short, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$\mathrm{median}\;|\mathrm{TF}-\mathrm{PT}|$")
    ax.set_yscale("log")
    ax.set_title(panel_labels[1], fontsize=10, pad=14)
    ax.legend(fontsize=8)
    _clean_log_yaxis(ax)

    # (C) mean — P
    ax = axes[1, 0]
    for i, r in enumerate(results):
        prof = r["profile"]
        m = [s["mean_abs_diff_mean"] for s in r["stages_p"]]
        errs = [s["mean_abs_diff_std"] for s in r["stages_p"]]
        off = (i - (len(results) - 1) / 2) * width
        _bar(ax, x_p, m, errs, off, palette[("P", prof)], f"{_profile_label(prof)}")
    ax.set_xticks(x_p)
    ax.set_xticklabels(p_short, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$\mathrm{mean}\;|\mathrm{TF}-\mathrm{PT}|$")
    ax.set_yscale("log")
    ax.set_title(panel_labels[2], fontsize=10, pad=14)
    ax.legend(fontsize=8)
    _clean_log_yaxis(ax)

    # (D) mean — S
    ax = axes[1, 1]
    for i, r in enumerate(results):
        prof = r["profile"]
        m = [s["mean_abs_diff_mean"] for s in r["stages_s"]]
        errs = [s["mean_abs_diff_std"] for s in r["stages_s"]]
        off = (i - (len(results) - 1) / 2) * width
        _bar(ax, x_s, m, errs, off, palette[("S", prof)], f"{_profile_label(prof)}")
    ax.set_xticks(x_s)
    ax.set_xticklabels(s_short, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$\mathrm{mean}\;|\mathrm{TF}-\mathrm{PT}|$")
    ax.set_yscale("log")
    ax.set_title(panel_labels[3], fontsize=10, pad=14)
    ax.legend(fontsize=8)
    _clean_log_yaxis(ax)

    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.35)
    print("Wrote", out_png)


if __name__ == "__main__":
    main()
