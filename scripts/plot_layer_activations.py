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

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from plot_panel_style import (
    FIG_SAVE_DPI,
    apply_manuscript_style,
    panel_letter,
    panel_subtitle,
    SUBPLOTS_ADJUST_2X2_LAYER,
)


REPO = Path(__file__).resolve().parents[1]


def _clean_log_yaxis(ax, *, vals: list[float] | None = None) -> None:
    """Force readable major decade ticks (narrow log spans otherwise show none)."""
    import math

    ax.set_yscale("log")
    if vals:
        finite = [v for v in vals if v is not None and v > 0 and np.isfinite(v)]
    else:
        finite = []
        for line in ax.lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            finite.extend(y[np.isfinite(y) & (y > 0)].tolist())
        for patch in getattr(ax, "patches", []):
            try:
                h = float(patch.get_height())
                if h > 0 and np.isfinite(h):
                    finite.append(h)
            except Exception:
                pass
    if finite:
        vmin = min(finite)
        vmax = max(finite)
        lo_exp = math.floor(math.log10(vmin)) - 1
        hi_exp = math.ceil(math.log10(vmax)) + 1
        if hi_exp - lo_exp < 2:
            mid = 0.5 * (math.log10(vmin) + math.log10(vmax))
            lo_exp = math.floor(mid) - 1
            hi_exp = lo_exp + 2
        lo = 10.0 ** lo_exp
        hi = 10.0 ** hi_exp
        ax.set_ylim(lo, hi)
        ticks = [10.0 ** e for e in range(lo_exp, hi_exp + 1)]
        ax.set_yticks(ticks)
    else:
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    ax.tick_params(axis="y", which="major", labelleft=True, length=4)
    ax.grid(True, axis="y", which="major", alpha=0.3)


def _profile_label(name: str) -> str:
    return {"cpu": "CPU", "gpu0": "GPU", "gpu1": "GPU"}.get(name, name)


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

    apply_manuscript_style()
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
    fig.subplots_adjust(**SUBPLOTS_ADJUST_2X2_LAYER)

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

    panel_titles = (
        "P branch — median $|\\mathrm{TF}-\\mathrm{PT}|$",
        "S branch — median $|\\mathrm{TF}-\\mathrm{PT}|$",
        r"P branch — mean$|\mathrm{TF}-\mathrm{PT}|/(\max|\mathrm{TF}|+\varepsilon)$",
        r"S branch — mean$|\mathrm{TF}-\mathrm{PT}|/(\max|\mathrm{TF}|+\varepsilon)$",
    )

    # (A) median — P
    ax = axes[0, 0]
    med_vals: list[float] = []
    for i, r in enumerate(results):
        prof = r["profile"]
        m = []
        e = []
        for s in r["stages_p"]:
            mm, ss = _median_mean_std(s)
            m.append(mm)
            e.append(ss)
        med_vals.extend(m)
        off = (i - (len(results) - 1) / 2) * width
        _bar(ax, x_p, m, e, off, palette[("P", prof)], f"{_profile_label(prof)}")
    ax.set_xticks(x_p)
    ax.set_xticklabels(p_short, rotation=20, ha="right")
    ax.set_ylabel(r"$\mathrm{median}\;|\mathrm{TF}-\mathrm{PT}|$")
    panel_letter(ax, "A")
    panel_subtitle(ax, panel_titles[0])
    ax.legend()

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
        med_vals.extend(m)
        off = (i - (len(results) - 1) / 2) * width
        _bar(ax, x_s, m, e, off, palette[("S", prof)], f"{_profile_label(prof)}")
    ax.set_xticks(x_s)
    ax.set_xticklabels(s_short, rotation=20, ha="right")
    ax.set_ylabel(r"$\mathrm{median}\;|\mathrm{TF}-\mathrm{PT}|$")
    panel_letter(ax, "B")
    panel_subtitle(ax, panel_titles[1])
    ax.legend()
    # Shared y-limits within the median row so P vs S are comparable.
    _clean_log_yaxis(axes[0, 0], vals=med_vals)
    _clean_log_yaxis(axes[0, 1], vals=med_vals)

    # Stabilize relative error when |TF| is near zero (reviewer-suggested scale).
    rel_eps = 1e-12
    _REL_FALLBACK_WARNED = False

    def _relative_mean(stage: dict) -> tuple[float, float]:
        """mean|TF−PT| / (max|TF| + ε), so stages with different activation scales are comparable."""
        nonlocal _REL_FALLBACK_WARNED
        num = float(stage["mean_abs_diff_mean"])
        if "tf_abs_max_mean" in stage:
            denom = float(stage["tf_abs_max_mean"]) + rel_eps
        else:
            if not _REL_FALLBACK_WARNED:
                warnings.warn(
                    "layer_activations.json lacks tf_abs_max_*; normalizing by max|TF-PT| "
                    "instead of max|TF|. Re-run validation.tf_pt_layer_activations.",
                    stacklevel=2,
                )
                _REL_FALLBACK_WARNED = True
            denom = float(stage["max_abs_diff_mean"]) + rel_eps
        rel = num / denom
        rel_std = float(stage["mean_abs_diff_std"]) / denom
        return rel, rel_std

    # (C) relative mean — P
    ax = axes[1, 0]
    rel_vals: list[float] = []
    for i, r in enumerate(results):
        prof = r["profile"]
        m = []
        errs = []
        for s in r["stages_p"]:
            mm, ss = _relative_mean(s)
            m.append(mm)
            errs.append(ss)
        rel_vals.extend(m)
        off = (i - (len(results) - 1) / 2) * width
        _bar(ax, x_p, m, errs, off, palette[("P", prof)], f"{_profile_label(prof)}")
    ax.set_xticks(x_p)
    ax.set_xticklabels(p_short, rotation=20, ha="right")
    ax.set_ylabel(r"relative $\mathrm{mean}\,|\mathrm{TF}-\mathrm{PT}|$")
    panel_letter(ax, "C")
    panel_subtitle(ax, panel_titles[2])
    ax.legend()

    # (D) relative mean — S
    ax = axes[1, 1]
    for i, r in enumerate(results):
        prof = r["profile"]
        m = []
        errs = []
        for s in r["stages_s"]:
            mm, ss = _relative_mean(s)
            m.append(mm)
            errs.append(ss)
        rel_vals.extend(m)
        off = (i - (len(results) - 1) / 2) * width
        _bar(ax, x_s, m, errs, off, palette[("S", prof)], f"{_profile_label(prof)}")
    ax.set_xticks(x_s)
    ax.set_xticklabels(s_short, rotation=20, ha="right")
    ax.set_ylabel(r"relative $\mathrm{mean}\,|\mathrm{TF}-\mathrm{PT}|$")
    panel_letter(ax, "D")
    panel_subtitle(ax, panel_titles[3])
    ax.legend()
    # Shared y-limits within the relative-error row so P vs S are comparable.
    _clean_log_yaxis(axes[1, 0], vals=rel_vals)
    _clean_log_yaxis(axes[1, 1], vals=rel_vals)

    fig.savefig(out_png, dpi=FIG_SAVE_DPI, bbox_inches="tight", pad_inches=0.35)
    print("Wrote", out_png)


if __name__ == "__main__":
    main()
