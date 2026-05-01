#!/usr/bin/env python3
"""Figure 6: ablation study of TF -> PT weight transfer.

Reads the NPZ written by ``validation.tf_pt_ablation`` and renders:

  * Top row (A, B, C): P-probability overlay on one representative window for
    each PT variant -- correct, no-transpose Patch Encoder, missing picker.
  * Bottom row:
        D) MAE per window across 1000 test windows (log y, violin + median
           bars) for each variant.
        E) Pick-time error (|argmax(TF) - argmax(PT)|) in samples at 100 Hz.

Uses the first available profile in the NPZ (``cpu`` preferred, falls back
to the first GPU profile).

Usage::

    python scripts/plot_ablation.py results/ablation.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parents[1]

VARIANTS = ("correct", "no_transpose", "missing_picker")
VARIANT_LABELS = {
    "correct": "A) Correct conversion",
    "no_transpose": "B) No transpose on\nPatch Encoder Dense",
    "missing_picker": "C) Missing picker\n(random init)",
}
SHORT_LABELS = {
    "correct": "A  Correct",
    "no_transpose": "B  No transpose",
    "missing_picker": "C  Missing picker",
}
VARIANT_COLORS = {
    "correct": "#2ca02c",
    "no_transpose": "#ff7f0e",
    "missing_picker": "#d62728",
}


def _pick_profile(keys) -> str:
    for cand in ("cpu", "gpu0", "gpu1"):
        if any(k.startswith(cand + "_") for k in keys):
            return cand
    raise SystemExit("NPZ has no recognised profile; expected keys like cpu_correct_mae_p.")


def _safe(x, floor=1e-16):
    return np.maximum(np.asarray(x, dtype=np.float64), floor)


def _clean_log_yaxis(ax):
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(True, axis="y", which="major", alpha=0.3)


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "ablation.npz"
    out_dir = REPO / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "tf_pt_ablation.png"

    z = dict(np.load(in_path, allow_pickle=True))
    prof = _pick_profile(z.keys())

    example_indices = sorted({
        int(k.split("_")[2]) for k in z
        if k.startswith(f"{prof}_example_") and k.endswith("_wf")
    })
    if not example_indices:
        raise SystemExit(f"No example windows stored for profile {prof}.")
    ex0 = example_indices[0]
    wf = z[f"{prof}_example_{ex0}_wf"]
    p_tf = z[f"{prof}_example_{ex0}_p_tf"]
    p_curves = {v: z[f"{prof}_example_{ex0}_p_{v}"] for v in VARIANTS}
    ds_label = str(z[f"{prof}_example_{ex0}_dataset"])
    tr_label = str(z[f"{prof}_example_{ex0}_trace"])[:30]

    mae_p = {v: z[f"{prof}_{v}_mae_p"] for v in VARIANTS}
    mae_s = {v: z[f"{prof}_{v}_mae_s"] for v in VARIANTS}
    pterr = {v: z[f"{prof}_{v}_pterr"].astype(np.float64) for v in VARIANTS}
    sterr = {v: z[f"{prof}_{v}_sterr"].astype(np.float64) for v in VARIANTS}
    n_windows = int(mae_p["correct"].size)

    fig = plt.figure(figsize=(13.5, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25])

    n_samples = p_tf.shape[-1]
    x = np.arange(n_samples)
    for col, v in enumerate(VARIANTS):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(x, p_tf, color="#1f77b4", lw=1.2, label="TF P", alpha=0.95)
        ax.plot(x, p_curves[v], color=VARIANT_COLORS[v], lw=1.1, ls="--",
                label=f"PT P ({v.replace('_', ' ')})", alpha=0.9)
        ax.set_ylim(-0.15, 1.15)
        ax.set_title(VARIANT_LABELS[v], fontsize=11)
        if col == 0:
            ax.set_ylabel(f"{ds_label.upper()} — {tr_label}\nP probability", fontsize=9)
        ax.set_xlabel("sample (in 6000-pt window)", fontsize=9)
        mae_val = float(np.mean(np.abs(p_tf - p_curves[v])))
        pe_val = int(abs(int(np.argmax(p_tf)) - int(np.argmax(p_curves[v]))))
        residual_text = (f"MAE = {mae_val:.2e}\n"
                         f"pick Δ = {pe_val} samples "
                         f"({pe_val / 100.0:.2f} s)")
        ax.text(0.02, 0.98, residual_text, transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9,
                          boxstyle="round,pad=0.3"))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(gs[1, 0:2])
    positions_p = np.arange(len(VARIANTS)) * 2.0
    positions_s = positions_p + 0.75
    log_data_p = [np.log10(_safe(mae_p[v])) for v in VARIANTS]
    log_data_s = [np.log10(_safe(mae_s[v])) for v in VARIANTS]

    for pos, log_data, v in zip(positions_p, log_data_p, VARIANTS):
        parts = ax.violinplot([log_data], positions=[pos], widths=0.65,
                              showmedians=True, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(VARIANT_COLORS[v])
            b.set_edgecolor("0.2")
            b.set_alpha(0.6)
    for pos, log_data, v in zip(positions_s, log_data_s, VARIANTS):
        parts = ax.violinplot([log_data], positions=[pos], widths=0.65,
                              showmedians=True, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(VARIANT_COLORS[v])
            b.set_edgecolor("0.2")
            b.set_alpha(0.25)

    for i, v in enumerate(VARIANTS):
        med_log_p = float(np.median(log_data_p[i]))
        med_log_s = float(np.median(log_data_s[i]))
        ax.text(positions_p[i], med_log_p, f"{10 ** med_log_p:.1e}",
                ha="center", va="bottom", fontsize=8, color="0.15")
        ax.text(positions_s[i], med_log_s, f"{10 ** med_log_s:.1e}",
                ha="center", va="bottom", fontsize=8, color="0.35")

    ticks = (positions_p + positions_s) / 2.0
    ax.set_xticks(ticks)
    ax.set_xticklabels([SHORT_LABELS[v] for v in VARIANTS], fontsize=10)
    lo = min(d.min() for d in log_data_p + log_data_s) - 0.5
    hi = max(d.max() for d in log_data_p + log_data_s) + 0.8
    ax.set_ylim(lo, hi)
    y_decades = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1)
    ax.set_yticks(y_decades)
    ax.set_yticklabels([f"$10^{{{int(k)}}}$" for k in y_decades])
    ax.set_ylabel("MAE per window  (TF vs PT, all output samples)")
    ax.set_title(f"(D) Quantitative impact of each bug on MAE  "
                 f"(n = {n_windows} windows; dark = P branch, light = S branch)")
    ax.grid(True, axis="y", which="major", alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 2])
    positions = np.arange(len(VARIANTS))
    medians_p = [float(np.median(pterr[v])) for v in VARIANTS]
    p90_p = [float(np.percentile(pterr[v], 90)) for v in VARIANTS]
    medians_s = [float(np.median(sterr[v])) for v in VARIANTS]
    p90_s = [float(np.percentile(sterr[v], 90)) for v in VARIANTS]
    w = 0.35
    bars_p = ax2.bar(positions - w / 2, medians_p, width=w,
                     color=[VARIANT_COLORS[v] for v in VARIANTS],
                     edgecolor="0.2", label="P branch (median)")
    bars_s = ax2.bar(positions + w / 2, medians_s, width=w,
                     color=[VARIANT_COLORS[v] for v in VARIANTS],
                     edgecolor="0.2", alpha=0.55, label="S branch (median)")
    ax2.errorbar(positions - w / 2, medians_p,
                 yerr=[[0] * 3, np.subtract(p90_p, medians_p)],
                 fmt="none", ecolor="0.2", capsize=3, lw=0.9,
                 label="90th percentile")
    ax2.errorbar(positions + w / 2, medians_s,
                 yerr=[[0] * 3, np.subtract(p90_s, medians_s)],
                 fmt="none", ecolor="0.2", capsize=3, lw=0.9)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([SHORT_LABELS[v] for v in VARIANTS], fontsize=10)
    ax2.set_ylabel("Pick-time error (samples @ 100 Hz)")
    ax2.set_title("(E) Arrival-time bias\n(bar = median, whisker = P90)")
    for xi, m in zip(positions - w / 2, medians_p):
        ax2.text(xi, m, f"{m:.0f}", ha="center", va="bottom", fontsize=8)
    for xi, m in zip(positions + w / 2, medians_s):
        ax2.text(xi, m, f"{m:.0f}", ha="center", va="bottom", fontsize=8,
                 color="0.3")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Ablation study: weight-transfer bugs produce catastrophic, not subtle, errors\n"
        f"(Top: representative {ds_label.upper()} window {tr_label}.   "
        f"Bottom: aggregate metrics across {n_windows} SeisBench windows, profile = {prof.upper()}.)",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Wrote", out_png)


if __name__ == "__main__":
    main()
