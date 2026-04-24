#!/usr/bin/env python3
"""Plot TF vs PT performance benchmark (inference time, RAM, VRAM, throughput).

Reads the JSON written by ``eqcct_sb.validation.tf_pt_perf_benchmark`` and writes
``figures/tf_pt_perf_benchmark.png``.

Usage::

    python scripts/plot_perf_benchmark.py results/perf_benchmark.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _clean_log_yaxis(ax) -> None:
    """Major decade ticks only on a log y-axis; grid on majors only."""
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(True, axis="y", which="major", alpha=0.3)


def _legend_upper_right(ax, *, ncol: int = 1) -> None:
    """Consistent in-axes legend in the upper-right corner."""
    leg = ax.legend(
        loc="upper right",
        ncol=ncol,
        fontsize=8,
        frameon=True,
        borderaxespad=0.5,
        handlelength=1.6,
        columnspacing=1.2,
    )
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_alpha(0.95)


def _headroom_log(ax, factor: float = 3.0) -> None:
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * factor)


def _headroom_linear(ax, factor: float = 1.3) -> None:
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * factor)

REPO = Path(__file__).resolve().parents[1]


def _profile_label(name: str) -> str:
    return {"cpu": "CPU", "gpu0": "GPU 0", "gpu1": "GPU 1"}.get(name, name)


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "perf_benchmark.json"
    out_dir = REPO / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "tf_pt_perf_benchmark.png"

    payload = json.loads(in_path.read_text())
    results = payload["results"]
    profiles = [r["profile"] for r in results]
    labels = [_profile_label(p) for p in profiles]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)

    # (A) Mean inference time per window (ms), TF vs PT, P + S
    ax = axes[0, 0]
    x = np.arange(len(profiles))
    w = 0.2
    tf_p = [r["tf_p"]["mean_s"] * 1000 for r in results]
    tf_s = [r["tf_s"]["mean_s"] * 1000 for r in results]
    pt_p = [r["pt_p"]["mean_s"] * 1000 for r in results]
    pt_s = [r["pt_s"]["mean_s"] * 1000 for r in results]
    tf_p_err = [r["tf_p"]["std_s"] * 1000 for r in results]
    tf_s_err = [r["tf_s"]["std_s"] * 1000 for r in results]
    pt_p_err = [r["pt_p"]["std_s"] * 1000 for r in results]
    pt_s_err = [r["pt_s"]["std_s"] * 1000 for r in results]
    ax.bar(x - 1.5 * w, tf_p, w, yerr=tf_p_err, label="TF P", color="#2c7bb6", capsize=3)
    ax.bar(x - 0.5 * w, tf_s, w, yerr=tf_s_err, label="TF S", color="#abd9e9", capsize=3)
    ax.bar(x + 0.5 * w, pt_p, w, yerr=pt_p_err, label="PT P", color="#d7191c", capsize=3)
    ax.bar(x + 1.5 * w, pt_s, w, yerr=pt_s_err, label="PT S", color="#fdae61", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean inference time per window (ms)")
    ax.set_yscale("log")
    ax.set_title(f"(A) Per-window inference latency (n={payload['n_windows']} windows, 60 s @ 100 Hz)")
    _clean_log_yaxis(ax)
    _headroom_log(ax, factor=5.0)
    _legend_upper_right(ax, ncol=2)

    # (B) Throughput (windows/sec), TF vs PT
    ax = axes[0, 1]
    tf_thr = [
        1.0 / r["tf_p"]["mean_s"] + 1.0 / r["tf_s"]["mean_s"]  # not used; reset below
        for r in results
    ]
    # Throughput per single forward (P alone, S alone), TF vs PT
    tf_p_thr = [r["tf_p"]["throughput_win_per_s"] for r in results]
    tf_s_thr = [r["tf_s"]["throughput_win_per_s"] for r in results]
    pt_p_thr = [r["pt_p"]["throughput_win_per_s"] for r in results]
    pt_s_thr = [r["pt_s"]["throughput_win_per_s"] for r in results]
    ax.bar(x - 1.5 * w, tf_p_thr, w, label="TF P", color="#2c7bb6")
    ax.bar(x - 0.5 * w, tf_s_thr, w, label="TF S", color="#abd9e9")
    ax.bar(x + 0.5 * w, pt_p_thr, w, label="PT P", color="#d7191c")
    ax.bar(x + 1.5 * w, pt_s_thr, w, label="PT S", color="#fdae61")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Throughput (windows / second)")
    ax.set_yscale("log")
    ax.set_title("(B) Single-stream throughput (60 s windows)")
    _clean_log_yaxis(ax)
    _headroom_log(ax, factor=5.0)
    _legend_upper_right(ax, ncol=2)

    # (C) Host RAM delta after model load (TF only, then TF+PT)
    ax = axes[1, 0]
    rss_tf = [r["rss_tf_load_delta_mb"] for r in results]
    rss_pt = [r["rss_pt_load_delta_mb"] for r in results]
    ax.bar(x - 0.2, rss_tf, 0.4, label="TF P+S load", color="#2c7bb6")
    ax.bar(x + 0.2, rss_pt, 0.4, label="PT P+S load (additional)", color="#d7191c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Host RAM increase (MB, RSS delta)")
    ax.set_title("(C) Host RAM usage after model load")
    ax.grid(True, axis="y", alpha=0.3)
    for xi, v in zip(x - 0.2, rss_tf):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + 0.2, rss_pt):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    _headroom_linear(ax, factor=1.45)
    _legend_upper_right(ax, ncol=1)

    # (D) Peak GPU VRAM after warm inference (TF vs PT). GPU profiles only.
    ax = axes[1, 1]
    gpu_results = [r for r in results if r["profile"].startswith("gpu")]
    if gpu_results:
        gpu_labels = [_profile_label(r["profile"]) for r in gpu_results]
        gx = np.arange(len(gpu_results))
        tf_vram = [r["tf_peak_vram_mb"] if r["tf_peak_vram_mb"] is not None else 0.0 for r in gpu_results]
        pt_vram = [r["pt_peak_vram_mb"] if r["pt_peak_vram_mb"] is not None else 0.0 for r in gpu_results]
        ax.bar(gx - 0.2, tf_vram, 0.4, label="TF peak VRAM", color="#2c7bb6")
        ax.bar(gx + 0.2, pt_vram, 0.4, label="PT peak VRAM", color="#d7191c")
        ax.set_xticks(gx)
        ax.set_xticklabels(gpu_labels)
        for xi, v in zip(gx - 0.2, tf_vram):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        for xi, v in zip(gx + 0.2, pt_vram):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No GPU profiles in this run", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_ylabel("Peak GPU memory after inference (MB)")
    ax.set_title("(D) Peak VRAM during inference")
    ax.grid(True, axis="y", alpha=0.3)
    _headroom_linear(ax, factor=1.45)
    if gpu_results:
        _legend_upper_right(ax, ncol=1)

    fig.suptitle(
        "EQCCT performance: TensorFlow versus PyTorch\n"
        f"({payload['n_windows']} timed windows of 6000 samples, {payload['warmup']} warmup forwards)",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Wrote", out_png)


if __name__ == "__main__":
    main()
