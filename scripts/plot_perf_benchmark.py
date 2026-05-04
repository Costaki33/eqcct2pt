#!/usr/bin/env python3
"""Plot TF vs PT performance benchmark.

Panels (A)-(B): **paired P→S timings** — for each timed window ``k``, total seconds
``tf_p[k]+tf_s[k]`` (TensorFlow) and ``pt_p[k]+pt_s[k]`` (PyTorch), matching benchmark order.
Throughput panels use ``1/(t_P + t_S)`` for each timed window (windows per second).
Strip markers lie at fixed x (categorical).

Panels (C)-(D): **grouped bars** for scalar RSS / VRAM (one value per metric per profile).

Reads ``validation.tf_pt_perf_benchmark`` JSON and writes
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

REPO = Path(__file__).resolve().parents[1]

# Bottom panels: same legend wording—both compare TF vs PT with P + S models in memory.
MEM_TF_LEGEND = "TF (P + S)"
MEM_PT_LEGEND = "PT (P + S)"


def _clean_log_yaxis(ax) -> None:
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(True, axis="y", which="major", alpha=0.3)


def _headroom_log(ax, factor: float = 3.0) -> None:
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * factor)


def _headroom_linear(ax, factor: float = 1.18) -> None:
    lo, hi = ax.get_ylim()
    ax.set_ylim(max(0, lo * 0.98), hi * factor)


def _profile_label(name: str) -> str:
    if name == "cpu":
        return "CPU"
    if name.startswith("gpu"):
        return "GPU"
    return name


def _paired_p_s_wall_s(r: dict, backend_prefix: str) -> np.ndarray:
    """Sequential P-then-S wall time per timed window ``k``, in seconds (matches benchmark loop)."""
    tp = np.asarray(r[f"{backend_prefix}_p"]["all_s"], dtype=np.float64)
    ts = np.asarray(r[f"{backend_prefix}_s"]["all_s"], dtype=np.float64)
    if tp.shape != ts.shape:
        raise ValueError(f"{backend_prefix}: P vs S timing list lengths mismatch")
    return tp + ts


def _violin_strip_one(
    ax,
    y: np.ndarray,
    *,
    position: float,
    width: float,
    facecolor: str,
    strip_alpha: float = 0.82,
    strip_size: float = 26,
) -> None:
    """KDE-shaped violin plus markers at fixed x — categorical axis, no horizontal jitter."""
    v = np.asarray(y, dtype=np.float64).ravel()
    v = np.clip(v, np.finfo(np.float64).tiny, np.inf)

    vp = ax.violinplot(
        [v],
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body in vp["bodies"]:
        body.set_facecolor(facecolor)
        body.set_edgecolor("0.35")
        body.set_linewidth(0.8)
        body.set_alpha(0.52)
    if vp.get("cmedians") is not None:
        vp["cmedians"].set_color("0.1")
        vp["cmedians"].set_linewidth(1.15)

    ax.scatter(
        np.full(v.shape[0], position, dtype=np.float64),
        v,
        s=strip_size,
        c=facecolor,
        edgecolors="0.15",
        linewidths=0.35,
        alpha=strip_alpha,
        zorder=5,
    )


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "perf_benchmark.json"
    out_dir = REPO / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "tf_pt_perf_benchmark.png"

    payload = json.loads(in_path.read_text())
    all_results = payload["results"]
    # Filter: keep CPU and only the first GPU (gpu0) — call it just "GPU"
    results = []
    seen_gpu = False
    for r in all_results:
        if r["profile"] == "cpu":
            results.append(r)
        elif r["profile"].startswith("gpu") and not seen_gpu:
            results.append(r)
            seen_gpu = True
    labels = [_profile_label(r["profile"]) for r in results]

    top_backends = (("TF", "tf"), ("PT", "pt"))
    top_colors = ("#2c7bb6", "#d7191c")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.96, top=0.96, bottom=0.08, wspace=0.32, hspace=0.38)

    n_prof = len(results)
    n_pair = len(top_backends)
    offsets = np.linspace(-0.18, 0.18, n_pair)
    violin_width = min(0.24, abs(offsets[1] - offsets[0]) * 0.75 if n_pair > 1 else 0.24)
    h_top = [
        plt.Rectangle((0, 0), 1, 1, fc=top_colors[0], ec="0.25", alpha=0.72),
        plt.Rectangle((0, 0), 1, 1, fc=top_colors[1], ec="0.25", alpha=0.72),
    ]
    top_labels = [top_backends[0][0], top_backends[1][0]]

    # (A) P→S wall time summed per timed window — matches benchmark iteration order (ms).
    ax = axes[0, 0]
    for i, r in enumerate(results):
        for j, (_, prefix) in enumerate(top_backends):
            ms = _paired_p_s_wall_s(r, prefix) * 1000.0
            pos = float(i + offsets[j])
            _violin_strip_one(ax, ms, position=pos, width=float(violin_width), facecolor=top_colors[j])

    ax.set_xticks(np.arange(n_prof))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Sequential P$\to$S latency (ms)")
    ax.set_yscale("log")
    _clean_log_yaxis(ax)
    _headroom_log(ax, factor=3.2)
    ax.set_xlim(-0.55, n_prof - 0.45 + 0.05)
    if n_prof > 1:
        ax.axvline(0.5, color="0.5", ls="--", lw=1.0, alpha=0.7)
    ax.legend(h_top, top_labels, loc="upper right", fontsize=8, ncol=1, frameon=True)

    # (B) Throughput = inverse total P→S time for that window — consistent with (A).
    ax = axes[0, 1]
    for i, r in enumerate(results):
        for j, (_, prefix) in enumerate(top_backends):
            combo_s = _paired_p_s_wall_s(r, prefix)
            thr = 1.0 / np.maximum(combo_s, 1e-12)
            pos = float(i + offsets[j])
            _violin_strip_one(ax, thr, position=pos, width=float(violin_width), facecolor=top_colors[j])

    ax.set_xticks(np.arange(n_prof))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Throughput (windows$\,$s$^{-1}$)")
    ax.set_yscale("log")
    _clean_log_yaxis(ax)
    _headroom_log(ax, factor=3.2)
    ax.set_xlim(-0.55, n_prof - 0.45 + 0.05)
    if n_prof > 1:
        ax.axvline(0.5, color="0.5", ls="--", lw=1.0, alpha=0.7)
    ax.legend(h_top, top_labels, loc="upper right", fontsize=8, ncol=1, frameon=True)

    # (C) Host RAM deltas — grouped bars
    ax = axes[1, 0]
    ram_bar_w = min(0.14, (offsets[-1] - offsets[0] + violin_width * 2) / 4.8)
    x_tf = np.arange(n_prof) - ram_bar_w * 0.55
    x_pt = np.arange(n_prof) + ram_bar_w * 0.55
    rss_tf = [float(r["rss_tf_load_delta_mb"]) for r in results]
    rss_pt = [float(r["rss_pt_load_delta_mb"]) for r in results]
    ax.bar(x_tf, rss_tf, width=ram_bar_w, label=MEM_TF_LEGEND, color="#2c7bb6", edgecolor="0.28", linewidth=0.6, alpha=0.88)
    ax.bar(x_pt, rss_pt, width=ram_bar_w, label=MEM_PT_LEGEND, color="#d7191c", edgecolor="0.28", linewidth=0.6, alpha=0.88)
    for xi, v in zip(x_tf, rss_tf):
        ax.text(float(xi), v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="0.25")
    for xi, v in zip(x_pt, rss_pt):
        ax.text(float(xi), v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="0.25")
    ax.set_xticks(np.arange(n_prof))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Host RAM increase after model load (MB, RSS Δ)")
    ax.grid(True, axis="y", alpha=0.33)
    _headroom_linear(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    # (D) GPU VRAM peaks — grouped bars (single GPU only)
    ax = axes[1, 1]
    gpu_results = [r for r in results if r["profile"].startswith("gpu")]
    if gpu_results:
        r = gpu_results[0]  # single GPU
        tf_v = 0.0 if r["tf_peak_vram_mb"] is None else float(r["tf_peak_vram_mb"])
        pt_v = 0.0 if r["pt_peak_vram_mb"] is None else float(r["pt_peak_vram_mb"])
        bar_x = np.array([0, 1])
        bar_vals = [tf_v, pt_v]
        bar_colors = ["#2c7bb6", "#d7191c"]
        ax.bar(bar_x, bar_vals, width=0.55, color=bar_colors, edgecolor="0.28", linewidth=0.6, alpha=0.88)
        for xi, v in zip(bar_x, bar_vals):
            ax.text(float(xi), v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color="0.25")
        ax.set_xticks(bar_x)
        ax.set_xticklabels(["TF", "PT"], fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        h = [
            plt.Rectangle((0, 0), 1, 1, fc=bar_colors[0], ec="0.28", alpha=0.88),
            plt.Rectangle((0, 0), 1, 1, fc=bar_colors[1], ec="0.28", alpha=0.88),
        ]
        ax.legend(h, [MEM_TF_LEGEND, MEM_PT_LEGEND], loc="upper right", fontsize=8, frameon=True)
    else:
        ax.text(0.5, 0.5, "No GPU profiles in this run", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])

    ax.set_ylabel("GPU VRAM during inference (MB, peak)")
    ax.grid(True, axis="y", alpha=0.33)
    _headroom_linear(ax)

    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.28)
    print("Wrote", out_png)


if __name__ == "__main__":
    main()
