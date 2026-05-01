#!/usr/bin/env python3
"""Render the EQCCT TensorFlow -> PyTorch architecture mapping as a figure.

Three columns: TensorFlow/Keras | Transform | PyTorch.  Each row is a
stage (Input, Conv1D stack, S-branch extra Conv, Patch Encoder, Transformer
block, Picker, Output).  The middle column labels the exact tensor
transformation applied during weight transfer, with warning glyphs for the
most common pitfalls (dense transpose, square MHA projections, picker
Conv1D load).

Writes ``figures/tf_pt_architecture.png``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parents[1]

# ---- Colours -----------------------------------------------------------------
TF_FILL = "#e8f1fb"
TF_EDGE = "#2c7bb6"
PT_FILL = "#fdecea"
PT_EDGE = "#d7191c"
MID_FILL = "#fff7e6"
MID_EDGE = "#c88319"
WARN = "#c0392b"

# ---- Stage specification -----------------------------------------------------
# Each row: (tf_title, tf_sub, mid_label, mid_warn, pt_title, pt_sub)
STAGES = [
    (
        "Input",
        "tensor  (batch, 6000, 3)",
        "Permute axes",
        "",
        "Input",
        "tensor  (batch, 3, 6000)",
    ),
    (
        "Conv1D stack",
        "kernel  (k, C_in, C_out)",
        "Reorder kernel axes\nto  (C_out, C_in, k)",
        "",
        "Conv1d stack",
        "weight  (C_out, C_in, k)",
    ),
    (
        "S-branch extra Conv1D",
        "pre / post blocks  (S only)",
        "Copy into\nextra_pre / extra_post",
        "",
        "extra_pre / extra_post",
        "Conv1d blocks",
    ),
    (
        "Patch Encoder",
        "Dense  kernel (in, out)",
        "Transpose to  (out, in)",
        "CRITICAL",
        "Patch Encoder",
        "Linear  weight (out, in)",
    ),
    (
        "Transformer \u00d7N  (MHA)",
        "q / k / v / o  3-D kernels",
        "Reshape + transpose\nto 2-D Linear layers",
        "SQUARE",
        "Transformer \u00d7N  (MHA)",
        "q, k, v, o  Linear weights",
    ),
    (
        "Transformer \u00d7N  (MLP)",
        "Dense  40 \u00d7 40",
        "Transpose to  (40, 40)",
        "SQUARE",
        "Transformer \u00d7N  (MLP)",
        "fc1 / fc2  Linear weights",
    ),
    (
        "picker_P / picker_S",
        "Conv1D  kernel (k, C_in, 1)",
        "Reorder to  (1, C_in, k)\nmust load explicitly",
        "EXPLICIT",
        "Picker head",
        "Conv1d  weight (1, C_in, k)",
    ),
    (
        "Output",
        "(batch, 6000, 1)",
        "Permute axes",
        "",
        "Output",
        "(batch, 1, 6000)",
    ),
]


def _box(ax, x, y, w, h, *, fill, edge, text_top, text_bottom, top_fs=10, bot_fs=8.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.1, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    cx = x + w / 2
    ax.text(cx, y + h * 0.64, text_top, ha="center", va="center",
            fontsize=top_fs, fontweight="bold", color="0.15")
    ax.text(cx, y + h * 0.28, text_bottom, ha="center", va="center",
            fontsize=bot_fs, color="0.25")


def _mid_box(ax, x, y, w, h, *, label, warn):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, edgecolor=MID_EDGE, facecolor=MID_FILL,
    )
    ax.add_patch(patch)
    cx = x + w / 2
    cy = y + h / 2
    if warn:
        ax.text(cx, y + h * 0.72, label, ha="center", va="center",
                fontsize=8.5, color="0.20")
        ax.text(cx, y + h * 0.26, f"\u26a0  {warn}", ha="center", va="center",
                fontsize=8.5, color=WARN, fontweight="bold")
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8.8, color="0.20")


def _down_arrow(ax, x, y_top, y_bot, *, color="0.35"):
    ax.add_patch(FancyArrowPatch(
        (x, y_top), (x, y_bot),
        arrowstyle="-|>", mutation_scale=11, lw=1.1, color=color,
    ))


def _h_arrow(ax, x0, x1, y, *, color="0.35"):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y),
        arrowstyle="-|>", mutation_scale=10, lw=1.0, color=color,
    ))


def main() -> None:
    out_dir = REPO / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "tf_pt_architecture.png"

    n = len(STAGES)
    row_h = 1.1
    row_gap = 0.35
    col_w = 3.8
    mid_w = 3.2
    left_x = 0.0
    mid_x = left_x + col_w + 0.9
    right_x = mid_x + mid_w + 0.9
    total_w = right_x + col_w

    header_h = 0.55
    legend_h = 0.55
    top_pad = 0.25
    bot_pad = 0.25
    rows_h = n * row_h + (n - 1) * row_gap
    total_h = top_pad + header_h + 0.25 + rows_h + 0.25 + legend_h + bot_pad

    fig, ax = plt.subplots(figsize=(13.5, total_h * 0.92))
    ax.set_xlim(-0.3, total_w + 0.3)
    ax.set_ylim(0, total_h)
    ax.set_aspect("equal")
    ax.axis("off")

    header_center_y = total_h - top_pad - header_h / 2
    sep_y = header_center_y - header_h / 2 - 0.05
    ax.text(left_x + col_w / 2, header_center_y, "TensorFlow / Keras",
            ha="center", va="center", fontsize=13, fontweight="bold", color=TF_EDGE)
    ax.text(mid_x + mid_w / 2, header_center_y, "Weight Transform",
            ha="center", va="center", fontsize=13, fontweight="bold", color=MID_EDGE)
    ax.text(right_x + col_w / 2, header_center_y, "PyTorch",
            ha="center", va="center", fontsize=13, fontweight="bold", color=PT_EDGE)
    ax.add_patch(Rectangle((-0.25, sep_y), total_w + 0.5, 0.02,
                           facecolor="0.4", edgecolor="none"))

    rows_top = sep_y - 0.15
    for i, (tf_t, tf_s, mid_label, warn, pt_t, pt_s) in enumerate(STAGES):
        y = rows_top - (i + 1) * row_h - i * row_gap
        _box(ax, left_x, y, col_w, row_h,
             fill=TF_FILL, edge=TF_EDGE, text_top=tf_t, text_bottom=tf_s)
        _box(ax, right_x, y, col_w, row_h,
             fill=PT_FILL, edge=PT_EDGE, text_top=pt_t, text_bottom=pt_s)
        _mid_box(ax, mid_x, y + 0.05, mid_w, row_h - 0.1,
                 label=mid_label, warn=warn)
        _h_arrow(ax, left_x + col_w + 0.08, mid_x - 0.08, y + row_h / 2)
        _h_arrow(ax, mid_x + mid_w + 0.08, right_x - 0.08, y + row_h / 2)
        if i < n - 1:
            y_next = y - row_gap
            _down_arrow(ax, left_x + col_w / 2, y, y_next, color=TF_EDGE)
            _down_arrow(ax, right_x + col_w / 2, y, y_next, color=PT_EDGE)

    legend_y = bot_pad + legend_h / 2
    ax.text(total_w / 2, legend_y,
            "\u26a0  Common conversion pitfall        \u2192  Required tensor transformation",
            ha="center", va="center", fontsize=10, color="0.25",
            bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.4"))

    fig.suptitle(
        "EQCCT weight-transfer mapping: TensorFlow/Keras  \u2192  PyTorch",
        fontsize=14, y=0.995,
    )
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print("Wrote", out_png)


if __name__ == "__main__":
    main()
