"""Shared typography and layout for manuscript figures.

All multi-panel PNGs are rendered ~12 in wide but typeset at ~6.5 in
(0.97\\linewidth in an 11 pt article), i.e. printed at roughly half size.
Fonts here are sized so they remain >= ~7 pt after that reduction.
Call :func:`apply_manuscript_style` before building a figure.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.axes import Axes

# Single source of truth: letter position + typography across multi-panel PNGs.
PANEL_LETTER_FONTSIZE = 18
PANEL_TITLE_FONTSIZE = 13
FIG_SAVE_DPI = 300
PANEL_TITLE_PAD = 14
PANEL_LETTER_XY = (-0.085, 1.035)  # axes fraction, upper-left corner of panel

# Base font sizes (pre-reduction; ~half size in print).
FONT_BASE = 13
FONT_AXIS_LABEL = 14
FONT_TICK = 12
FONT_LEGEND = 11
FONT_ANNOTATION = 11

# Shared 2x2 layout (error distributions + performance benchmark).
SUBPLOTS_ADJUST_2X2 = dict(left=0.09, right=0.96, top=0.96, bottom=0.09, wspace=0.30, hspace=0.40)
# Layer activations need more vertical gap between rows and room for panel letters.
SUBPLOTS_ADJUST_2X2_LAYER = dict(left=0.09, right=0.96, top=0.92, bottom=0.12, wspace=0.30, hspace=0.62)


def apply_manuscript_style() -> None:
    """Set rcParams so every figure shares legible print-ready typography."""
    mpl.rcParams.update(
        {
            "font.size": FONT_BASE,
            "axes.labelsize": FONT_AXIS_LABEL,
            "axes.titlesize": PANEL_TITLE_FONTSIZE,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
            "axes.linewidth": 0.9,
            "savefig.dpi": FIG_SAVE_DPI,
        }
    )


def panel_letter(ax: Axes, letter: str) -> None:
    """Bold letter outside the subplot box (no parentheses)."""
    x, y = PANEL_LETTER_XY
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=PANEL_LETTER_FONTSIZE,
        fontweight="bold",
        clip_on=False,
    )


def panel_subtitle(ax: Axes, text: str) -> None:
    """Title line below the letter; no (A)/(B) prefix — use panel_letter separately."""
    ax.set_title(text, fontsize=PANEL_TITLE_FONTSIZE, pad=PANEL_TITLE_PAD)
