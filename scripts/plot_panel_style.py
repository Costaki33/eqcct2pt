"""Shared subplot panel letters (A, B, C, …) and optional subtitles for manuscript figures."""

from __future__ import annotations

from matplotlib.axes import Axes

# Single source of truth: letter position + typography across multi-panel PNGs.
PANEL_LETTER_FONTSIZE = 13
PANEL_TITLE_FONTSIZE = 10
PANEL_TITLE_PAD = 14
PANEL_LETTER_XY = (-0.085, 1.035)  # axes fraction, upper-left corner of panel

# Shared 2×2 layout (error distributions + performance benchmark).
SUBPLOTS_ADJUST_2X2 = dict(left=0.09, right=0.96, top=0.96, bottom=0.08, wspace=0.30, hspace=0.38)
# Layer activations need more vertical gap between rows and room for panel letters.
SUBPLOTS_ADJUST_2X2_LAYER = dict(left=0.09, right=0.96, top=0.92, bottom=0.11, wspace=0.30, hspace=0.50)


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
