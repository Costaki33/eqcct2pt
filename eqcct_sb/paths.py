"""Canonical filesystem locations for the ``eqcct_sb`` package."""

from __future__ import annotations

from pathlib import Path

# Directory containing ``conversion/``, ``validation/``, etc.
PACKAGE_ROOT: Path = Path(__file__).resolve().parent

# Git repo root when this repo layout is EQCCT_to_Seisbench/eqcct_sb/
REPO_ROOT: Path = PACKAGE_ROOT.parent

# Default Keras H5 checkpoints and exported PyTorch weights for EQCCT
MODELPS_DIR: Path = PACKAGE_ROOT / "ModelPS"
