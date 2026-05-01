"""Filesystem locations for weights and artifacts (repository root).

``paths.py`` lives at the project root next to ``conversion/``, ``validation/``, …
"""

from __future__ import annotations

from pathlib import Path

# Repository root (parent of conversion/, validation/, ModelPS/, …)
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PACKAGE_ROOT

# Bundled/default Keras H5 checkpoints and exported PyTorch weights
MODELPS_DIR: Path = PACKAGE_ROOT / "ModelPS"
