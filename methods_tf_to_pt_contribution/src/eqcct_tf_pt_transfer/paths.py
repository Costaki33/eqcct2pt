"""Default artifact locations for the methods bundle."""

from __future__ import annotations

from pathlib import Path

# ``methods_tf_to_pt_contribution/`` folder (contains ``README.md``, ``src/``).
BUNDLE_ROOT: Path = Path(__file__).resolve().parents[2]


def _resolve_modelps() -> Path:
    """Prefer monorepo ``../eqcct_sb/ModelPS`` next to this bundle; else ``bundle/ModelPS``."""
    sibling = BUNDLE_ROOT.parent / "eqcct_sb" / "ModelPS"
    if sibling.is_dir():
        return sibling
    return BUNDLE_ROOT / "ModelPS"


MODELPS_DIR: Path = _resolve_modelps()
