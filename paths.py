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

# STEAD / TXED are **not** stored under this repository.
#
# Validation entry points (e.g. ``validation/tf_pt_waveform_compare_figure.py``,
# ``validation/tf_pt_seisbench_dataset_benchmark.py``) load community data via SeisBench::
#
#     import seisbench.data as sbd
#     ds = sbd.STEAD(sampling_rate=100, component_order="ZNE")
#
# That reads the **preprocessed** STEAD copy under SeisBench’s data cache, by default::
#
#     ~/.seisbench/datasets/stead/
#
# (``seisbench.cache_data_root`` is ``<cache_root>/datasets``, so STEAD is
# ``seisbench.cache_data_root / "stead"`` — not a second ``datasets`` level.)
#
# First-time setup from **raw** STEAD (``merged.csv`` + ``merged.hdf5``) uses::
#
#     sbd.STEAD(compile_from_source=True, download_kwargs={"basepath": Path("/path/to/unpacked")})
