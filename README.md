# eqcct2pt — TensorFlow to PyTorch weight transfer (EQCCT case study)

Seismic research groups train their own phase-picking models using whichever machine learning framework that fits their process workflow, and TensorFlow/Keras remains a popular choice. At the same time, the modern open-source seismology ecosystem has largely standardized around PyTorch, with SeisBench reinforcing this shift by adopting PyTorch as its exclusive framework. This exclusivity creates a practical challenge for researchers who have existing models using other frameworks: how can these models be transferred to PyTorch without retraining? In practice, transferring model weights can be difficult, and without a reliable conversion process, research groups often   
struggle to integrate their prior work into modern toolkits. In this repository, we document a practical path to **reuse the same weights** in PyTorch: layout rules, loaders, and checks that hold when you point at real EQCCT checkpoints. It is built around **EQCCT’s split design**, a **P-branch** model and a separate **S-branch** model, each with its own checkpoint.

## How to read this repository

**Model checkpoints and exports** live under `ModelPS/` at the repo root (paired `.h5` Keras bundles and optional `.pt` exports). `paths.py` pins `MODELPS_DIR` and `REPO_ROOT` so scripts and notebooks do not depend on fragile relative paths. The original TensorFlow P model is `test_trainer_024.h5`, and the original S model is `test_trainer_021.h5`.

**Dual implementations:** `models/` defines the PyTorch modules you would ship (`EQCCTModelP`, `EQCCTModelS` and related code). `reference/` holds the TensorFlow/Keras side used to load the original weights and to run numerical comparisons. You will read both when asking whether a layer order still matches after a Keras upgrade.

**Conversion** (`conversion/`) is the main weight transfer system. It includes flattening HDF5 groups, normalizing names, and mapping each weight to the right PyTorch parameter, including the easy-to-get-wrong cases (Conv1d axis order, dense transposes, attention projections).

**Validation** (`validation/`) provides quick parity checks against synthetic tensors, structured traces per branch, SeisBench-window error statistics, timing and memory benchmarks, and optional ONNX export for an extra reference path. Heavy or multi-GPU jobs can be split into documented CLI modules; read the module docstring for the exact flags.

**Figures for writeups** are produced from saved JSON/NPZ by `scripts/` (matplotlib only), so you can refresh plots without re-running TensorFlow if the data files already exist.

**Optional:** `training/` contains SeisBench-style finetune glue; `misc/` small experiments; `notebooks/` exploratory work. `methods_tf_to_pt_contribution/` is a flattened copy for supplementary material (imports renamed to `eqcct_tf_pt_transfer`).

## Environment

Create a Conda environment from the pinned stack (CPU PyTorch by default; see comments in the file for GPU):

```bash
conda env create -f environment.yml
conda activate eqcct2pt
```

Optional ONNX path (P-model export and ORT check only): `pip install tf2onnx onnx onnxruntime` as described in `validation/p_model_onnx.py`.

TensorFlow and PyTorch together are sensitive to CUDA/driver pairings; if the solve fails on your platform, create a minimal env with your lab’s standard TF+Torch stack, then `pip install seisbench silence-tensorflow` and the conda packages you still need (`h5py`, `matplotlib`, etc.).

## Quick start

From the repository root:

```bash
export PYTHONPATH=.

# Fast TensorFlow vs PyTorch check on synthetic input (both frameworks required)
python -m validation.parity_p_model

# Structured weight + activation diff — P branch, then S branch
python -m validation.tf_pt_p_trace
python -m validation.tf_pt_s_trace
```

Larger studies (SeisBench slices, layer activations, per-window errors, performance JSON) live alongside these modules. Run `python -m validation.<module> --help` for each entry point.

## Directory map


| Path          | Role                                                                    |
| ------------- | ----------------------------------------------------------------------- |
| `ModelPS/`    | Bundled Keras `.h5` checkpoints, exported `.pt` weights, legacy pickles |
| `paths.py`    | Canonical `MODELPS_DIR`, `REPO_ROOT`                                    |
| `models/`     | PyTorch EQCCT definitions intended for reuse                            |
| `reference/`  | TensorFlow/Keras mirror for loading and comparison                      |
| `conversion/` | HDF5 to `state_dict` loaders and tensor layout helpers                   |
| `validation/` | Parity, benchmarks, exports, dataset-driven checks                      |
| `scripts/`    | Manuscript-style plots from `results/*.json` and `results/*.npz`        |
| `training/`   | Finetuning / dataset utilities                                          |
| `misc/`       | Small standalone experiments                                            |
| `notebooks/`  | Exploration and teaching                                                |


