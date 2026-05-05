# A TensorFlow to PyTorch weight transfer toolkit

Seismic research groups train their own phase-picking models using whichever machine learning framework that fits their process workflow, and TensorFlow/Keras remains a popular choice. At the same time, the modern open-source seismology ecosystem has largely standardized around PyTorch, with SeisBench reinforcing this shift by adopting PyTorch as its exclusive framework. This exclusivity creates a practical challenge for researchers who have existing models using other frameworks: how can these models be transferred to PyTorch without retraining? In practice, transferring model weights can be difficult, and without a reliable conversion process, research groups often   
struggle to integrate their prior work into modern toolkits. In this repository, we document a practical path to **reuse the same weights** in PyTorch: layout rules, loaders, and checks that hold when you point at real EQCCT checkpoints. It is built around **EQCCT’s split design**, a **P-branch** model and a separate **S-branch** model, each with its own checkpoint. This work is part of a larger research paper currently in the publication process. The preprint release can be read [here](Porting%20Institutional%20Phase%20Pickers%20to%20PyTorch%20from%20TensorFlow.pdf).

## Environment

Create a Conda environment from the pinned stack (CPU PyTorch by default; see comments in the file for GPU):

```bash
conda env create -f environment.yml
conda activate eqcct2pt
```

TensorFlow **2.15.1** with **Keras 2.15.0** matches the version stack used in our `eqcctpro` baseline and avoids Keras 3 + TensorFlow 2.2x changing Keras model load semantics for the S checkpoint (which showed up as bad TF S picks in some environments). Prefer that pair for parity work:

```bash
python -m pip install 'numpy>=1.26,<2' 'tensorflow==2.15.1' 'keras==2.15.0'
```

If your env only needs `import tensorflow`, you can omit the separate `keras` pin; for an ad-hoc env, use **`tensorflow==2.15.1`** (not `tensorflow>=2.21`) unless you intentionally re-verify weights on a newer stack.

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


