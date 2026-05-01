# eqcct2pt

Bring **EQCCT** seismic phase pickers from their original **TensorFlow / Keras** training stack into **PyTorch**, with enough checks that you can honestly say “this module is still the same model,” not “we rewrote something similar.”

## Why this exists

Lots of institutional pickers live in TensorFlow because that was the path of least resistance when they were built. Meanwhile the stack you probably want next (SeisBench tooling, newer lab code) is largely PyTorch. Your checkpoints are still Keras HDF5 blobs from the legacy pipeline. This repo exists to narrow that bridge: carry weights across faithfully, preserve forward‑pass semantics, emit `.pt` checkpoints that downstream code can load without surprises.

## How to read the repo (mental walk‑through)

1. **Understand what you are cloning.** EQCCT is not one network; it ships **two** models: a **P branch** and an **S branch**, each with its own HDF5 checkpoint. Those files end up next to exported PyTorch weights under **`ModelPS/`** at this repository root.
2. **See the architectures in code.** **`models/`** (`predictor_pt_p.py`) is where the definitive PyTorch `EQCCTModelP` and `EQCCTModelS` live. The **`reference/`** tree holds the TensorFlow mirror used when you compare numerically or load Keras checkpoints.
3. **Move tensors, not guesses.** HDF5 lays weights out differently from PyTorch `state_dict`s (transpose rules, convolution axis order, attention splits). **`conversion/`** is the seriousness: loaders that map keys, coerce shapes, and refuse “close enough” guesses that would silently nail the wrong layer.
4. **Prove it.** **`validation/`** is the story you tell reviewers or your future self after a long break: parity on random tensors, traces through individual layers when something drifts, then larger SeisBench window pulls when you dare trust aggregate error. ONNX export lives here too when you want a second opinion beside TensorFlow.
5. **`paths.py`** is unglamorous but important: one place that knows **`ModelPS/`** sits at repo root so scripts do not reinvent brittle `../../` navigation.

Supporting pieces: **`training/`** for SeisBench‑style finetune experiments people had open; **`misc/`** for tiny one‑off timings; notebooks under **`notebooks/`** mostly for exploratory figures and teaching.

There is also **`methods_tf_to_pt_contribution/`**, a flattened copy packaged for supplementary material (imports renamed to `eqcct_tf_pt_transfer`). Sync it from here when those files drift.

## Quick start

```bash
# From this repository root
export PYTHONPATH=.

# Fast TensorFlow ⇄ PyTorch check on synthetic input (needs TensorFlow installed)
python -m validation.parity_p_model

# Structured weight + activation diff for the P branch
python -m validation.tf_pt_p_trace

# Same for the S branch
python -m validation.tf_pt_s_trace
```

SeisBench heavy jobs (large window counts, multiple GPU profiles for timing) expose subcommands alongside these; skim the docstrings at the tops of **`validation/tf_pt_seisbench_dataset_benchmark.py`**, **`tf_pt_perf_benchmark.py`**, **`tf_pt_layer_activations.py`**, etc., for invocation examples.

Dependencies are not minimal (TensorFlow, PyTorch, h5py, SeisBench for some paths). Spin up environments the way your lab already prefers; the code assumes you intentionally installed both frameworks when parity matters.

## Layout

| Path | Role |
|------|------|
| `ModelPS/` | Example Keras `.h5` bundles, exported `.pt` checkpoints, pickles when you inherit legacy flows |
| `paths.py` | `MODELPS_DIR`, `REPO_ROOT` anchors |
| `models/` | PyTorch architectures you actually ship |
| `reference/` | TensorFlow counterpart for loaders and benchmarks |
| `conversion/` | HDF5 flattening, key normalization, coercion into tensors |
| `validation/` | Parity tooling, ONNX helper, plotting scripts orchestration helpers |
| `training/` | Finetuning / dataset glue |
| `scripts/` | Post‑processing JSON / NPZ benchmarks into manuscript figures |

## Naming on GitHub

This tree is **`eqcct2pt`‑shaped**: the folders you care about (`conversion`, `validation`, …) surface **directly under the repo root** so cloning on GitHub does not hide everything behind yet another nesting level.

## License / citation

Honor whatever license you attach downstream. Academic users should cite the EQCCT paper and any SeisBench corpora invoked in experiments; add a sentence if you rerun the parity scripts bundled here.
