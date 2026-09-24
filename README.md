# A TensorFlow to PyTorch weight transfer toolkit

Seismic research groups train their own phase-picking models using whichever machine learning framework that fits their process workflow, and TensorFlow/Keras remains a popular choice. At the same time, the modern open-source seismology ecosystem has largely standardized around PyTorch, with SeisBench reinforcing this shift by adopting PyTorch as its exclusive framework. This exclusivity creates a practical challenge for researchers who have existing models using other frameworks: how can these models be transferred to PyTorch without retraining? In practice, transferring model weights can be difficult, and without a reliable conversion process, research groups often   
struggle to integrate their prior work into modern toolkits. In this repository, we document a practical path to **reuse the same weights** in PyTorch: layout rules, loaders, and checks that hold when you point at real EQCCT checkpoints. It is built around **EQCCT’s split design**, a **P-branch** model and a separate **S-branch** model, each with its own checkpoint. This work is part of a larger research paper currently in the publication process. The preprint release can be read [here](Porting%20Institutional%20Phase%20Pickers%20to%20PyTorch%20from%20TensorFlow.pdf).

## Environment

**Paper tables (ESS 2026EA005507)** used TensorFlow 2.19.0, Keras 3.10.0, and PyTorch 2.7.1 (`environment.paper.yml`):

```bash
conda env create -f environment.paper.yml
conda activate eqcct2pt-paper
```

The default `environment.yml` still pins TensorFlow **2.15.1** / Keras **2.15.0** for `eqcctpro`-compatible loading tests. That older stack is **not** the stack used for Table 1.

```bash
conda env create -f environment.yml
conda activate eqcct2pt
```

### Reproduce Table 1 and window IDs

From the repository root, on the tagged revision `ess-2026ea005507-r1`:

```bash
export PYTHONPATH=.
export EQCCT_REQUIRE_STRICT_LOAD=1

python scripts/export_window_ids.py --output results/window_ids.csv

python -m validation.tf_pt_seisbench_dataset_benchmark \
  --datasets both --max-windows 50000 --stride 1 --profiles cpu \
  --output-json results/tf_pt_benchmark_cpu.json

# GPU Table 1 from one TF32-off run that stores per-window MAE, MSE, and D_w:
python -m validation.tf_pt_per_window_errors \
  --datasets txed,stead --max-windows 50000 --stride 1 --profiles gpu0 \
  --output-npz results/per_window_errors_gpu100k_tf32off_mse.npz \
  --output-summary-json results/per_window_errors_gpu100k_tf32off_mse_summary.json

python scripts/recompute_table1.py \
  --cpu-json results/tf_pt_benchmark_cpu.json \
  --gpu-npz results/per_window_errors_gpu100k_tf32off_mse.npz \
  --out results/table1_same_run.json

python -m validation.tf_pt_pick_equivalence \
  --datasets both --max-windows 100000 --profiles cpu \
  --thresholds 0.1,0.3,0.5 \
  --output-json results/pick_equivalence_cpu_thresholds.json

python -m validation.find_cpu_argmax_mismatch
python scripts/plot_cpu_argmax_mismatch.py
```

GPU runs disable TF32 unless `EQCCT_ALLOW_TF32=1`. Paper runs set `EQCCT_REQUIRE_STRICT_LOAD=1`, which forbids `skip_mismatch`. The P branch loads by name; under Keras 3 the S branch loads positionally and picker kernels are checked against HDF5 (`results/load_weights_strategy.json`). Skipped variables fail the run. Checkpoints: `ModelPS/test_trainer_024.h5` (P) and `ModelPS/test_trainer_021.h5` (S); hashes in `results/checkpoint_manifest.json`. The intended revision tag is `ess-2026ea005507-r1` (apply after the revision commit).

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
| `models/`     | Canonical PyTorch EQCCT implementation (`eqcct`)                |
| `reference/`  | TensorFlow/Keras mirror for loading and comparison                      |
| `conversion/` | HDF5 to `state_dict` loaders (`loader.py`, `catalog.py`, pickle path via `transfer_weights_legacy.py`) |
| `validation/` | Parity, benchmarks, exports, dataset-driven checks                    |
| `scripts/`    | Manuscript-style plots from `results/*.json` and `results/*.npz`        |


