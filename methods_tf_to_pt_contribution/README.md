# TensorFlow → PyTorch weight transfer for EQCCT-style pickers (methods bundle)

This folder is a **self-contained copy** of the code used to transfer trained **Keras
(HDF5)** weights for EQCCT **P** and **S** branches into **PyTorch** modules, and to
**validate** parity (layer-wise weights, activations, and SeisBench-scale outputs).

Use it as a **reproducible methods appendix**: readers can drop in their own
``ModelPS/*.h5`` files, install dependencies, and rerun the same checks.

## Layout

| Path | Role |
|------|------|
| `src/eqcct_tf_pt_transfer/models/predictor_pt_p.py` | PyTorch ``EQCCTModelP`` / ``EQCCTModelS`` |
| `src/eqcct_tf_pt_transfer/conversion/loader.py` | Map flat Keras H5 tensors → PyTorch parameters (layout rules) |
| `src/eqcct_tf_pt_transfer/conversion/transfer_weights_legacy.py` | Optional legacy pickle path (if you use ``pickle_path=`` in loader) |
| `src/eqcct_tf_pt_transfer/reference/predictor_tf.py` | TensorFlow ``load_eqcct_model`` (same graph as training) |
| `src/eqcct_tf_pt_transfer/validation/tf_pt_p_trace.py` | P branch: weight + activation parity vs TF |
| `src/eqcct_tf_pt_transfer/validation/tf_pt_s_trace.py` | S branch: same for ``EQCCTModelS`` |
| `src/eqcct_tf_pt_transfer/validation/tf_pt_seisbench_dataset_benchmark.py` | TXED/STEAD windows: MSE/MAE/max-diff aggregates |

Detailed methodology: **`docs/WEIGHT_TRANSFER.md`**, validation & datasets: **`docs/VALIDATION_AND_DATASETS.md`**.

## Setup

```bash
cd methods_tf_to_pt_contribution
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Place Keras checkpoints under ``ModelPS/`` (or pass explicit paths):

- ``test_trainer_024.h5`` — P branch weights (example name from the reference project)
- ``test_trainer_021.h5`` — S branch weights

**Import path:** add ``src`` to ``PYTHONPATH`` (not the repo root):

```bash
export PYTHONPATH="$(pwd)/src"
```

## Quick workflow

1. **Transfer / load weights into PyTorch** (from H5; no TensorFlow required for this step if you only instantiate PT and load):

   ```bash
   PYTHONPATH=src python -c "
   from eqcct_tf_pt_transfer.models.predictor_pt_p import EQCCTModelP
   from eqcct_tf_pt_transfer.conversion.loader import load_eqcct_model_p_weights
   m = EQCCTModelP()
   load_eqcct_model_p_weights(m, h5_path='ModelPS/test_trainer_024.h5')
   print('P weights loaded')
   "
   ```

2. **Strict parity vs TensorFlow** (requires TF + ``predictor_tf``):

   ```bash
   PYTHONPATH=src python -m eqcct_tf_pt_transfer.validation.tf_pt_p_trace \
     --p-h5 ModelPS/test_trainer_024.h5 --s-h5 ModelPS/test_trainer_021.h5
   PYTHONPATH=src python -m eqcct_tf_pt_transfer.validation.tf_pt_s_trace \
     --p-h5 ModelPS/test_trainer_024.h5 --s-h5 ModelPS/test_trainer_021.h5
   ```

3. **Export PyTorch checkpoints** (``.pt``) for inference or downstream code:

   ```bash
   PYTHONPATH=src python -m eqcct_tf_pt_transfer.validation.tf_pt_p_trace \
     --p-h5 ModelPS/test_trainer_024.h5 --s-h5 ModelPS/test_trainer_021.h5 \
     --skip-weights --skip-activations --save-model ModelPS/eqcct_model_p.pt
   PYTHONPATH=src python -m eqcct_tf_pt_transfer.validation.tf_pt_s_trace \
     --p-h5 ModelPS/test_trainer_024.h5 --s-h5 ModelPS/test_trainer_021.h5 \
     --skip-weights --skip-activations --save-model ModelPS/eqcct_model_s.pt
   ```

4. **SeisBench dataset benchmark** (optional; slow on full corpora — use ``--max-windows`` / ``--stride``):

   ```bash
   PYTHONPATH=src python -m eqcct_tf_pt_transfer.validation.tf_pt_seisbench_dataset_benchmark \
     --datasets txed --max-windows 500 --profiles cpu \
     --output-json results/tf_pt_benchmark.json
   ```

The benchmark resolves ``--repo`` to the current working directory by default; with this
bundle layout it prepends ``src/`` automatically so ``import eqcct_tf_pt_transfer``
works without extra flags.

## Relationship to the main project

The canonical implementation lives under the parent repository as ``eqcct_sb/``. This
bundle duplicates the same files under the package name ``eqcct_tf_pt_transfer`` so it
can be archived, cited, or distributed without the rest of the codebase. When updating
the paper pipeline, **refresh this folder from** ``eqcct_sb`` (same filenames under
``src/eqcct_tf_pt_transfer/``) and keep ``README.md`` / ``docs/`` in sync.

## Citation

If you use this bundle in publication, cite your manuscript and optionally this
repository snapshot; add a sentence that parity was verified with the included trace
and (if applicable) SeisBench benchmark scripts.
