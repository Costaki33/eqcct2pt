# Weight transfer: Keras HDF5 → PyTorch

## Idea

EQCCT stores trained weights as **Keras** checkpoints (``.h5``): one file for the **P**
picker branch and one for the **S** branch. The PyTorch implementation mirrors the same
architecture (conv front-end, patch encoder, transformer blocks, picker head) but uses
different **parameter layouts** for some layers (e.g. ``nn.Linear`` vs Keras ``Dense``,
``nn.Conv1d`` vs Keras ``Conv1D``, multi-head attention kernels).

The module ``conversion/loader.py``:

1. **Flattens** the HDF5 tree into a single dict keyed by slash-separated names
   (``patch_encoder/dense/kernel:0``, …). Helper: ``flat_torchish_from_h5``.
2. **Coerces** each numpy array to the shape expected by the target PyTorch parameter:
   - Dense: Keras ``(in, out)`` → PyTorch ``(out, in)`` via transpose when needed.
   - MHA: 3D Keras tensors → 2D ``nn.Linear`` weights with documented reshape rules.
   - Conv1d: Keras ``(kw, inC, outC)`` → PyTorch ``(outC, inC, kw)``.
3. **Copies** tensors into ``EQCCTModelP`` / ``EQCCTModelS`` in place.

Entry points:

- ``load_eqcct_model_p_weights(model, h5_path=...)``
- ``load_eqcct_model_s_weights(model, h5_path=...)``

Legacy **pickle** dictionaries are supported for ``EQCCTModelP`` only via
``transfer_weights_legacy.transfer_weights_p`` when ``pickle_path=`` is passed (see
``loader.py``).

## What you need

- Matching **graph**: the PyTorch class definitions must correspond to the Keras model
  built in ``reference/predictor_tf.py`` (``create_cct_modelP`` / ``create_cct_modelS``).
  If you change layer counts or names, update the loader’s regex keys and coercion
  paths.
- **Files**: typically two H5 files (P and S). The validation scripts also load TF’s
  combined graph with both paths so branch-wise weights match training.

## Minimal code example

```python
from eqcct_tf_pt_transfer.models.predictor_pt_p import EQCCTModelP, EQCCTModelS
from eqcct_tf_pt_transfer.conversion.loader import (
    load_eqcct_model_p_weights,
    load_eqcct_model_s_weights,
)

p = EQCCTModelP()
s = EQCCTModelS()
load_eqcct_model_p_weights(p, h5_path="ModelPS/your_p_branch.h5")
load_eqcct_model_s_weights(s, h5_path="ModelPS/your_s_branch.h5")
p.eval()
s.eval()
```

Run with ``PYTHONPATH`` pointing at the ``src`` directory that contains
``eqcct_tf_pt_transfer`` (see main ``README.md``).

## Why a separate “reference” TensorFlow module?

``reference/predictor_tf.py`` defines **the same** Keras model used for parity checks:
``load_eqcct_model(p_h5, s_h5)`` returns ``modelP`` and ``modelS`` with weights loaded.
It is **not** required for deployment if you only trust the PyTorch side after
validation; it **is** required for the trace scripts that compare TF vs PT tensors.
