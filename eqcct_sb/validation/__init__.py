"""
Validation entry points for EQCCT PyTorch ↔ TensorFlow parity and SeisBench smoke tests.

- ``parity_p_model``: random-input PT vs Keras (if TensorFlow imports) + H5 structure check.
- ``p_model_onnx``: export TF ``modelP`` to ONNX; TF vs ONNX Runtime (optional PT vs ORT).
- ``tf_pt_p_trace``: per-parameter TF vs PT check and per-stage activation diff for **P** (finds first skewed layer).
- ``tf_pt_s_trace``: same workflow for **S** (``modelS`` vs ``EQCCTModelS``).
- ``seisbench_p_model``: STEAD/TXED batches through ``EQCCTModelP``.
- ``run_checks``: runs parity then SeisBench (STEAD) subprocess steps.
- ``tf_gpu_env_sniff``: print ``sys.executable`` and whether TensorFlow lists GPUs (use to match Jupyter kernel to a working conda env).
"""
