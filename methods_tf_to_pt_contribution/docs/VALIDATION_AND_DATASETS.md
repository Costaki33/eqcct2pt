# Validation: traces and SeisBench benchmarks

## Layer and activation traces (`tf_pt_p_trace`, `tf_pt_s_trace`)

These scripts compare **TensorFlow** (Keras) and **PyTorch** for the same branch:

1. **Weights** — Selected Keras layers vs corresponding ``state_dict`` entries (after
   coercion), with configurable ``rtol`` / ``atol``.
2. **Activations** — Intermediate tensors at fixed **layer indices** in the TF model
   vs hand-written **partial forward** functions on the PT model (see
   ``_TF_ACTIVATION_STAGES`` in each file).

**Important:** TF layer indices are **fragile**. If ``predictor_tf.py`` changes
(ordering of layers, extra ops), you must refresh the stage definitions and index tables
documented in each trace module.

**GPU:** By default TF is forced to CPU (``CUDA_VISIBLE_DEVICES=-1``) unless
``--tf-on-gpu`` is set.

**Export only (no TF import):** use ``--skip-weights --skip-activations`` together with
``--save-model`` to write a ``.pt`` file after loading from H5 in pure PyTorch.

Commands (from bundle root, ``PYTHONPATH=src``):

```bash
python -m eqcct_tf_pt_transfer.validation.tf_pt_p_trace \
  --p-h5 ModelPS/test_trainer_024.h5 --s-h5 ModelPS/test_trainer_021.h5
python -m eqcct_tf_pt_transfer.validation.tf_pt_s_trace \
  --p-h5 ModelPS/test_trainer_024.h5 --s-h5 ModelPS/test_trainer_021.h5
```

Default H5 paths assume ``ModelPS/`` under the **bundle root** (parent of ``src/``).

---

## SeisBench dataset benchmark (`tf_pt_seisbench_dataset_benchmark`)

This script scans **TXED** and/or **STEAD** (via ``seisbench``), builds **identical**
6000-sample **ZNE** windows (P-centered; catalog S must fall inside the window), and
compares TF vs PT outputs for **both** P and S models on the same windows.

**Metrics** (aggregated per device profile): mean MSE / MAE between outputs, per-window
max absolute difference, global max, and fractions of windows below tolerance levels.

**Device profiles:** ``cpu``, ``gpu0``, ``gpu1`` (second GPU skipped if not present).
Multiple profiles run in **separate subprocesses** so TensorFlow CPU visibility does not
hide GPUs for later profiles.

**Cost:** Full metadata sweeps can take **many hours** (I/O bound). Cap with
``--max-windows`` and/or subsample with ``--stride``.

Example:

```bash
mkdir -p results
PYTHONPATH=src python -m eqcct_tf_pt_transfer.validation.tf_pt_seisbench_dataset_benchmark \
  --datasets both --max-windows 5000 --stride 1 \
  --profiles cpu,gpu0,gpu1 \
  --output-json results/tf_pt_benchmark.json
```

Outputs:

- Merged JSON at ``--output-json`` when the parent run completes.
- Per-profile sidecar files ``<stem>_cpu.json``, ``<stem>_gpu0.json``, … when merging.

**Dependencies:** ``seisbench`` (datasets may download on first use), ``tqdm`` optional
for progress.

---

## Suggested validation order

1. Run **P** and **S** trace scripts on CPU until clean (or document tolerances).
2. Save **.pt** checkpoints for inference.
3. Optionally run the **SeisBench** benchmark on a **small** ``--max-windows`` first,
   then scale up for publication-grade statistics.
