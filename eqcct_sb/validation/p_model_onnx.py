#!/usr/bin/env python3
"""
Export the TensorFlow ``modelP`` from ``load_eqcct_model`` to ONNX and cross-check
inference with ONNX Runtime.

Install (not part of core EQCCT deps)::

    pip install tf2onnx onnx onnxruntime

Why this helps:

- **TF vs ORT** should match to numerical noise if the export is faithful. A large
  gap means the ONNX graph or runtime, not your PyTorch port, is wrong.
- **PyTorch (H5) vs ORT** is the same discrepancy as PT vs TF when ORT matches TF,
  so the ONNX path does not automatically “fix” a PT implementation gap — it
  isolates whether the reference is TF/ONNX vs the hand-written ``EQCCTModelP``.

**Batch size:** tf2onnx with a fully dynamic batch ``(None, 6000, 3)`` can produce
graphs that pass ``onnx.checker`` but **fail to load** in ONNX Runtime (known
shape-inference issues on some Reshape chains). For reliable ORT checks, pass
``--batch`` to freeze the leading dimension (e.g. the same batch you use in
``parity_p_model``).

**TF → PyTorch via ONNX:** ``onnx2torch`` and similar converters often choke on
full vision/transformer stacks; treat that route as experimental and fall back to
this export + ORT when debugging parity.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tf_output_to_numpy(raw):
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if hasattr(raw, "numpy"):
        return raw.numpy()
    return np.asarray(raw)


def export_keras_p_to_onnx(model_p, onnx_path: Path, *, batch_size: int | None, opset: int) -> None:
    """Write ONNX; ``batch_size`` None → dynamic batch in the tensor spec (ORT may not load)."""
    import tensorflow as tf
    import tf2onnx

    if batch_size is None:
        shape: tuple[int | None, int, int] = (None, 6000, 3)
    else:
        shape = (batch_size, 6000, 3)
    spec = (tf.TensorSpec(shape, tf.float32, name="input"),)
    tf2onnx.convert.from_keras(
        model_p,
        input_signature=spec,
        opset=opset,
        output_path=str(onnx_path),
    )


def run_ort(onnx_path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    y = sess.run(None, {name: x})[0]
    return np.asarray(y)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export EQCCT TF modelP to ONNX; compare TF vs ORT (optional PT).")
    p.add_argument("--p-h5", type=Path, default=None, help="Keras P weights (.h5)")
    p.add_argument("--s-h5", type=Path, default=None, help="Keras S weights (.h5) for load_eqcct_model")
    p.add_argument("--output", type=Path, default=None, help="Output .onnx path")
    p.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Freeze batch dim for export and ORT (default 2, same as parity_p_model). "
        "Use -1 for dynamic batch (export only; ORT compare may fail).",
    )
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--compare-pt",
        action="store_true",
        help="Load EQCCTModelP from --p-h5 and print PyTorch vs ORT stats.",
    )
    p.add_argument(
        "--tf-on-gpu",
        action="store_true",
        help="Allow TensorFlow to see GPUs (default hides them like parity_p_model).",
    )
    args = p.parse_args(argv)

    root = _repo_root()
    p_h5 = args.p_h5 or root / "ModelPS" / "test_trainer_024.h5"
    s_h5 = args.s_h5 or root / "ModelPS" / "test_trainer_021.h5"
    out = args.output or (root / "eqcct_model_p.onnx")

    if not p_h5.is_file():
        print(f"Missing P weights: {p_h5}", file=sys.stderr)
        return 2
    if not s_h5.is_file():
        print(f"Missing S weights (needed for load_eqcct_model): {s_h5}", file=sys.stderr)
        return 2

    try:
        import tensorflow as tf
        import tf2onnx  # noqa: F401
    except ImportError as e:
        print(f"Need tf2onnx and TensorFlow: {e}", file=sys.stderr)
        print("  pip install tf2onnx onnx onnxruntime", file=sys.stderr)
        return 2

    if not args.tf_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    from eqcct_sb.reference.predictor_tf import load_eqcct_model

    model_p, _ = load_eqcct_model(str(p_h5), str(s_h5))

    batch = None if args.batch < 0 else args.batch
    x_batch = batch if batch is not None else 2
    x = np.random.randn(x_batch, 6000, 3).astype(np.float32)

    print(f"[info] Exporting ONNX → {out} (batch spec: {'dynamic' if batch is None else batch})")
    export_keras_p_to_onnx(model_p, out, batch_size=batch, opset=args.opset)

    tf_y = _tf_output_to_numpy(model_p(x, training=False))

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("[warn] onnxruntime not installed; skip ORT compare. pip install onnxruntime", file=sys.stderr)
        return 0

    try:
        ort_y = run_ort(out, x)
    except Exception as e:
        print(
            f"[warn] ONNX Runtime could not load or run the model ({e!r}). "
            "Retry with a fixed --batch (e.g. 2) instead of dynamic export.",
            file=sys.stderr,
        )
        return 1

    d = np.abs(tf_y - ort_y)
    print(
        "TF vs ORT: max |diff|",
        float(d.max()),
        "mean",
        float(d.mean()),
        "| TF mean",
        float(tf_y.mean()),
        "| ORT mean",
        float(ort_y.mean()),
    )

    if args.compare_pt:
        import torch

        from eqcct_sb.conversion.loader import load_eqcct_model_p_weights
        from eqcct_sb.models.predictor_pt_p import EQCCTModelP

        m = EQCCTModelP()
        load_eqcct_model_p_weights(m, h5_path=str(p_h5))
        m.eval()
        with torch.no_grad():
            pt_y = m(torch.from_numpy(x)).numpy()
        d2 = np.abs(pt_y - ort_y)
        print(
            "PT vs ORT: max |diff|",
            float(d2.max()),
            "mean",
            float(d2.mean()),
            "| PT mean",
            float(pt_y.mean()),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
