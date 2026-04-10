#!/usr/bin/env python3
"""
Numerical parity: TensorFlow Keras P picker vs PyTorch EQCCTModelP.

Run from repo root:
  PYTHONPATH=/path/to/EQCCT_to_Seisbench python -m eqcct_sb.validation.parity_p_model

Requires TensorFlow only for the Keras branch; PyTorch + h5py always runs.

By default TensorFlow runs on **CPU** for this script (`CUDA_VISIBLE_DEVICES='-1'`)
so parity is not corrupted by XLA / old ptxas on Ada-class GPUs. Pass ``--tf-on-gpu``
to use the GPU (compare with CPU if results look wrong).

To export the same Keras ``modelP`` to ONNX and compare **TensorFlow vs ONNX Runtime**
(matched to ~1e-7 here), run ``python -m eqcct_sb.validation.p_model_onnx`` (see that
module for ``pip install tf2onnx onnx onnxruntime``). That isolates PT-vs-reference:
if ORT matches TF but PT does not, the gap is in ``EQCCTModelP``, not in ONNX.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from eqcct_sb.conversion.loader import (
    flat_torchish_from_h5,
    load_eqcct_model_p_weights,
)
from eqcct_sb.models.predictor_pt_p import EQCCTModelP


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tf_output_to_numpy(raw):
    """Keras may return a Tensor, a list/tuple of tensors, or an ndarray."""
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if hasattr(raw, "numpy"):
        return raw.numpy()
    return np.asarray(raw)


def _verify_tf_p_weights_match_disk(model_p_tf, p_h5: Path) -> tuple[bool, str]:
    """
    Compare a few critical kernels to the .h5 file. If these diverge, load_weights
    did not bind variables to the checkpoint (or the wrong file was used).
    """
    try:
        import tensorflow as tf
    except ImportError:
        return True, "[skip] TensorFlow not available"

    from eqcct_sb.conversion.loader import flat_torchish_from_h5

    disk = flat_torchish_from_h5(str(p_h5))
    conv_key = "conv1d/kernel:0"
    pick_key = "picker_P/kernel:0"
    if conv_key not in disk or pick_key not in disk:
        return False, f"missing {conv_key} or {pick_key} in flat H5"

    k_conv = disk[conv_key]
    k_pick = disk[pick_key]

    pick_layer = next(
        (
            lay
            for lay in model_p_tf.layers
            if isinstance(lay, tf.keras.layers.Conv1D) and lay.name == "picker_P"
        ),
        None,
    )
    if pick_layer is None:
        return False, "TF model has no Conv1D layer named picker_P"

    conv_layer = next(
        (
            lay
            for lay in model_p_tf.layers
            if isinstance(lay, tf.keras.layers.Conv1D) and lay.name == "conv1d"
        ),
        None,
    )

    if conv_layer is None:
        return False, "Could not locate TF layer named 'conv1d' (expected first backbone Conv1D)"

    w_conv, _ = conv_layer.get_weights()
    w_pick, _ = pick_layer.get_weights()
    if w_pick.shape != k_pick.shape:
        return (
            False,
            f"picker_P shape {w_pick.shape} != disk {k_pick.shape}",
        )
    if not np.allclose(w_conv, k_conv, rtol=1e-6, atol=1e-8):
        return (
            False,
            f"backbone Conv1D '{conv_layer.name}' kernel differs from H5 "
            "(wrong graph or load_weights did not apply)",
        )
    if not np.allclose(w_pick, k_pick, rtol=1e-6, atol=1e-8):
        return False, "picker_P kernel differs from H5 (checkpoint not loaded?)"

    mha_key = "multi_head_attention/query/kernel:0"
    if mha_key in disk:
        mha_layer = next(
            (
                lay
                for lay in model_p_tf.layers
                if isinstance(lay, tf.keras.layers.MultiHeadAttention)
            ),
            None,
        )
        if mha_layer is None:
            return False, "H5 has MHA weights but TF model has no MultiHeadAttention layer"
        wq = mha_layer.get_weights()[0]
        k_q = disk[mha_key]
        if wq.shape != k_q.shape:
            return (
                False,
                f"first MHA query kernel shape {wq.shape} != disk {k_q.shape}",
            )
        if not np.allclose(wq, k_q, rtol=1e-6, atol=1e-8):
            return (
                False,
                "first multi_head_attention query kernel differs from H5 "
                "(strict load may have bound the wrong tensors for MHA)",
            )
        return True, "TF Conv1D / picker_P / first MHA query kernels match H5 file"
    return True, "TF Conv1D / picker_P kernels match H5 file (no MHA query key in flat H5)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="EQCCT P-model TF vs PT parity")
    parser.add_argument(
        "--p-h5",
        type=Path,
        default=None,
        help="Keras weights for P model (e.g. test_trainer_024.h5)",
    )
    parser.add_argument(
        "--s-h5",
        type=Path,
        default=None,
        help="Keras weights for S model (needed only to call load_eqcct_model)",
    )
    parser.add_argument(
        "--pickle",
        type=Path,
        default=None,
        help="Optional legacy Pweights.pkl; compared against H5-loaded PT",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tf-on-gpu",
        action="store_true",
        help="Allow TensorFlow to use GPU for predict(). Default is CPU only (avoids "
        "XLA/ptxas miscompile warnings and wrong near-zero outputs on some setups).",
    )
    parser.add_argument(
        "--debug-pre-picker",
        action="store_true",
        help="Print mean/std/min/max of the tensor fed into picker_P (before final Conv1D).",
    )
    parser.add_argument(
        "--verbose-tf-error",
        action="store_true",
        help="Print full Python traceback when the TensorFlow branch fails.",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    p_h5 = args.p_h5 or root / "ModelPS" / "test_trainer_024.h5"
    s_h5 = args.s_h5 or root / "ModelPS" / "test_trainer_021.h5"

    if not p_h5.is_file():
        print(f"Missing P weights: {p_h5}", file=sys.stderr)
        return 2

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    x = np.random.randn(2, 6000, 3).astype(np.float32)

    # --- PyTorch (H5) ---
    m_h5 = EQCCTModelP()
    load_eqcct_model_p_weights(m_h5, h5_path=str(p_h5))
    m_h5.eval()
    with torch.no_grad():
        pt_h5 = m_h5(torch.from_numpy(x)).numpy()

    m_h5_b = EQCCTModelP()
    load_eqcct_model_p_weights(m_h5_b, h5_path=str(p_h5))
    sd1, sd2 = m_h5.state_dict(), m_h5_b.state_dict()
    if sd1.keys() != sd2.keys():
        print("[warn] State dict keys differ after second H5 load.", file=sys.stderr)
    elif any(not torch.equal(sd1[k], sd2[k]) for k in sd1):
        print("[warn] Repeated H5 load produced differing weights.", file=sys.stderr)
    else:
        print("[ok] Two H5 loads produce identical state dicts.")

    # --- PyTorch (pickle), optional ---
    pt_pkl = None
    if args.pickle and args.pickle.is_file():
        m_pkl = EQCCTModelP()
        load_eqcct_model_p_weights(m_pkl, pickle_path=str(args.pickle))
        m_pkl.eval()
        with torch.no_grad():
            pt_pkl = m_pkl(torch.from_numpy(x)).numpy()
        try:
            np.testing.assert_allclose(pt_h5, pt_pkl, rtol=args.rtol, atol=args.atol)
            print("[ok] H5-loaded and pickle-loaded PyTorch outputs match.")
        except AssertionError as e:
            d = np.abs(pt_h5 - pt_pkl)
            print(
                "[warn] H5 vs pickle PT mismatch — max",
                d.max(),
                "mean",
                d.mean(),
                file=sys.stderr,
            )
            print(e, file=sys.stderr)

    # --- TensorFlow ---
    tf_out = None
    try:
        import os

        if not args.tf_on_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        import tensorflow as tf

        try:
            tf.config.optimizer.set_jit(False)
        except Exception:
            pass

        from eqcct_sb.reference.predictor_tf import load_eqcct_model

        tf.random.set_seed(args.seed)
        if not args.tf_on_gpu:
            print("[info] TensorFlow parity using CPU (use --tf-on-gpu for GPU).")

        if not s_h5.is_file():
            print(f"[skip] S weights missing for Keras load: {s_h5}", file=sys.stderr)
        else:
            model_p_tf, _ = load_eqcct_model(str(p_h5), str(s_h5))
            ok_load, load_msg = _verify_tf_p_weights_match_disk(model_p_tf, p_h5)
            print(f"[info] {load_msg}")
            if not ok_load:
                print(
                    "[warn] TensorFlow variables do not match the P checkpoint — "
                    "fix weight loading before expecting PT/TF parity.",
                    file=sys.stderr,
                )

            if args.debug_pre_picker:
                pick = model_p_tf.get_layer("picker_P")
                pre = pick.input
                if isinstance(pre, (list, tuple)):
                    pre = pre[0]
                sub = tf.keras.Model(model_p_tf.input, pre)
                z = sub(x, training=False)
                z = _tf_output_to_numpy(z)
                print(
                    "[debug] TF pre-picker",
                    float(np.mean(z)),
                    float(np.std(z)),
                    float(np.min(z)),
                    float(np.max(z)),
                    "shape",
                    z.shape,
                )
                with torch.no_grad():
                    xi = torch.from_numpy(x)
                    xi = xi.transpose(1, 2)
                    xi = m_h5.conv1(xi)
                    xi = m_h5.conv2(xi)
                    xi = m_h5.conv3(xi)
                    xi = xi.unsqueeze(2).permute(0, 3, 2, 1)
                    xi = m_h5.patch(xi)
                    xi = m_h5.encoder(xi)
                    xi = m_h5.transformer(xi)
                    xi = m_h5.norm(xi)
                    xi = xi.reshape(xi.size(0), 6000, 1)
                print(
                    "[debug] PT pre-picker",
                    float(xi.mean()),
                    float(xi.std()),
                    float(xi.min()),
                    float(xi.max()),
                    "shape",
                    tuple(xi.shape),
                )

            # Explicit training=False avoids any path where dropout/MHA dropout stays on.
            raw = model_p_tf(x, training=False)
            tf_out = _tf_output_to_numpy(raw)
    except Exception as e:
        print(f"[skip] TensorFlow parity not available: {e}", file=sys.stderr)
        if args.verbose_tf_error:
            traceback.print_exc()

    print(
        "PyTorch (H5) output: mean",
        pt_h5.mean(),
        "std",
        pt_h5.std(),
        "min",
        pt_h5.min(),
        "max",
        pt_h5.max(),
    )

    if tf_out is not None:
        print(
            "TensorFlow P output: mean",
            tf_out.mean(),
            "std",
            tf_out.std(),
            "min",
            tf_out.min(),
            "max",
            tf_out.max(),
        )
        if pt_h5.shape != tf_out.shape:
            print(
                f"[warn] Shape mismatch PT {pt_h5.shape} vs TF {tf_out.shape}",
                file=sys.stderr,
            )
            return 1
        diff = np.abs(pt_h5 - tf_out)
        print("|PT - TF| max", diff.max(), "mean", diff.mean())
        rel = diff.max() / (np.abs(pt_h5).max() + 1e-12)
        if rel > 10.0 or diff.max() > 0.01:
            print(
                "[note] Differences are huge (not a rtol/atol tuning issue). "
                "Typical causes: TF load_weights did not match the checkpoint graph, "
                "patch/tensor layout mismatch, or BN/attention implementation drift.",
                file=sys.stderr,
            )
        try:
            np.testing.assert_allclose(pt_h5, tf_out, rtol=args.rtol, atol=args.atol)
            print(f"[ok] PyTorch matches TensorFlow (rtol={args.rtol}, atol={args.atol}).")
        except AssertionError:
            print(
                "[warn] PT vs TF differs beyond tolerance — only raise rtol/atol if "
                "diagnostics show matching weights and you expect float noise (~1e-4–1e-6).",
                file=sys.stderr,
            )
            return 1

    # Structural check: flat dict matches catalog expectations
    torchish = flat_torchish_from_h5(str(p_h5))
    from eqcct_sb.conversion.catalog import assert_and_print

    try:
        assert_and_print(torchish)
    except AssertionError as e:
        print(f"[warn] catalog sanity: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
