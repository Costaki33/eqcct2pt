#!/usr/bin/env python3
"""
Per-parameter and per-stage activation trace: TensorFlow ``modelP`` vs PyTorch ``EQCCTModelP``.

Run from repo root::

    PYTHONPATH=. python -m validation.tf_pt_p_trace \\
      --p-h5 ModelPS/test_trainer_024.h5 --s-h5 ModelPS/test_trainer_021.h5

Several RNG seeds (weights checked once; activations per seed)::

    PYTHONPATH=. python -m validation.tf_pt_p_trace \\
      --p-h5 ... --s-h5 ... --seeds 0,1,2,3,4

Save a PyTorch ``.pt`` checkpoint (state dict + meta with P/S H5 paths), e.g. for notebooks
or deployment (runs after PT weights are loaded from ``--p-h5``)::

    PYTHONPATH=. python -m validation.tf_pt_p_trace \\
      --p-h5 ... --s-h5 ... --skip-weights --skip-activations --save-model weights/eqcct_model_p.pt

TensorFlow layer indices are fixed to the graph produced by ``load_eqcct_model`` /
``create_cct_modelP`` (see ``predictor_tf.py``). If that graph changes, update
``_TF_ACTIVATION_STAGES`` and the ``compare_weights`` layer-index tables.

Use ``--block0-substeps`` to localize mismatches inside the first transformer block
(norm1 → MHA → MLP).

Requires TensorFlow for the Keras branch.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from conversion.loader import (
    _coerce_conv1d,
    _coerce_linear,
    _coerce_mha_bias,
    _coerce_mha_out_weight,
    _coerce_mha_qkv_weight,
    load_eqcct_model_p_weights,
)
from models.predictor_pt_p import EQCCTModelP
from paths import MODELPS_DIR


def _np(p: torch.Tensor) -> np.ndarray:
    return p.detach().cpu().numpy()


@dataclass(frozen=True)
class ActStage:
    key: str
    tf_layer_index: int
    pt_fn: Callable[[EQCCTModelP, torch.Tensor], torch.Tensor]
    """PT forward fragment; takes (model, x_tensor [B,6000,3]) returns activation."""
    layout: str
    """'nlc' = (B,T,C) TF-style; 'ncl_pt' = PT conv tensor (B,C,T), TF tensor is (B,T,C)."""

    def tf_to_common(self, y: np.ndarray) -> np.ndarray:
        # Always normalize to (B, time, channels) for comparison.
        if self.layout == "nlc":
            return y
        if self.layout == "ncl_pt":
            # TensorFlow conv front-end is still (B, L, C); leave as-is.
            return y
        raise ValueError(self.layout)

    def pt_to_common(self, y: torch.Tensor) -> np.ndarray:
        t = _np(y)
        if self.layout == "nlc":
            return t
        if self.layout == "ncl_pt":
            # PT Conv1d: (B, C, L) -> (B, L, C)
            return np.transpose(t, (0, 2, 1))
        raise ValueError(self.layout)


# End of front-end conv stack: last dropout after third ConvF1 tower.
# Patch encoder output: (B, 150, 40).
# Transformer block ends at add_4, add_6, add_8, add_10 (0-based block indices).
# Final LN on sequence, then reshape to (B, 6000, 1) before picker_P.
_TF_ACTIVATION_STAGES: tuple[ActStage, ...] = (
    ActStage(
        "post_front (after 3x ConvF1, channels-last)",
        33,
        lambda m, xt: _pt_feed_to(m, xt, through="conv3"),
        "ncl_pt",
    ),
    ActStage(
        "post_patch_encoder (B,150,40)",
        36,
        lambda m, xt: _pt_feed_to(m, xt, through="encoder"),
        "nlc",
    ),
    ActStage(
        "post_transformer_block0",
        47,
        lambda m, xt: _pt_feed_to(m, xt, through="tb0"),
        "nlc",
    ),
    ActStage(
        "post_transformer_block1",
        58,
        lambda m, xt: _pt_feed_to(m, xt, through="tb1"),
        "nlc",
    ),
    ActStage(
        "post_transformer_block2",
        69,
        lambda m, xt: _pt_feed_to(m, xt, through="tb2"),
        "nlc",
    ),
    ActStage(
        "post_transformer_block3",
        80,
        lambda m, xt: _pt_feed_to(m, xt, through="tb3"),
        "nlc",
    ),
    ActStage(
        "pre_picker_after_final_LN (B,150,40)",
        81,
        lambda m, xt: _pt_feed_to(m, xt, through="norm"),
        "nlc",
    ),
    ActStage(
        "pre_picker_after_reshape (B,6000,1)",
        82,
        lambda m, xt: _pt_feed_to(m, xt, through="reshape"),
        "nlc",
    ),
)


def _pt_feed_to(m: EQCCTModelP, xt: torch.Tensor, *, through: str) -> torch.Tensor:
    """``xt`` is (B, 6000, 3) float tensor."""
    x = xt.transpose(1, 2)
    x = m.conv1(x)
    x = m.conv2(x)
    x = m.conv3(x)
    if through == "conv3":
        return x
    x = x.unsqueeze(2).permute(0, 3, 2, 1)
    x = m.patch(x)
    x = m.encoder(x)
    if through == "encoder":
        return x
    for i, blk in enumerate(m.transformer):
        x = blk(x)
        stop = f"tb{i}"
        if through == stop:
            return x
    x = m.norm(x)
    if through == "norm":
        return x
    x = x.reshape(x.size(0), 6000, 1)
    if through == "reshape":
        return x
    raise ValueError(through)


def _tf_subforward(model_p_tf, layer_index: int, x_np: np.ndarray):
    import tensorflow as tf

    lay = model_p_tf.layers[layer_index]
    sub = tf.keras.Model(model_p_tf.input, lay.output)
    y = sub(x_np, training=False)
    if hasattr(y, "numpy"):
        y = y.numpy()
    return np.asarray(y)


def compare_weights(
    model_p_tf,
    m_pt: EQCCTModelP,
    *,
    rtol: float,
    atol: float,
) -> list[str]:
    """Return list of error messages (empty if all match)."""
    import tensorflow as tf

    sd = m_pt.state_dict()
    errors: list[str] = []

    def ok(a: np.ndarray, b: torch.Tensor, label: str) -> bool:
        an = np.asarray(a)
        bn = _np(b)
        if an.shape != bn.shape:
            errors.append(f"{label}: shape TF/coerced {an.shape} vs PT {bn.shape}")
            return False
        if not np.allclose(an, bn, rtol=rtol, atol=atol):
            d = np.abs(an - bn)
            errors.append(
                f"{label}: max|diff|={d.max():.6g} mean={d.mean():.6g} rtol={rtol} atol={atol}"
            )
            return False
        return True

    # --- Convs + BN (9 blocks × conv + 4 bn tensors); TF layer indices from graph dump ---
    conv_idx = [1, 4, 8, 12, 15, 19, 23, 26, 30]
    bn_idx = [2, 5, 9, 13, 16, 20, 24, 27, 31]
    conv_pt = [
        ("conv1.conv1", "conv1.conv2", "conv1.conv3"),
        ("conv2.conv1", "conv2.conv2", "conv2.conv3"),
        ("conv3.conv1", "conv3.conv2", "conv3.conv3"),
    ]
    flat_conv = [c for t in conv_pt for c in t]

    for i, (c_tf, prefix) in enumerate(zip(conv_idx, flat_conv)):
        lay = model_p_tf.layers[c_tf]
        assert isinstance(lay, tf.keras.layers.Conv1D), (i, lay)
        w_tf, b_tf = lay.get_weights()
        pw = sd[f"{prefix}.weight"]
        pb = sd[f"{prefix}.bias"]
        w_c = _coerce_conv1d(w_tf, pw)
        ok(w_c, pw, f"Conv1D {prefix}.weight (TF layer {c_tf} {lay.name})")
        ok(b_tf, pb, f"Conv1D {prefix}.bias")

        b_tf_l = model_p_tf.layers[bn_idx[i]]
        assert isinstance(b_tf_l, tf.keras.layers.BatchNormalization), b_tf_l
        gamma, beta, mm, mv = b_tf_l.get_weights()
        prefix_bn = prefix.replace(".conv1", ".bn1").replace(".conv2", ".bn2").replace(".conv3", ".bn3")
        ok(gamma, sd[f"{prefix_bn}.weight"], f"BN {prefix_bn}.weight (gamma)")
        ok(beta, sd[f"{prefix_bn}.bias"], f"BN {prefix_bn}.bias")
        ok(mm, sd[f"{prefix_bn}.running_mean"], f"BN {prefix_bn}.running_mean")
        ok(mv, sd[f"{prefix_bn}.running_var"], f"BN {prefix_bn}.running_var")

    # --- PatchEncoder (TF composite layer index 36) ---
    pe = model_p_tf.layers[36]
    w_proj, b_proj, emb = pe.get_weights()
    Wlin = _coerce_linear(w_proj, m_pt.encoder.projection.weight)
    ok(Wlin, m_pt.encoder.projection.weight, "patch_encoder projection kernel")
    ok(b_proj, m_pt.encoder.projection.bias, "patch_encoder projection bias")
    ok(emb, m_pt.encoder.position_embedding.weight, "patch_encoder position_embedding")

    # --- Transformer × 4 ---
    mha_tf_idx = [38, 49, 60, 71]
    dense_pairs = [(42, 44), (53, 55), (64, 66), (75, 77)]
    ln_pre_mha_tf = [37, 48, 59, 70]
    ln_mid_tf = [41, 52, 63, 74]

    for b in range(4):
        attn = m_pt.transformer[b].attn
        mha = model_p_tf.layers[mha_tf_idx[b]]
        assert isinstance(mha, tf.keras.layers.MultiHeadAttention), mha
        wg = mha.get_weights()
        # 0–1 Q, 2–3 K, 4–5 V, 6–7 output
        pairs = [
            ("q", wg[0], wg[1]),
            ("k", wg[2], wg[3]),
            ("v", wg[4], wg[5]),
        ]
        for name, wk, wb in pairs:
            lin = getattr(attn, name)
            wc = _coerce_mha_qkv_weight(wk, lin.weight)
            bc = _coerce_mha_bias(wb, lin.bias)
            ok(wc, lin.weight, f"blk{b} MHA {name} kernel")
            ok(bc, lin.bias, f"blk{b} MHA {name} bias")
        w_o, b_o = wg[6], wg[7]
        wc = _coerce_mha_out_weight(w_o, attn.o.weight)
        bc = _coerce_mha_bias(b_o, attn.o.bias)
        ok(wc, attn.o.weight, f"blk{b} MHA out kernel")
        ok(bc, attn.o.bias, f"blk{b} MHA out bias")

        for kind, idx in ("norm1", ln_pre_mha_tf[b]), ("norm2", ln_mid_tf[b]):
            ln = model_p_tf.layers[idx]
            g, bb = ln.get_weights()
            mod = getattr(m_pt.transformer[b], kind)
            norm_name = f"transformer.{b}.{kind}"
            ok(g, mod.weight, f"{norm_name}.weight (TF {ln.name})")
            ok(bb, mod.bias, f"{norm_name}.bias")

        d1i, d2i = dense_pairs[b]
        d1 = model_p_tf.layers[d1i]
        d2 = model_p_tf.layers[d2i]
        W1, B1 = d1.get_weights()
        W2, B2 = d2.get_weights()
        mlp = m_pt.transformer[b].mlp
        W1p = _coerce_linear(W1, mlp.fc1.weight)
        W2p = _coerce_linear(W2, mlp.fc2.weight)
        ok(W1p, mlp.fc1.weight, f"blk{b} dense fc1 kernel ({d1.name})")
        ok(B1, mlp.fc1.bias, f"blk{b} dense fc1 bias")
        ok(W2p, mlp.fc2.weight, f"blk{b} dense fc2 kernel ({d2.name})")
        ok(B2, mlp.fc2.bias, f"blk{b} dense fc2 bias")

    # Final LN (TF layer 81)
    ln_f = model_p_tf.layers[81]
    g, bb = ln_f.get_weights()
    ok(g, m_pt.norm.weight, "final norm.weight")
    ok(bb, m_pt.norm.bias, "final norm.bias")

    # picker_P (TF 83)
    pick = model_p_tf.layers[83]
    wk, bk = pick.get_weights()
    w_c = _coerce_conv1d(wk, m_pt.head.conv.weight)
    ok(w_c, m_pt.head.conv.weight, "picker_P kernel")
    ok(bk, m_pt.head.conv.bias, "picker_P bias")

    return errors


def _pt_block0_trace(m: EQCCTModelP, xt: torch.Tensor, stop: str) -> torch.Tensor:
    """Mirror transformer block 0 up to ``stop`` (for fine-grained TF layer alignment)."""
    x = xt.transpose(1, 2)
    x = m.conv1(x)
    x = m.conv2(x)
    x = m.conv3(x)
    x = x.unsqueeze(2).permute(0, 3, 2, 1)
    x = m.patch(x)
    x = m.encoder(x)
    blk = m.transformer[0]
    identity = x
    u = blk.norm1(x)
    if stop == "b0_after_norm1":
        return u
    attn_out = blk.attn(u)
    if stop == "b0_after_mha":
        return attn_out
    attn_out = blk.drop_path1(attn_out)
    if stop == "b0_after_drop_path_attn":
        return attn_out
    x = identity + attn_out
    if stop == "b0_after_add_attn":
        return x
    identity = x
    u2 = blk.norm2(x)
    if stop == "b0_after_norm2":
        return u2
    mlp_out = blk.mlp(u2)
    if stop == "b0_after_mlp":
        return mlp_out
    mlp_out = blk.drop_path2(mlp_out)
    if stop == "b0_after_drop_path_mlp":
        return mlp_out
    x = identity + mlp_out
    if stop == "b0_after_block":
        return x
    raise ValueError(stop)


# TF layer indices for block 0 (see ``python -c`` introspection on modelP).
_BLOCK0_TF_SUB = (
    ("b0_after_norm1", 37),
    ("b0_after_mha", 38),
    ("b0_after_drop_path_attn", 39),
    ("b0_after_add_attn", 40),
    ("b0_after_norm2", 41),
    ("b0_after_mlp", 45),  # output of second Dense+GELU in mlp (before drop_path2)
    ("b0_after_drop_path_mlp", 46),
    ("b0_after_block", 47),
)


def compare_block0_substeps(
    model_p_tf,
    m_pt: EQCCTModelP,
    x_np: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> list[str]:
    """First transformer block only: find first opacity mismatch."""
    xt = torch.from_numpy(x_np)
    m_pt.eval()
    errs: list[str] = []
    with torch.no_grad():
        for key, tf_i in _BLOCK0_TF_SUB:
            try:
                y_tf = _tf_subforward(model_p_tf, tf_i, x_np)
            except Exception as e:
                errs.append(f"[block0 {key}] TF submodel failed: {e!r}")
                continue
            y_pt = _pt_block0_trace(m_pt, xt, key)
            a = np.asarray(y_tf)
            b = _np(y_pt)
            if a.shape != b.shape:
                errs.append(f"[block0 {key}] shape TF {a.shape} vs PT {b.shape}")
                continue
            d = np.abs(a - b)
            mx = float(d.max())
            me = float(d.mean())
            if np.allclose(a, b, rtol=rtol, atol=atol):
                print(f"[ok] block0 {key}: max|diff|={mx:.3g}")
            else:
                errs.append(
                    f"[block0 {key}] max|diff|={mx:.6g} mean={me:.6g} "
                    f"(rtol={rtol} atol={atol}) — first serious mismatch here implicates "
                    f"the subgraph above this checkpoint."
                )
                break
    return errs


def compare_activations(
    model_p_tf,
    m_pt: EQCCTModelP,
    x_np: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> list[str]:
    import tensorflow as tf

    xt = torch.from_numpy(x_np)
    m_pt.eval()
    errs: list[str] = []
    with torch.no_grad():
        for st in _TF_ACTIVATION_STAGES:
            try:
                y_tf = _tf_subforward(model_p_tf, st.tf_layer_index, x_np)
            except Exception as e:
                errs.append(f"[{st.key}] TF submodel failed: {e!r}")
                continue
            y_pt_t = st.pt_fn(m_pt, xt)
            a = st.tf_to_common(np.asarray(y_tf))
            b = st.pt_to_common(y_pt_t)
            if a.shape != b.shape:
                errs.append(f"[{st.key}] shape TF {a.shape} vs PT(as common) {b.shape}")
                continue
            d = np.abs(a - b)
            mx = float(d.max())
            me = float(d.mean())
            if not np.allclose(a, b, rtol=rtol, atol=atol):
                errs.append(
                    f"[{st.key}] activation mismatch max|diff|={mx:.6g} mean={me:.6g} "
                    f"(rtol={rtol} atol={atol})"
                )
            else:
                print(f"[ok] activations {st.key}: max|diff|={mx:.3g}")
    return errs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TF vs PT weight and activation trace (EQCCT P model)")
    parser.add_argument("--p-h5", type=Path, default=None)
    parser.add_argument("--s-h5", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated RNG seeds (e.g. 0,1,2,3,4). Runs activation checks "
        "for each; weights are compared once. If omitted, uses --seed only.",
    )
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--act-rtol", type=float, default=1e-4)
    parser.add_argument("--act-atol", type=float, default=1e-4)
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--skip-activations", action="store_true")
    parser.add_argument(
        "--block0-substeps",
        action="store_true",
        help="After coarse stages, run fine checkpoints inside transformer block 0 "
        "(stops at first mismatch).",
    )
    parser.add_argument(
        "--tf-on-gpu",
        action="store_true",
        help="Allow TensorFlow GPU (default: CPU only).",
    )
    parser.add_argument(
        "--save-model",
        "--save_model",
        type=Path,
        default=None,
        dest="save_model",
        help="Write a PyTorch checkpoint (.pt): state_dict plus meta (p_h5, s_h5 paths).",
    )
    args = parser.parse_args(argv)

    p_h5 = args.p_h5 or MODELPS_DIR / "test_trainer_024.h5"
    s_h5 = args.s_h5 or MODELPS_DIR / "test_trainer_021.h5"

    if not p_h5.is_file() or not s_h5.is_file():
        print(f"Missing weights p={p_h5} s={s_h5}", file=sys.stderr)
        return 2

    if args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if not seed_list:
            print("--seeds was empty", file=sys.stderr)
            return 2
    else:
        seed_list = [args.seed]

    m_pt = EQCCTModelP()
    load_eqcct_model_p_weights(m_pt, h5_path=str(p_h5))
    m_pt.eval()

    if args.save_model is not None:
        out = Path(args.save_model)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": m_pt.state_dict(),
                "meta": {
                    "class": "EQCCTModelP",
                    "p_h5": str(p_h5.resolve()),
                    "s_h5": str(s_h5.resolve()),
                },
            },
            out,
        )
        print(f"[info] Saved PyTorch checkpoint: {out.resolve()}")

    # Avoid importing TensorFlow / predictor_tf (and heavy deps like ObsPy) when
    # only saving or when there is nothing left to compare.
    if args.skip_weights and args.skip_activations:
        return 0

    if not args.tf_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow required.", file=sys.stderr)
        return 2

    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass

    from reference.predictor_tf import load_eqcct_model

    model_p_tf, _ = load_eqcct_model(str(p_h5), str(s_h5))

    bad = False
    if not args.skip_weights:
        print("--- Weight comparison (TF layer.get_weights vs PT state_dict) ---")
        werr = compare_weights(model_p_tf, m_pt, rtol=args.rtol, atol=args.atol)
        if werr:
            bad = True
            for e in werr:
                print("[weights]", e, file=sys.stderr)
        else:
            print(f"[ok] All compared weights match (rtol={args.rtol} atol={args.atol}).")

    for run_seed in seed_list:
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        tf.random.set_seed(run_seed)
        x = np.random.randn(2, 6000, 3).astype(np.float32)

        if not args.skip_activations:
            tag = f"seed={run_seed}" if len(seed_list) > 1 else ""
            print(
                f"--- Activation comparison (TF subgraph vs PT forward fragment){' — ' + tag if tag else ''} ---"
            )
            aerr = compare_activations(
                model_p_tf, m_pt, x, rtol=args.act_rtol, atol=args.act_atol
            )
            if aerr:
                bad = True
                for e in aerr:
                    print(f"[activations seed={run_seed}]", e, file=sys.stderr)
            elif not aerr and args.skip_weights and len(seed_list) == 1:
                print("[ok] All activation stages within tolerance.")

        if args.block0_substeps and not args.skip_activations:
            print(
                f"--- Block 0 fine substeps — seed={run_seed} ---"
                if len(seed_list) > 1
                else "--- Block 0 fine substeps (TF layer vs PT partial forward) ---"
            )
            sub_err = compare_block0_substeps(
                model_p_tf, m_pt, x, rtol=args.act_rtol, atol=args.act_atol
            )
            if sub_err:
                bad = True
                for e in sub_err:
                    print(f"[block0 seed={run_seed}]", e, file=sys.stderr)

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
