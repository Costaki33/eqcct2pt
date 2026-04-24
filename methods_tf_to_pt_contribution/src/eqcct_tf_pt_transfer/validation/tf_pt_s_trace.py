#!/usr/bin/env python3
"""
Per-parameter and per-stage activation trace: TensorFlow ``modelS`` vs PyTorch ``EQCCTModelS``.

Mirrors ``tf_pt_p_trace`` for the S branch (extra pre/post ConvF1 around each transformer block).

Run from the **methods bundle root** (folder with ``README.md``), with ``src`` on
``PYTHONPATH``::

    cd /path/to/methods_tf_to_pt_contribution
    PYTHONPATH=src python -m eqcct_tf_pt_transfer.validation.tf_pt_s_trace \\
      --p-h5 ModelPS/test_trainer_024.h5 --s-h5 ModelPS/test_trainer_021.h5

Save PyTorch ``.pt`` only (no TF)::

    PYTHONPATH=src python -m eqcct_tf_pt_transfer.validation.tf_pt_s_trace \\
      --p-h5 ... --s-h5 ... --skip-weights --skip-activations --save-model ModelPS/eqcct_model_s.pt

TensorFlow layer indices match the graph from ``load_eqcct_model`` → ``modelS`` (see module
source dump). If ``create_cct_modelS`` changes, refresh ``_TF_ACTIVATION_STAGES``, weight
indices, and ``_BLOCK0_TF_SUB``.
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

from eqcct_tf_pt_transfer.conversion.loader import (
    _coerce_conv1d,
    _coerce_linear,
    _coerce_mha_bias,
    _coerce_mha_out_weight,
    _coerce_mha_qkv_weight,
    load_eqcct_model_s_weights,
)
from eqcct_tf_pt_transfer.models.predictor_pt_p import EQCCTModelS


def _repo_root() -> Path:
    """Bundle root (``README`` lives here; default ``ModelPS/`` paths are under this directory)."""
    return Path(__file__).resolve().parents[3]


def _np(p: torch.Tensor) -> np.ndarray:
    return p.detach().cpu().numpy()


@dataclass(frozen=True)
class ActStage:
    key: str
    tf_layer_index: int
    pt_fn: Callable[[EQCCTModelS, torch.Tensor], torch.Tensor]
    layout: str

    def tf_to_common(self, y: np.ndarray) -> np.ndarray:
        if self.layout == "nlc":
            return y
        if self.layout == "ncl_pt":
            return y
        raise ValueError(self.layout)

    def pt_to_common(self, y: torch.Tensor) -> np.ndarray:
        t = _np(y)
        if self.layout == "nlc":
            return t
        if self.layout == "ncl_pt":
            return np.transpose(t, (0, 2, 1))
        raise ValueError(self.layout)


# Layer indices from ``enumerate(modelS.layers)`` after ``load_eqcct_model`` (see repo history).
_TF_ACTIVATION_STAGES: tuple[ActStage, ...] = (
    ActStage(
        "post_front (after 3x ConvF1, channels-last)",
        33,
        lambda m, xt: _pt_feed_to_s(m, xt, through="conv3"),
        "ncl_pt",
    ),
    ActStage(
        "post_patch_encoder (B,150,40)",
        36,
        lambda m, xt: _pt_feed_to_s(m, xt, through="encoder"),
        "nlc",
    ),
    ActStage("post_transformer_block0", 69, lambda m, xt: _pt_feed_to_s(m, xt, through="tb0"), "nlc"),
    ActStage("post_transformer_block1", 102, lambda m, xt: _pt_feed_to_s(m, xt, through="tb1"), "nlc"),
    ActStage("post_transformer_block2", 135, lambda m, xt: _pt_feed_to_s(m, xt, through="tb2"), "nlc"),
    ActStage("post_transformer_block3", 168, lambda m, xt: _pt_feed_to_s(m, xt, through="tb3"), "nlc"),
    ActStage(
        "pre_picker_after_final_LN (B,150,40)",
        169,
        lambda m, xt: _pt_feed_to_s(m, xt, through="norm"),
        "nlc",
    ),
    ActStage(
        "pre_picker_after_reshape (B,6000,1)",
        170,
        lambda m, xt: _pt_feed_to_s(m, xt, through="reshape"),
        "nlc",
    ),
)


def _pt_feed_to_s(m: EQCCTModelS, xt: torch.Tensor, *, through: str) -> torch.Tensor:
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
    for i in range(4):
        x_pre = x.transpose(1, 2)
        x_pre = m.extra_pre[i](x_pre).transpose(1, 2)
        x = x_pre
        identity = x
        x_norm1 = m.transformers[i].norm1(x)
        attn_out = m.transformers[i].attn(x_norm1)
        attn_out = m.extra_post[i](attn_out.transpose(1, 2)).transpose(1, 2)
        x = identity + m.transformers[i].drop_path1(attn_out)
        identity2 = x
        x = identity2 + m.transformers[i].drop_path2(
            m.transformers[i].mlp(m.transformers[i].norm2(x))
        )
        if through == f"tb{i}":
            return x

    x = m.norm(x)
    if through == "norm":
        return x
    x = x.reshape(x.size(0), 6000, 1)
    if through == "reshape":
        return x
    raise ValueError(through)


def _tf_subforward(model_s_tf, layer_index: int, x_np: np.ndarray):
    import tensorflow as tf

    lay = model_s_tf.layers[layer_index]
    sub = tf.keras.Model(model_s_tf.input, lay.output)
    y = sub(x_np, training=False)
    if hasattr(y, "numpy"):
        y = y.numpy()
    return np.asarray(y)


def compare_weights(
    model_s_tf,
    m_pt: EQCCTModelS,
    *,
    rtol: float,
    atol: float,
) -> list[str]:
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

    conv_idx = [
        1,
        4,
        8,
        12,
        15,
        19,
        23,
        26,
        30,
        37,
        40,
        44,
        50,
        53,
        57,
        70,
        73,
        77,
        83,
        86,
        90,
        103,
        106,
        110,
        116,
        119,
        123,
        136,
        139,
        143,
        149,
        152,
        156,
    ]
    bn_idx = [
        2,
        5,
        9,
        13,
        16,
        20,
        24,
        27,
        31,
        38,
        41,
        45,
        51,
        54,
        58,
        71,
        74,
        78,
        84,
        87,
        91,
        104,
        107,
        111,
        117,
        120,
        124,
        137,
        140,
        144,
        150,
        153,
        157,
    ]

    towers = [
        ("conv1.conv1", "conv1.conv2", "conv1.conv3"),
        ("conv2.conv1", "conv2.conv2", "conv2.conv3"),
        ("conv3.conv1", "conv3.conv2", "conv3.conv3"),
    ]
    for bi in range(4):
        for stem in ("extra_pre", "extra_post"):
            towers.append(
                (
                    f"{stem}.{bi}.conv1",
                    f"{stem}.{bi}.conv2",
                    f"{stem}.{bi}.conv3",
                )
            )
    flat_conv = [c for t in towers for c in t]
    assert len(flat_conv) == len(conv_idx) == len(bn_idx) == 33

    for i, (c_tf, prefix) in enumerate(zip(conv_idx, flat_conv)):
        lay = model_s_tf.layers[c_tf]
        assert isinstance(lay, tf.keras.layers.Conv1D), (i, lay)
        w_tf, b_tf = lay.get_weights()
        pw = sd[f"{prefix}.weight"]
        pb = sd[f"{prefix}.bias"]
        w_c = _coerce_conv1d(w_tf, pw)
        ok(w_c, pw, f"Conv1D {prefix}.weight (TF layer {c_tf} {lay.name})")
        ok(b_tf, pb, f"Conv1D {prefix}.bias")

        b_tf_l = model_s_tf.layers[bn_idx[i]]
        assert isinstance(b_tf_l, tf.keras.layers.BatchNormalization), b_tf_l
        gamma, beta, mm, mv = b_tf_l.get_weights()
        prefix_bn = (
            prefix.replace(".conv1", ".bn1").replace(".conv2", ".bn2").replace(".conv3", ".bn3")
        )
        ok(gamma, sd[f"{prefix_bn}.weight"], f"BN {prefix_bn}.weight (gamma)")
        ok(beta, sd[f"{prefix_bn}.bias"], f"BN {prefix_bn}.bias")
        ok(mm, sd[f"{prefix_bn}.running_mean"], f"BN {prefix_bn}.running_mean")
        ok(mv, sd[f"{prefix_bn}.running_var"], f"BN {prefix_bn}.running_var")

    pe = model_s_tf.layers[36]
    w_proj, b_proj, emb = pe.get_weights()
    Wlin = _coerce_linear(w_proj, m_pt.encoder.projection.weight)
    ok(Wlin, m_pt.encoder.projection.weight, "patch_encoder projection kernel")
    ok(b_proj, m_pt.encoder.projection.bias, "patch_encoder projection bias")
    ok(emb, m_pt.encoder.position_embedding.weight, "patch_encoder position_embedding")

    mha_tf_idx = [49, 82, 115, 148]
    dense_pairs = [(64, 66), (97, 99), (130, 132), (163, 165)]
    ln_pre_mha_tf = [48, 81, 114, 147]
    ln_mid_tf = [63, 96, 129, 162]

    for b in range(4):
        attn = m_pt.transformers[b].attn
        mha = model_s_tf.layers[mha_tf_idx[b]]
        assert isinstance(mha, tf.keras.layers.MultiHeadAttention), mha
        wg = mha.get_weights()
        pairs = [("q", wg[0], wg[1]), ("k", wg[2], wg[3]), ("v", wg[4], wg[5])]
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
            ln = model_s_tf.layers[idx]
            g, bb = ln.get_weights()
            mod = getattr(m_pt.transformers[b], kind)
            norm_name = f"transformers.{b}.{kind}"
            ok(g, mod.weight, f"{norm_name}.weight (TF {ln.name})")
            ok(bb, mod.bias, f"{norm_name}.bias")

        d1i, d2i = dense_pairs[b]
        d1 = model_s_tf.layers[d1i]
        d2 = model_s_tf.layers[d2i]
        W1, B1 = d1.get_weights()
        W2, B2 = d2.get_weights()
        mlp = m_pt.transformers[b].mlp
        W1p = _coerce_linear(W1, mlp.fc1.weight)
        W2p = _coerce_linear(W2, mlp.fc2.weight)
        ok(W1p, mlp.fc1.weight, f"blk{b} dense fc1 kernel ({d1.name})")
        ok(B1, mlp.fc1.bias, f"blk{b} dense fc1 bias")
        ok(W2p, mlp.fc2.weight, f"blk{b} dense fc2 kernel ({d2.name})")
        ok(B2, mlp.fc2.bias, f"blk{b} dense fc2 bias")

    ln_f = model_s_tf.layers[169]
    g, bb = ln_f.get_weights()
    ok(g, m_pt.norm.weight, "final norm.weight")
    ok(bb, m_pt.norm.bias, "final norm.bias")

    pick = model_s_tf.layers[171]
    wk, bk = pick.get_weights()
    w_c = _coerce_conv1d(wk, m_pt.head.conv.weight)
    ok(w_c, m_pt.head.conv.weight, "picker_S kernel")
    ok(bk, m_pt.head.conv.bias, "picker_S bias")

    return errors


def _pt_block0_trace_s(m: EQCCTModelS, xt: torch.Tensor, stop: str) -> torch.Tensor:
    x = xt.transpose(1, 2)
    x = m.conv1(x)
    x = m.conv2(x)
    x = m.conv3(x)
    x = x.unsqueeze(2).permute(0, 3, 2, 1)
    x = m.patch(x)
    x = m.encoder(x)
    blk = m.transformers[0]
    x = m.extra_pre[0](x.transpose(1, 2)).transpose(1, 2)
    identity = x
    u = blk.norm1(x)
    if stop == "b0_after_norm1":
        return u
    attn_out = blk.attn(u)
    if stop == "b0_after_mha":
        return attn_out
    attn_pc = m.extra_post[0](attn_out.transpose(1, 2)).transpose(1, 2)
    attn_pc = blk.drop_path1(attn_pc)
    if stop == "b0_after_drop_path_attn":
        return attn_pc
    x = identity + attn_pc
    if stop == "b0_after_add_attn":
        return x
    identity2 = x
    u2 = blk.norm2(x)
    if stop == "b0_after_norm2":
        return u2
    mlp_out = blk.mlp(u2)
    if stop == "b0_after_mlp":
        return mlp_out
    mlp_out = blk.drop_path2(mlp_out)
    if stop == "b0_after_drop_path_mlp":
        return mlp_out
    x = identity2 + mlp_out
    if stop == "b0_after_block":
        return x
    raise ValueError(stop)


_BLOCK0_TF_SUB_S = (
    ("b0_after_norm1", 48),
    ("b0_after_mha", 49),
    ("b0_after_drop_path_attn", 61),
    ("b0_after_add_attn", 62),
    ("b0_after_norm2", 63),
    ("b0_after_mlp", 66),
    ("b0_after_drop_path_mlp", 68),
    ("b0_after_block", 69),
)


def compare_block0_substeps(
    model_s_tf,
    m_pt: EQCCTModelS,
    x_np: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> list[str]:
    xt = torch.from_numpy(x_np)
    m_pt.eval()
    errs: list[str] = []
    with torch.no_grad():
        for key, tf_i in _BLOCK0_TF_SUB_S:
            try:
                y_tf = _tf_subforward(model_s_tf, tf_i, x_np)
            except Exception as e:
                errs.append(f"[block0 {key}] TF submodel failed: {e!r}")
                continue
            y_pt = _pt_block0_trace_s(m_pt, xt, key)
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
    model_s_tf,
    m_pt: EQCCTModelS,
    x_np: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> list[str]:
    xt = torch.from_numpy(x_np)
    m_pt.eval()
    errs: list[str] = []
    with torch.no_grad():
        for st in _TF_ACTIVATION_STAGES:
            try:
                y_tf = _tf_subforward(model_s_tf, st.tf_layer_index, x_np)
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
    parser = argparse.ArgumentParser(description="TF vs PT weight and activation trace (EQCCT S model)")
    parser.add_argument("--p-h5", type=Path, default=None)
    parser.add_argument("--s-h5", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated RNG seeds for activation checks (weights once).",
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
        help="Fine checkpoints inside S transformer block 0.",
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
        help="Write PyTorch .pt (EQCCTModelS state_dict + meta).",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    p_h5 = args.p_h5 or root / "ModelPS" / "test_trainer_024.h5"
    s_h5 = args.s_h5 or root / "ModelPS" / "test_trainer_021.h5"

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

    m_pt = EQCCTModelS()
    load_eqcct_model_s_weights(m_pt, h5_path=str(s_h5))
    m_pt.eval()

    if args.save_model is not None:
        out = Path(args.save_model)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": m_pt.state_dict(),
                "meta": {
                    "class": "EQCCTModelS",
                    "p_h5": str(p_h5.resolve()),
                    "s_h5": str(s_h5.resolve()),
                },
            },
            out,
        )
        print(f"[info] Saved PyTorch checkpoint: {out.resolve()}")

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

    from eqcct_tf_pt_transfer.reference.predictor_tf import load_eqcct_model

    _, model_s_tf = load_eqcct_model(str(p_h5), str(s_h5))

    bad = False
    if not args.skip_weights:
        print("--- Weight comparison (TF modelS vs PT state_dict) ---")
        werr = compare_weights(model_s_tf, m_pt, rtol=args.rtol, atol=args.atol)
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
                f"--- Activation comparison (TF subgraph vs PT fragment){' — ' + tag if tag else ''} ---"
            )
            aerr = compare_activations(
                model_s_tf, m_pt, x, rtol=args.act_rtol, atol=args.act_atol
            )
            if aerr:
                bad = True
                for e in aerr:
                    print(f"[activations seed={run_seed}]", e, file=sys.stderr)
            elif not aerr and args.skip_weights and len(seed_list) == 1:
                print("[ok] All activation stages within tolerance.")

        if args.block0_substeps and not args.skip_activations:
            print(
                f"--- S block 0 substeps — seed={run_seed} ---"
                if len(seed_list) > 1
                else "--- S block 0 substeps ---"
            )
            sub_err = compare_block0_substeps(
                model_s_tf, m_pt, x, rtol=args.act_rtol, atol=args.act_atol
            )
            if sub_err:
                bad = True
                for e in sub_err:
                    print(f"[block0 seed={run_seed}]", e, file=sys.stderr)

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
