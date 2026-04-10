import re 
import numpy as np 
from collections import defaultdict, Counter

def flatten_h5dict(d):
    """Return a flat dict of leaf tensors using their own absolute keys."""
    flat = {}
    def rec(node):
        for k, v in node.items():
            if isinstance(v, dict):
                rec(v)
            else:
                flat[k] = v   # keep k as-is; don't prefix
    rec(d)
    return flat

def assert_and_print(flat, *, patch_dim=1600, proj_dim=40, num_patches=150, head_kw=15):
    """
    Sanity-check presence & shapes of key tensors and print a compact summary.
    Raises if anything essential is missing or has an unexpected shape.
    """

    def _one(pattern, what):
        ks = [k for k in flat if re.fullmatch(pattern, k)]
        if len(ks) != 1:
            raise AssertionError(f"{what}: expected 1 match, got {len(ks)} -> {ks[:6]}")
        return ks[0]

    def _sh(a):  # shape string
        return "x".join(map(str, a.shape))

    print("— sanity check —")

    # ---- PatchEncoder ----
    pe_w_k = _one(r"patch_encoder/dense/kernel:0",   "PatchEncoder kernel")
    pe_b_k = _one(r"patch_encoder/dense/bias:0",     "PatchEncoder bias")
    pe_e_k = _one(r"patch_encoder/embedding/embeddings:0", "PatchEncoder pos-emb")

    pe_w = flat[pe_w_k]; pe_b = flat[pe_b_k]; pe_e = flat[pe_e_k]
    print(f"PatchEncoder: kernel {_sh(pe_w)}, bias {_sh(pe_b)}, pos-emb {_sh(pe_e)}")
    assert pe_w.shape == (patch_dim, proj_dim), f"PatchEncoder kernel expected {(patch_dim, proj_dim)}"
    assert pe_b.shape == (proj_dim,),           f"PatchEncoder bias expected {(proj_dim,)}"
    assert pe_e.shape == (num_patches, proj_dim), f"Pos-emb expected {(num_patches, proj_dim)}"

    # ---- Output heads ----
    have_picker_S = "picker_S/kernel:0" in flat and "picker_S/bias:0" in flat
    have_picker_P = "picker_P/kernel:0" in flat and "picker_P/bias:0" in flat
    print(f"Heads present: picker_S={have_picker_S}, picker_P={have_picker_P}")

    if have_picker_S:
        sW, sB = flat["picker_S/kernel:0"], flat["picker_S/bias:0"]
        print(f"picker_S: kernel {_sh(sW)}, bias {_sh(sB)}")
        # Typical TF Conv1D layout (kw, inC, outC)
        assert sW.ndim == 3 and sW.shape[0] == head_kw and sW.shape[1] == 1, "picker_S kernel expected (15,1,1)"
        assert sB.shape == (1,), "picker_S bias expected (1,)"

    if have_picker_P:
        pW, pB = flat["picker_P/kernel:0"], flat["picker_P/bias:0"]
        print(f"picker_P: kernel {_sh(pW)}, bias {_sh(pB)}")

    # ---- Conv1D / BN / Dense / LN counts ----
    conv_k = sorted([k for k in flat if re.search(r"(^|/)conv1d(?:_\d+)?/kernel:0$", k)])
    conv_b = sorted([k for k in flat if re.search(r"(^|/)conv1d(?:_\d+)?/bias:0$",   k)])
    bn_g   = sorted([k for k in flat if re.search(r"(^|/)batch_normalization(?:_\d+)?/gamma:0$", k)])
    bn_b   = sorted([k for k in flat if re.search(r"(^|/)batch_normalization(?:_\d+)?/beta:0$",  k)])
    bn_mm  = sorted([k for k in flat if re.search(r"(^|/)batch_normalization(?:_\d+)?/moving_mean:0$", k)])
    bn_mv  = sorted([k for k in flat if re.search(r"(^|/)batch_normalization(?:_\d+)?/moving_variance:0$", k)])
    dense_k = sorted([k for k in flat if re.search(r"(^|/)dense(?:_\d+)?/kernel:0$", k)])
    dense_b = sorted([k for k in flat if re.search(r"(^|/)dense(?:_\d+)?/bias:0$",   k)])
    ln_g    = sorted([k for k in flat if re.search(r"(^|/)layer_normalization(?:_\d+)?/gamma:0$", k)])
    ln_bet  = sorted([k for k in flat if re.search(r"(^|/)layer_normalization(?:_\d+)?/beta:0$",  k)])

    print(f"Counts — conv1d: {len(conv_k)} kernels, bn: {len(bn_g)} γ/{len(bn_b)} β,"
          f" dense: {len(dense_k)}, ln: {len(ln_g)}")

    # Show a small histogram of Conv1D kernel shapes (helps separate S vs P branch)
    shape_hist = Counter([flat[k].shape for k in conv_k])
    top = ", ".join([f"{sh}×{cnt}" for sh, cnt in list(shape_hist.items())[:6]])
    print(f"conv1d kernel shapes (first few): {top}")

    # ---- MHA blocks (robust to output / output_dense naming) ----
    mha = defaultdict(lambda: defaultdict(dict))
    for k in flat: 
        m = re.search(r'(^|/)(multi_head_attention(?:_\d+)?)/'
                      r'((?:query|key|value)|(?:output(?:_dense)?)|(?:attention_output(?:_dense)?))/'
                      r'(kernel|bias):0$',
                      k)
        if m:
            blk = m.group(2)                          # e.g. "multi_head_attention_3"
            raw_part = m.group(3)                     # "query" | "key" | "value" | "output" | "output_dense" | "attention_output"...
            what = m.group(4)                         # "kernel" | "bias"
            part = "output" if raw_part.startswith(("output", "attention_output")) else raw_part
            mha[blk][part][what] = flat[k]

    print(f"MHA blocks found: {len(mha)}")
    for blk in sorted(mha.keys()):
        parts = mha[blk]
        def shp(p,w): 
            arr = parts.get(p,{}).get(w)
            return None if arr is None else arr.shape
        qk = shp("query","kernel") or shp("key","kernel")
        ov = shp("output","kernel")
        print(f"  {blk}: Q/K kernel {qk}, V kernel {shp('value','kernel')}, "
              f"out kernel {ov}")

        # Basic sanity: expect either 3D (E,H,D) or 2D (E,H*D)
        if qk is not None:
            assert (len(qk) == 3 and qk[0] == proj_dim) or (len(qk) == 2 and qk[0] == proj_dim), \
                   f"{blk} query/key kernel unexpected shape {qk}"
        if ov is not None:
            assert (len(ov) == 3 and ov[-1] == proj_dim) or (len(ov) == 2 and ov[-1] == proj_dim), \
                   f"{blk} output kernel unexpected shape {ov}"

    print("— sanity ok —")
    return {
        "patch_encoder": {"dense_w": pe_w_k, "dense_b": pe_b_k, "pos": pe_e_k},
        "heads": {"picker_S": have_picker_S, "picker_P": have_picker_P},
        "conv1d_kernels": conv_k,
        "bn": {"gamma": bn_g, "beta": bn_b, "mean": bn_mm, "var": bn_mv},
        "dense": {"kernels": dense_k, "biases": dense_b},
        "layernorm": {"gamma": ln_g, "beta": ln_bet},
        "mha": mha,
    }