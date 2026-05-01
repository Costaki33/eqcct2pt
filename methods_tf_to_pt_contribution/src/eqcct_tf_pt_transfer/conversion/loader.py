import pickle
import re
import h5py
import torch
import numpy as np
from eqcct_tf_pt_transfer.models.predictor_pt_p import EQCCTModelP, EQCCTModelS


def collapse_duplicate_layer_prefix(path: str) -> str:
    """
    Keras H5 checkpoints often use duplicated path segments, e.g.
    conv1d/conv1d/kernel:0 -> conv1d/kernel:0
    """
    parts = path.split("/")
    out = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i] == parts[i + 1]:
            out.append(parts[i])
            i += 2
        else:
            out.append(parts[i])
            i += 1
    return "/".join(out)


def flat_torchish_from_h5(h5_path: str) -> dict:
    """
    Load a Keras .h5 weight file into a flat dict with slash keys matching
    catalog.assert_and_print expectations (e.g. patch_encoder/dense/kernel:0).
    """
    out = {}
    with h5py.File(h5_path, "r") as f:

        def walk(g, prefix=""):
            for k, v in g.items():
                if isinstance(v, h5py.Dataset):
                    full = prefix + k
                    out[collapse_duplicate_layer_prefix(full)] = np.array(v)
                else:
                    walk(v, prefix + k + "/")

        walk(f)
    return out


def load_keras_h5_as_numpy_dict(h5_path):
    """
    Flattens the H5 weight file into a dict:
    {layer_name: [np.array(var_0), np.array(var_1), ...]}
    Compatible with get_tf_layer_names(...).
    """
    out = {}
    with h5py.File(h5_path, 'r') as f:
        for layer_name, group in f.items(): # Iterates through each top-level key in the hdf5 file 
            # print(f"Layer name: {layer_name} | Group: {group}")
            # Keras H5: layer/<weight_names> or weight names inside subgroups
            store_weights = {}
            def collect(g, prefix=''):
                for k, v in g.items():
                    if isinstance(v, h5py.Dataset): # Dataset contains actual weights/biases 
                        store_weights[prefix + k] = np.array(v) # Store weights
                    elif isinstance(v, h5py.Group): # If finds Group (subdir)
                        collect(v, prefix + k + '/') # Calls collect to go deeper into that group
            collect(group)
            if store_weights: 
                # Now we clean the stored_weights into the main output dir, using layer name as key
                out[layer_name] = store_weights 
    return out

def _as_tensor(x, like_param):
    return torch.from_numpy(np.asarray(x)).to(like_param.device, like_param.dtype)

def _grab(d, *candidates, regex=None, desc=""):
    for k in candidates:
        if k in d:
            return k
    if regex:
        matches = sorted([k for k in d if re.fullmatch(regex, k)])
        if matches:
            return matches[0]
    # helpful error: show what exists in this area
    hint = [k for k in d if 'patch_encoder' in k] if 'PatchEncoder' in (desc or '') else []
    raise KeyError(f"Missing {desc or candidates}. Nearby: {hint[:8]} ...")

def _coerce_linear(W, target_param):
    W = np.asarray(W)
    tgt = tuple(target_param.shape)          # PyTorch Linear: (out, in)
    # Keras Dense kernel is (in_dim, out_dim). Prefer W.T first so square matrices
    # (e.g. 40×40 MLP) are not accepted as already-PT layout when shapes coincide.
    if W.T.shape == tgt:
        return W.T
    if W.shape == tgt:
        return W
    raise AssertionError(f"Linear weight shape {W.shape} can’t match {tgt} (even with .T)")


def _coerce_mha_qkv_weight(W, target_param):
    """Keras MHA (E, H, D) -> nn.Linear weight (H*D, E)."""
    W = np.asarray(W)
    tgt = tuple(target_param.shape)
    if W.shape == tgt:
        return W
    if W.ndim == 3:
        W2 = W.reshape(W.shape[0], -1).T
        if W2.shape == tgt:
            return W2
    return _coerce_linear(W, target_param)


def _coerce_mha_out_weight(W, target_param):
    """Keras MHA output kernel (H, D, E) -> nn.Linear weight (E, H*D)."""
    W = np.asarray(W)
    tgt = tuple(target_param.shape)
    if W.shape == tgt:
        return W
    if W.ndim == 3:
        W2 = W.reshape(-1, W.shape[-1]).T
        if W2.shape == tgt:
            return W2
    return _coerce_linear(W, target_param)


def _coerce_mha_bias(b, target_param):
    b = np.asarray(b).reshape(-1)
    t = tuple(target_param.shape)
    if b.shape != t:
        raise AssertionError(f"MHA bias shape {b.shape} can’t match {t}")
    return b

def _coerce_conv1d(W, target_param):
    W = np.asarray(W)
    tgt = tuple(target_param.shape)          # PyTorch Conv1d: (outC, inC, kw)
    if W.shape == tgt:
        return W
    # TF Conv1D kernel: (kw, inC, outC)
    trans = W.transpose(2,1,0)
    if trans.shape == tgt:
        return trans
    raise AssertionError(f"Conv1d weight shape {W.shape} can’t match {tgt} (even with TF->PT permute)")

def _suffix_idx(name, stem):
    m = re.match(rf'^{re.escape(stem)}(?:_(\d+))?$', name)
    return 0 if m and not m.group(1) else (int(m.group(1)) if m else None)

def load_torchish_into_eqcctS(model: EQCCTModelS, torchish: dict):
    model.eval()

    # ---------------- PatchEncoder ----------------
    # weight
    kW = _grab(
        torchish,
        'patch_encoder.linear.weight',
        'patch_encoder.projection.weight',
        'patch_encoder.dense.weight',
        regex=r'patch_encoder/dense/kernel:0',
        desc="PatchEncoder weight",
    )
    W = _coerce_linear(torchish[kW], model.encoder.projection.weight)
    model.encoder.projection.weight.data.copy_(_as_tensor(W, model.encoder.projection.weight))

    # bias
    kB = _grab(
        torchish,
        'patch_encoder.linear.bias',
        'patch_encoder.projection.bias',
        'patch_encoder.dense.bias',
        regex=r'patch_encoder/dense/bias:0',
        desc="PatchEncoder bias",
    )
    model.encoder.projection.bias.data.copy_(_as_tensor(torchish[kB], model.encoder.projection.bias))

    # pos embedding
    kE = _grab(
        torchish,
        'patch_encoder.pos_embedding',
        'patch_encoder.position_embedding.weight',
        regex=r'patch_encoder/embedding/embeddings:0',
        desc="PatchEncoder pos-embedding",
    )
    model.encoder.position_embedding.weight.data.copy_(
        _as_tensor(torchish[kE], model.encoder.position_embedding.weight)
    )

    # ---------------- MHA (4 blocks) ----------------
    _keras_attn_name = {"q": "query", "k": "key", "v": "value"}

    def copy_mha(i, blk):
        attn = model.transformers[i].attn
        for proj in ("q", "k", "v"):
            # weights
            kw = _grab(
                torchish,
                f"{blk}.{proj}_proj.weight",
                regex=rf'{blk}/{_keras_attn_name[proj]}/kernel:0',
                desc=f"{blk}.{proj} weight",
            )
            W = _coerce_mha_qkv_weight(torchish[kw], getattr(attn, proj).weight)
            getattr(attn, proj).weight.data.copy_(_as_tensor(W, getattr(attn, proj).weight))
            # bias
            kb = _grab(
                torchish,
                f"{blk}.{proj}_proj.bias",
                regex=rf'{blk}/{_keras_attn_name[proj]}/bias:0',
                desc=f"{blk}.{proj} bias",
            )
            bb = _coerce_mha_bias(torchish[kb], getattr(attn, proj).bias)
            getattr(attn, proj).bias.data.copy_(_as_tensor(bb, getattr(attn, proj).bias))

        # output projection (predictor_pt_p uses attn.o; older code used attn.proj)
        out_mod = getattr(attn, "o", None) or getattr(attn, "proj", None)
        if out_mod is None:
            raise AttributeError("Attention module needs .o or .proj Linear")
        kow = _grab(
            torchish,
            f'{blk}.out_proj.weight',
            regex=rf'{blk}/(output|attention_output)(?:_dense)?/kernel:0',
            desc=f'{blk}.out_proj weight'
        )
        W = _coerce_mha_out_weight(torchish[kow], out_mod.weight)
        out_mod.weight.data.copy_(_as_tensor(W, out_mod.weight))

        kob = _grab(
            torchish,
            f'{blk}.out_proj.bias',
            regex=rf'{blk}/(output|attention_output)(?:_dense)?/bias:0',
            desc=f'{blk}.out_proj bias'
        )
        ob = _coerce_mha_bias(torchish[kob], out_mod.bias)
        out_mod.bias.data.copy_(_as_tensor(ob, out_mod.bias))

    names = ['multi_head_attention'] + [f'multi_head_attention_{i}' for i in range(1, 4)]
    for i, blk in enumerate(names):
        copy_mha(i, blk)

    # ---------------- Dense MLPs (fc1/fc2 across 4 blocks) ----------------
    def copy_dense(base, lin):
        kW = _grab(torchish, f'{base}.weight', regex=rf'{base}/kernel:0', desc=f'{base} weight')
        kB = _grab(torchish, f'{base}.bias',   regex=rf'{base}/bias:0',   desc=f'{base} bias')
        W = _coerce_linear(torchish[kW], lin.weight)
        lin.weight.data.copy_(_as_tensor(W, lin.weight))
        lin.bias.data.copy_(_as_tensor(torchish[kB], lin.bias))

    # Keras dense_1..dense_8 — MLP lives under TransformerBlock.mlp in predictor_pt_p
    for i in range(4):
        copy_dense(f'dense_{2*i+1}', model.transformers[i].mlp.fc1)
        copy_dense(f'dense_{2*i+2}', model.transformers[i].mlp.fc2)

    # ---------------- Conv1d + BN (33 each) ----------------
    conv_pt = sorted([k for k in torchish if re.match(r'^conv1d(?:_\d+)?\.weight$', k)],
                     key=lambda k: _suffix_idx(k.split('.')[0], 'conv1d') or 0)
    conv_tf = sorted([k for k in torchish if re.match(r'^conv1d(?:_\d+)?/kernel:0$', k)],
                     key=lambda k: _suffix_idx(k.split('/')[0], 'conv1d') or 0)
    conv_keys = conv_pt or conv_tf
    is_tf = bool(conv_tf and not conv_pt)

    conv_modules = [
        model.conv1.conv1, model.conv1.conv2, model.conv1.conv3,
        model.conv2.conv1, model.conv2.conv2, model.conv2.conv3,
        model.conv3.conv1, model.conv3.conv2, model.conv3.conv3,
    ]
    for i in range(4):
        conv_modules += [
            model.extra_pre[i].conv1, model.extra_pre[i].conv2, model.extra_pre[i].conv3,
            model.extra_post[i].conv1, model.extra_post[i].conv2, model.extra_post[i].conv3,
        ]
    assert len(conv_modules) == len(conv_keys) == 33, f"conv mismatch: {len(conv_modules)} vs {len(conv_keys)}"

    for k, mod in zip(conv_keys, conv_modules):
        W = torchish[k]
        W = _coerce_conv1d(W, mod.weight) if is_tf else np.asarray(W)
        mod.weight.data.copy_(_as_tensor(W, mod.weight))
        bk = k.replace('.weight', '.bias') if not is_tf else k.replace('/kernel:0', '/bias:0')
        if bk in torchish:
            mod.bias.data.copy_(_as_tensor(torchish[bk], mod.bias))

    # BN
    def _bn_collect(pt_pat, tf_name):
        pt = sorted([k for k in torchish if re.match(pt_pat, k)],
                    key=lambda k: _suffix_idx(k.split('.')[0], 'batch_normalization') or 0)
        if pt:
            return pt, False
        tf = sorted([k for k in torchish if re.match(rf'^batch_normalization(?:_\d+)?/{tf_name}:0$', k)],
                    key=lambda k: _suffix_idx(k.split('/')[0], 'batch_normalization') or 0)
        return tf, True

    gamma, _ = _bn_collect(r'^batch_normalization(?:_\d+)?\.weight$', 'gamma')
    beta , _ = _bn_collect(r'^batch_normalization(?:_\d+)?\.bias$',   'beta')
    mean , _ = _bn_collect(r'^batch_normalization(?:_\d+)?\.running_mean$', 'moving_mean')
    var  , _ = _bn_collect(r'^batch_normalization(?:_\d+)?\.running_var$',  'moving_variance')

    bn_modules = [
        model.conv1.bn1, model.conv1.bn2, model.conv1.bn3,
        model.conv2.bn1, model.conv2.bn2, model.conv2.bn3,
        model.conv3.bn1, model.conv3.bn2, model.conv3.bn3,
    ]
    for i in range(4):
        bn_modules += [
            model.extra_pre[i].bn1, model.extra_pre[i].bn2, model.extra_pre[i].bn3,
            model.extra_post[i].bn1, model.extra_post[i].bn2, model.extra_post[i].bn3,
        ]
    assert len(bn_modules) == len(gamma) == len(beta) == len(mean) == len(var) == 33, "BN count mismatch"

    for kg, kb, km, kv, bn in zip(gamma, beta, mean, var, bn_modules):
        bn.weight.data.copy_(_as_tensor(torchish[kg], bn.weight))
        bn.bias.data.copy_(_as_tensor(torchish[kb], bn.bias))
        bn.running_mean.data.copy_(_as_tensor(torchish[km], bn.running_mean))
        bn.running_var.data.copy_(_as_tensor(torchish[kv], bn.running_var))

    # LayerNorms: 2 per block + final
    ln_w = sorted([k for k in torchish if re.match(r'^layer_normalization(?:_\d+)?\.weight$', k)],
                  key=lambda k: _suffix_idx(k.split('.')[0], 'layer_normalization') or 0)
    ln_b = sorted([k for k in torchish if re.match(r'^layer_normalization(?:_\d+)?\.bias$', k)],
                  key=lambda k: _suffix_idx(k.split('.')[0], 'layer_normalization') or 0)
    if not ln_w:  # TF fallback
        ln_w = sorted([k for k in torchish if re.match(r'^layer_normalization(?:_\d+)?/gamma:0$', k)],
                      key=lambda k: _suffix_idx(k.split('/')[0], 'layer_normalization') or 0)
        ln_b = sorted([k for k in torchish if re.match(r'^layer_normalization(?:_\d+)?/beta:0$', k)],
                      key=lambda k: _suffix_idx(k.split('/')[0], 'layer_normalization') or 0)

    ln_modules = []
    for i in range(4):
        ln_modules += [model.transformers[i].norm1, model.transformers[i].norm2]
    ln_modules += [model.norm]
    assert len(ln_modules) == len(ln_w) == len(ln_b) == 9, "LayerNorm count mismatch"

    for kw, kb, mod in zip(ln_w, ln_b, ln_modules):
        mod.weight.data.copy_(_as_tensor(torchish[kw], mod.weight))
        mod.bias.data.copy_(_as_tensor(torchish[kb], mod.bias))

    hk = _grab(
        torchish,
        "picker_S.weight",
        regex=r"picker_S/kernel:0",
        desc="picker_S kernel",
    )
    W = _coerce_conv1d(torchish[hk], model.head.conv.weight)
    model.head.conv.weight.data.copy_(_as_tensor(W, model.head.conv.weight))
    hb = _grab(
        torchish,
        "picker_S.bias",
        regex=r"picker_S/bias:0",
        desc="picker_S bias",
    )
    model.head.conv.bias.data.copy_(_as_tensor(torchish[hb], model.head.conv.bias))

    return model


def load_eqcct_model_s_weights(
    model: EQCCTModelS,
    *,
    h5_path=None,
    pickle_path=None,
):
    """
    Populate EQCCTModelS from a Keras S-branch H5 or legacy pickle dict.
    Exactly one of ``h5_path`` or ``pickle_path`` must be set.
    """
    if (h5_path is None) == (pickle_path is None):
        raise ValueError("Provide exactly one of h5_path or pickle_path")
    if h5_path is not None:
        torchish = flat_torchish_from_h5(h5_path)
        return load_torchish_into_eqcctS(model, torchish)
    raise NotImplementedError("pickle_path for EQCCTModelS is not wired here yet")


def load_torchish_into_eqcctP(model: EQCCTModelP, torchish: dict):
    """
    Load flat torchish tensors (see flat_torchish_from_h5) into EQCCTModelP.
    Expects a P checkpoint: 9 conv1d, 9 batch norm, 4 MHA, 8 dense (MLP), picker_P.
    """
    model.eval()

    # ----- PatchEncoder -----
    kW = _grab(
        torchish,
        "patch_encoder.linear.weight",
        "patch_encoder.projection.weight",
        regex=r"patch_encoder/dense/kernel:0",
        desc="PatchEncoder weight",
    )
    W = _coerce_linear(torchish[kW], model.encoder.projection.weight)
    model.encoder.projection.weight.data.copy_(_as_tensor(W, model.encoder.projection.weight))

    kB = _grab(
        torchish,
        "patch_encoder.linear.bias",
        "patch_encoder.projection.bias",
        regex=r"patch_encoder/dense/bias:0",
        desc="PatchEncoder bias",
    )
    model.encoder.projection.bias.data.copy_(_as_tensor(torchish[kB], model.encoder.projection.bias))

    kE = _grab(
        torchish,
        "patch_encoder.pos_embedding",
        "patch_encoder.position_embedding.weight",
        regex=r"patch_encoder/embedding/embeddings:0",
        desc="PatchEncoder pos-embedding",
    )
    model.encoder.position_embedding.weight.data.copy_(
        _as_tensor(torchish[kE], model.encoder.position_embedding.weight)
    )

    _keras_attn_name = {"q": "query", "k": "key", "v": "value"}

    def copy_mha_p(i, blk_prefix):
        attn = model.transformer[i].attn
        for proj in ("q", "k", "v"):
            kw = _grab(
                torchish,
                f"{blk_prefix}.{proj}_proj.weight",
                regex=rf"{blk_prefix}/{_keras_attn_name[proj]}/kernel:0",
                desc=f"{blk_prefix}.{proj} weight",
            )
            W = _coerce_mha_qkv_weight(torchish[kw], getattr(attn, proj).weight)
            getattr(attn, proj).weight.data.copy_(_as_tensor(W, getattr(attn, proj).weight))
            kb = _grab(
                torchish,
                f"{blk_prefix}.{proj}_proj.bias",
                regex=rf"{blk_prefix}/{_keras_attn_name[proj]}/bias:0",
                desc=f"{blk_prefix}.{proj} bias",
            )
            bb = _coerce_mha_bias(torchish[kb], getattr(attn, proj).bias)
            getattr(attn, proj).bias.data.copy_(_as_tensor(bb, getattr(attn, proj).bias))

        out_mod = getattr(attn, "o", None) or getattr(attn, "proj", None)
        if out_mod is None:
            raise AttributeError("Attention module needs .o or .proj Linear")
        kow = _grab(
            torchish,
            f"{blk_prefix}.out_proj.weight",
            regex=rf"{blk_prefix}/(output|attention_output)(?:_dense)?/kernel:0",
            desc=f"{blk_prefix}.out_proj weight",
        )
        W = _coerce_mha_out_weight(torchish[kow], out_mod.weight)
        out_mod.weight.data.copy_(_as_tensor(W, out_mod.weight))
        kob = _grab(
            torchish,
            f"{blk_prefix}.out_proj.bias",
            regex=rf"{blk_prefix}/(output|attention_output)(?:_dense)?/bias:0",
            desc=f"{blk_prefix}.out_proj bias",
        )
        ob = _coerce_mha_bias(torchish[kob], out_mod.bias)
        out_mod.bias.data.copy_(_as_tensor(ob, out_mod.bias))

    mha_names = ["multi_head_attention"] + [f"multi_head_attention_{i}" for i in range(1, 4)]
    for i, blk in enumerate(mha_names):
        copy_mha_p(i, blk)

    def copy_dense(base, lin):
        kW = _grab(torchish, f"{base}.weight", regex=rf"{base}/kernel:0", desc=f"{base} weight")
        kB = _grab(torchish, f"{base}.bias", regex=rf"{base}/bias:0", desc=f"{base} bias")
        W = _coerce_linear(torchish[kW], lin.weight)
        lin.weight.data.copy_(_as_tensor(W, lin.weight))
        lin.bias.data.copy_(_as_tensor(torchish[kB], lin.bias))

    for i in range(4):
        copy_dense(f"dense_{2 * i + 1}", model.transformer[i].mlp.fc1)
        copy_dense(f"dense_{2 * i + 2}", model.transformer[i].mlp.fc2)

    # ----- 9 conv + BN (front-end only for P) -----
    conv_pt = sorted(
        [k for k in torchish if re.match(r"^conv1d(?:_\d+)?\.weight$", k)],
        key=lambda k: _suffix_idx(k.split(".")[0], "conv1d") or 0,
    )
    conv_tf = sorted(
        [k for k in torchish if re.match(r"^conv1d(?:_\d+)?/kernel:0$", k)],
        key=lambda k: _suffix_idx(k.split("/")[0], "conv1d") or 0,
    )
    conv_keys = conv_pt or conv_tf
    is_tf = bool(conv_tf and not conv_pt)

    conv_modules = [
        model.conv1.conv1,
        model.conv1.conv2,
        model.conv1.conv3,
        model.conv2.conv1,
        model.conv2.conv2,
        model.conv2.conv3,
        model.conv3.conv1,
        model.conv3.conv2,
        model.conv3.conv3,
    ]
    assert len(conv_modules) == len(conv_keys) == 9, (
        f"P-model conv mismatch: modules={len(conv_modules)} keys={len(conv_keys)}"
    )

    for k, mod in zip(conv_keys, conv_modules):
        W = torchish[k]
        W = _coerce_conv1d(W, mod.weight) if is_tf else np.asarray(W)
        mod.weight.data.copy_(_as_tensor(W, mod.weight))
        bk = k.replace(".weight", ".bias") if not is_tf else k.replace("/kernel:0", "/bias:0")
        if bk in torchish:
            mod.bias.data.copy_(_as_tensor(torchish[bk], mod.bias))

    def _bn_collect(pt_pat, tf_name):
        pt = sorted(
            [k for k in torchish if re.match(pt_pat, k)],
            key=lambda k: _suffix_idx(k.split(".")[0], "batch_normalization") or 0,
        )
        if pt:
            return pt, False
        tf = sorted(
            [k for k in torchish if re.match(rf"^batch_normalization(?:_\d+)?/{tf_name}:0$", k)],
            key=lambda k: _suffix_idx(k.split("/")[0], "batch_normalization") or 0,
        )
        return tf, True

    gamma, _ = _bn_collect(r"^batch_normalization(?:_\d+)?\.weight$", "gamma")
    beta, _ = _bn_collect(r"^batch_normalization(?:_\d+)?\.bias$", "beta")
    mean, _ = _bn_collect(r"^batch_normalization(?:_\d+)?\.running_mean$", "moving_mean")
    var, _ = _bn_collect(r"^batch_normalization(?:_\d+)?\.running_var$", "moving_variance")

    bn_modules = [
        model.conv1.bn1,
        model.conv1.bn2,
        model.conv1.bn3,
        model.conv2.bn1,
        model.conv2.bn2,
        model.conv2.bn3,
        model.conv3.bn1,
        model.conv3.bn2,
        model.conv3.bn3,
    ]
    assert len(bn_modules) == len(gamma) == len(beta) == len(mean) == len(var) == 9, "P-model BN count mismatch"

    for kg, kb_, km, kv, bn in zip(gamma, beta, mean, var, bn_modules):
        bn.weight.data.copy_(_as_tensor(torchish[kg], bn.weight))
        bn.bias.data.copy_(_as_tensor(torchish[kb_], bn.bias))
        bn.running_mean.data.copy_(_as_tensor(torchish[km], bn.running_mean))
        bn.running_var.data.copy_(_as_tensor(torchish[kv], bn.running_var))

    ln_w = sorted(
        [k for k in torchish if re.match(r"^layer_normalization(?:_\d+)?\.weight$", k)],
        key=lambda k: _suffix_idx(k.split(".")[0], "layer_normalization") or 0,
    )
    ln_b = sorted(
        [k for k in torchish if re.match(r"^layer_normalization(?:_\d+)?\.bias$", k)],
        key=lambda k: _suffix_idx(k.split(".")[0], "layer_normalization") or 0,
    )
    if not ln_w:
        ln_w = sorted(
            [k for k in torchish if re.match(r"^layer_normalization(?:_\d+)?/gamma:0$", k)],
            key=lambda k: _suffix_idx(k.split("/")[0], "layer_normalization") or 0,
        )
        ln_b = sorted(
            [k for k in torchish if re.match(r"^layer_normalization(?:_\d+)?/beta:0$", k)],
            key=lambda k: _suffix_idx(k.split("/")[0], "layer_normalization") or 0,
        )

    ln_modules = []
    for i in range(4):
        ln_modules += [model.transformer[i].norm1, model.transformer[i].norm2]
    ln_modules += [model.norm]
    assert len(ln_modules) == len(ln_w) == len(ln_b) == 9, "P-model LayerNorm count mismatch"

    for kw, kb_, mod in zip(ln_w, ln_b, ln_modules):
        mod.weight.data.copy_(_as_tensor(torchish[kw], mod.weight))
        mod.bias.data.copy_(_as_tensor(torchish[kb_], mod.bias))

    hk = _grab(
        torchish,
        "picker_P.weight",
        regex=r"picker_P/kernel:0",
        desc="picker_P kernel",
    )
    W = _coerce_conv1d(torchish[hk], model.head.conv.weight)
    model.head.conv.weight.data.copy_(_as_tensor(W, model.head.conv.weight))
    hb = _grab(
        torchish,
        "picker_P.bias",
        regex=r"picker_P/bias:0",
        desc="picker_P bias",
    )
    model.head.conv.bias.data.copy_(_as_tensor(torchish[hb], model.head.conv.bias))

    return model


def load_eqcct_model_p_weights(
    model: EQCCTModelP,
    *,
    h5_path=None,
    pickle_path=None,
):
    """
    Populate EQCCTModelP from either a Keras H5 file or a legacy pickle dict
    (same format as transfer_weights_legacy.transfer_weights_p).
    Exactly one of h5_path or pickle_path must be set.
    """
    if (h5_path is None) == (pickle_path is None):
        raise ValueError("Provide exactly one of h5_path or pickle_path")
    if h5_path is not None:
        torchish = flat_torchish_from_h5(h5_path)
        return load_torchish_into_eqcctP(model, torchish)
    from eqcct_tf_pt_transfer.conversion.transfer_weights_legacy import transfer_weights_p

    with open(pickle_path, "rb") as f:
        w = pickle.load(f)
    return transfer_weights_p(w, model)