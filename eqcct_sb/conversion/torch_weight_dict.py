import numpy as np
from eqcct_sb.conversion.catalog import assert_and_print

def tf_dense_to_torch(w, b=None): 
    # TF Dense: (inF, outF) -> Torch Linear: (outF, inF)
    # w = weight tensor (in, out)
    # b = bias vector from TF layer (1D array)
    wt = w.T.copy() # Weights transposed 
    bt = None if b is None else b.copy()
    return wt, bt 

def tf_conv1d_to_torch(w, b=None): 
    # TF Conv1D: (kw, inC, outC) -> Torch Conv1d: (outC, inC, kw)
    wt = np.transpose(w, (2, 1, 0)).copy()
    bt = None if b is None else b.copy()
    return wt, bt 

def _as_2d_qkv(w):
    # Keras MHA q/k/v kernel often (E, H, D). Make (H*D, E) for a Linear mapping E->H*D
    if w.ndim == 3:               # (E, H, D)
        E, H, D = w.shape
        return w.reshape(E, H*D).T.copy(), (E, H, D)
    elif w.ndim == 2:             # (E, H*D) or (H*D, E)
        # prefer (H*D, E) as output; if already that shape, just copy
        return (w if w.shape[0] > w.shape[1] else w.T).copy(), None
    else:
        raise ValueError(f"Unexpected qkv ndim: {w.ndim}")

def _as_2d_out(w, embed_dim):
    # Keras MHA out kernel often (H, D, E). Make (E, H*D) for a Linear mapping H*D->E
    if w.ndim == 3:               # (H, D, E)
        H, D, E = w.shape
        assert E == embed_dim
        return w.reshape(H*D, E).T.copy()
    elif w.ndim == 2:             # either (E, H*D) or (H*D, E)
        if w.shape[0] == embed_dim:    # already (E, H*D)
            return w.copy()
        elif w.shape[1] == embed_dim:  # (H*D, E) -> (E, H*D)
            return w.T.copy()
        else:
            raise ValueError(f"Out kernel 2D but neither axis equals embed_dim={embed_dim}: {w.shape}")
    else:
        raise ValueError(f"Unexpected out ndim: {w.ndim}")
    
def tf_mha_block_to_torch(parts):
    """
    parts: dict like {'query':{'kernel','bias'}, 'key':{...}, 'value':{...}, 'output':{...}}
    Returns 2D weights suitable for Linear layers:
      Wq/Wk/Wv: (H*D, E) ; bq/bk/bv: (H*D,)
      Wout: (E, H*D) ; bout: (E,)
    """
    # --- Q/K/V ---
    Wq, q_shape = _as_2d_qkv(parts['query']['kernel'])
    Wk, _       = _as_2d_qkv(parts['key']['kernel'])
    Wv, _       = _as_2d_qkv(parts['value']['kernel'])

    bq = parts['query'].get('bias');  bq = None if bq is None else bq.reshape(-1).copy()
    bk = parts['key'].get('bias');    bk = None if bk is None else bk.reshape(-1).copy()
    bv = parts['value'].get('bias');  bv = None if bv is None else bv.reshape(-1).copy()

    # embed_dim from q kernel (first axis when 3D or the smaller axis when 2D)
    if q_shape is not None:
        E = q_shape[0]
    else:
        E = min(Wq.shape[0], Wq.shape[1])  # conservative guess; typically 40 here

    # --- output proj ---
    ok = parts.get('output', {}).get('kernel')
    bout = parts.get('output', {}).get('bias')
    Wout = None if ok is None else _as_2d_out(ok, E)
    bout = None if bout is None else bout.reshape(-1).copy()

    return dict(Wq=Wq, bq=bq, Wk=Wk, bk=bk, Wv=Wv, bv=bv, Wout=Wout, bout=bout, embed_dim=E)

# Build Torch-ish state dict 
def build_torchish_state(flat):
    """
    Uses assert_and_print()'s parasing to: 
    * transpose Dense -> Linear
    * permute Conv1D weights TF -> Torch
    * reshape MHA (q/k/v/out) to 2D 
    Returns a dict {name: np.array}
    """
    cat = assert_and_print(flat)
    out = {}

    # PatchEncoder 
    pe_w = flat[cat['patch_encoder']['dense_w']]
    pe_b = flat[cat['patch_encoder']['dense_b']]
    pe_e = flat[cat['patch_encoder']['pos']]
    W, B = tf_dense_to_torch(pe_w, pe_b)
    out['patch_encoder.linear.weights'] = W 
    out['patch_encoder.linear.bias'] = B
    out['patch_encoder.linear.pos_embedding'] = pe_e.copy() 

    # Picker S head (Conv1d)
    if cat['heads']['picker_S']:
        W, B = tf_conv1d_to_torch(flat['picker_S/kernel:0'], flat['picker_S/bias:0'])
        out['picker_S.weight'] = W 
        out['picker_S.bias']   = B 

    # Conv1D blocks 
    for k in cat['conv1d_kernels']: 
        base = k[:-len('/kernel:0')]
        W, B = tf_conv1d_to_torch(flat[k], flat.get(base + '/bias:0'))
        out[f'{base}.weight'] = W 
        if B is not None: 
            out[f'{base}.bias'] = B 

    # BatchNorm (names follow Torch: weight/bias/running_mean/running_var) 
    for k in cat['bn']['gamma']: 
        base = k[:-len('/gamma:0')]
        out[f'{base}.weight'] = flat[k].copy()
    for k in cat['bn']['beta']: 
        base = k[:-len('/beta:0')]
        out[f'{base}.bias'] = flat[k].copy()
    for k in cat['bn']['mean']: 
        base = k[:-len('/moving_mean:0')]
        out[f'{base}.running_mean'] = flat[k].copy()
    for k in cat['bn']['var']: 
        base = k[:-len('/moving_variance:0')]
        out[f'{base}.running_var'] = flat[k].copy()

    # Dense Layers (MLP), excluding PatchEncoder 
    for k in cat['dense']['kernels']: 
        if k.startswith('patch_encoder/'): continue 
        base = k[:-len('/kernel:0')]
        W, _ = tf_dense_to_torch(flat[k])
        out[f'{base}.weight'] = W
    for k in cat['dense']['biases']: 
        if k.startswith('patch_encoder/'): continue 
        base = k[:-len('/bias:0')]
        out[f'{base}.bias'] = flat[k].copy()
    
    # LayerNorm 
    for k in cat['layernorm']['gamma']:
        base = k[:-len('/gamma:0')]
        out[f'{base}.weight'] = flat[k].copy()
    for k in cat['layernorm']['beta']:
        base = k[:-len('/beta:0')]
        out[f'{base}.bias'] = flat[k].copy()

    # MHA blocks
    for blk, parts in cat['mha'].items(): 
        pack = tf_mha_block_to_torch(parts)
        out[f'{blk}.q_proj.weight'] = pack['Wq'] # (H*D, E)
        if pack['bq'] is not None: 
            out[f'{blk}.q_proj.bias'] = pack['bq']
        out[f'{blk}.k_proj.weight'] = pack['Wk']
        if pack['bk'] is not None: 
            out[f'{blk}.k_proj.bias'] = pack['bk']
        out[f'{blk}.v_proj.weight'] = pack['Wv']
        if pack['bv'] is not None: 
            out[f'{blk}.v_proj.bias'] = pack['bv']
        if pack['Wout'] is not None: 
            out[f'{blk}.out_proj.weight'] = pack['Wout'] # (E, H*D)
        if pack['bout'] is not None: 
            out[f'{blk}.out_proj.bias'] = pack['bout']
            
    return out
