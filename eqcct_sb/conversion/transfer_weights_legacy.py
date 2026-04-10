import torch
import numpy as np
import pickle
# Layer names (mlp.fc1, attn.o) match this module layout, not predictor_pt.TransformerBlock.
from eqcct_sb.models.predictor_pt_p import EQCCTModelP, EQCCTModelS

def map_conv1d_weights(tf_weights):
    """Convert TF Conv1D weights to PyTorch format"""
    # TF: (kernel_width, in_channels, out_channels) -> PT: (out_channels, in_channels, kernel_width)
    kernel, bias = tf_weights
    kernel = np.transpose(kernel, (2, 1, 0))
    return kernel, bias

def map_batchnorm_weights(tf_weights):
    """Map TF BatchNorm weights to PyTorch format"""
    gamma, beta, moving_mean, moving_variance = tf_weights
    return gamma, beta, moving_mean, moving_variance

def map_dense_weights(tf_weights):
    """Convert TF Dense weights to PyTorch Linear format"""
    weights, bias = tf_weights
    weights = np.transpose(weights)
    return weights, bias

def map_layernorm_weights(tf_weights):
    """Map TF LayerNorm weights to PyTorch format"""
    gamma, beta = tf_weights
    return gamma, beta

def map_keras_mha_weights(tf_mha_weights):
    """
    Correctly maps weights from tf.keras.layers.MultiHeadAttention
    by reshaping the 3D weight tensors into 2D using standard C-style ordering.
    """
    # --- Q, K, V Projection Weights ---
    # TF Shape: (embed_dim, num_heads, key_dim) -> e.g., (40, 4, 40)
    # Target PT Shape: (num_heads * key_dim, embed_dim) -> e.g., (160, 40)
    q_w_tf, q_b_tf = tf_mha_weights[0:2]
    k_w_tf, k_b_tf = tf_mha_weights[2:4]
    v_w_tf, v_b_tf = tf_mha_weights[4:6]

    # Reshape using the default 'C' order, then transpose.
    q_w = q_w_tf.reshape(q_w_tf.shape[0], -1).T
    k_w = k_w_tf.reshape(k_w_tf.shape[0], -1).T
    v_w = v_w_tf.reshape(v_w_tf.shape[0], -1).T

    # Biases need to be flattened.
    q_b, k_b, v_b = q_b_tf.flatten(), k_b_tf.flatten(), v_b_tf.flatten()

    # --- Output Projection Weights ---
    # TF Shape: (num_heads, key_dim, embed_dim) -> e.g., (4, 40, 40)
    # Target PT Shape: (embed_dim, num_heads * key_dim) -> e.g., (40, 160)
    o_w_tf, o_b_tf = tf_mha_weights[6:8]
    
    # Reshape and transpose.
    o_w = o_w_tf.reshape(-1, o_w_tf.shape[-1]).T
    o_b = o_b_tf

    return {
        'q.weight': torch.from_numpy(q_w), 'q.bias': torch.from_numpy(q_b),
        'k.weight': torch.from_numpy(k_w), 'k.bias': torch.from_numpy(k_b),
        'v.weight': torch.from_numpy(v_w), 'v.bias': torch.from_numpy(v_b),
        'o.weight': torch.from_numpy(o_w), 'o.bias': torch.from_numpy(o_b),
    }


def get_layer_indices(weights_dict, layer_type):
    """Extract layer indices from weight dictionary"""
    indices = []
    for key in weights_dict.keys():
        if layer_type in key:
            # Extract number from layer name
            import re
            match = re.search(r'(\d+)$', key)
            if match:
                indices.append(int(match.group(1)))
    return sorted(indices)


def get_tf_layer_names(weights_dict, base_name):
    """
    Extracts and sorts layer names from the TensorFlow weights dictionary.
    This version handles base names without numerical suffixes.
    """
    names = [k for k in weights_dict.keys() if k.startswith(base_name)]
    
    def sort_key(layer_name):
        parts = layer_name.split('_')
        # Check if the last part is a number
        if parts[-1].isdigit():
            return int(parts[-1])
        # If not, it's the base layer (e.g., 'conv1d'). Sort it first.
        return 0

    return sorted(names, key=sort_key)


def transfer_weights_p(tf_weights, pt_model):
    """Transfers weights from the TF P-model to the PyTorch P-model."""
    print("🚀 Starting weight transfer for P-Model...")
    pt_state_dict = pt_model.state_dict()

    # Get sorted TF layer names
    conv_names = get_tf_layer_names(tf_weights, 'conv1d')
    bn_names = get_tf_layer_names(tf_weights, 'batch_normalization')
    ln_names = get_tf_layer_names(tf_weights, 'layer_normalization')
    mha_names = get_tf_layer_names(tf_weights, 'multi_head_attention')
    dense_names = get_tf_layer_names(tf_weights, 'dense')

    # --- 1. Initial Conv Blocks ---
    # 3 blocks, 3 convs/bns each = 9 total
    tf_conv_idx, tf_bn_idx = 0, 0
    for block_num in range(1, 4):  # For conv1, conv2, conv3
        for layer_num in range(1, 4): # For conv1, conv2, conv3 inside the block
            # Map Conv1D
            pt_conv_key = f'conv{block_num}.conv{layer_num}'
            kernel, bias = map_conv1d_weights(tf_weights[conv_names[tf_conv_idx]])
            pt_state_dict[f'{pt_conv_key}.weight'] = torch.from_numpy(kernel)
            pt_state_dict[f'{pt_conv_key}.bias'] = torch.from_numpy(bias)
            
            # Map BatchNorm
            pt_bn_key = f'conv{block_num}.bn{layer_num}'
            gamma, beta, mean, var = map_batchnorm_weights(tf_weights[bn_names[tf_bn_idx]])
            pt_state_dict[f'{pt_bn_key}.weight'] = torch.from_numpy(gamma)
            pt_state_dict[f'{pt_bn_key}.bias'] = torch.from_numpy(beta)
            pt_state_dict[f'{pt_bn_key}.running_mean'] = torch.from_numpy(mean)
            pt_state_dict[f'{pt_bn_key}.running_var'] = torch.from_numpy(var)

            tf_conv_idx += 1
            tf_bn_idx += 1
            
    print("✅ Mapped initial Conv blocks.")

    # --- 2. Patch Encoder ---
    pe_name = get_tf_layer_names(tf_weights, 'patch_encoder')[0]
    pe_proj_w, pe_proj_b = map_dense_weights(tf_weights[pe_name][0:2])
    pe_pos_emb = tf_weights[pe_name][2]
    pt_state_dict['encoder.projection.weight'] = torch.from_numpy(pe_proj_w)
    pt_state_dict['encoder.projection.bias'] = torch.from_numpy(pe_proj_b)
    pt_state_dict['encoder.position_embedding.weight'] = torch.from_numpy(pe_pos_emb)
    print("✅ Mapped Patch Encoder.")

    # --- 3. Transformer Blocks ---
    tf_ln_idx, tf_dense_idx = 0, 0
    for i in range(4): # 4 transformer blocks
        # LayerNorm 1
        ln1_w, ln1_b = map_layernorm_weights(tf_weights[ln_names[tf_ln_idx]])
        pt_state_dict[f'transformer.{i}.norm1.weight'] = torch.from_numpy(ln1_w)
        pt_state_dict[f'transformer.{i}.norm1.bias'] = torch.from_numpy(ln1_b)
        
        # MHA
        mha_pt_weights = map_keras_mha_weights(tf_weights[mha_names[i]])
        for key, val in mha_pt_weights.items():
            pt_state_dict[f'transformer.{i}.attn.{key}'] = val
            
        # LayerNorm 2
        ln2_w, ln2_b = map_layernorm_weights(tf_weights[ln_names[tf_ln_idx+1]])
        pt_state_dict[f'transformer.{i}.norm2.weight'] = torch.from_numpy(ln2_w)
        pt_state_dict[f'transformer.{i}.norm2.bias'] = torch.from_numpy(ln2_b)
        
        # MLP (2 Dense layers)
        mlp_fc1_w, mlp_fc1_b = map_dense_weights(tf_weights[dense_names[tf_dense_idx]])
        pt_state_dict[f'transformer.{i}.mlp.fc1.weight'] = torch.from_numpy(mlp_fc1_w)
        pt_state_dict[f'transformer.{i}.mlp.fc1.bias'] = torch.from_numpy(mlp_fc1_b)

        mlp_fc2_w, mlp_fc2_b = map_dense_weights(tf_weights[dense_names[tf_dense_idx+1]])
        pt_state_dict[f'transformer.{i}.mlp.fc2.weight'] = torch.from_numpy(mlp_fc2_w)
        pt_state_dict[f'transformer.{i}.mlp.fc2.bias'] = torch.from_numpy(mlp_fc2_b)

        tf_ln_idx += 2
        tf_dense_idx += 2
        
    print(f"✅ Mapped {len(mha_names)} Transformer blocks.")
    
    # --- 4. Final LayerNorm and Head ---
    # Final representation LayerNorm
    final_ln_w, final_ln_b = map_layernorm_weights(tf_weights[ln_names[-1]])
    pt_state_dict['norm.weight'] = torch.from_numpy(final_ln_w)
    pt_state_dict['norm.bias'] = torch.from_numpy(final_ln_b)

    # Final output Conv1D head ('picker_P')
    head_w, head_b = map_conv1d_weights(tf_weights['picker_P'])
    pt_state_dict['head.conv.weight'] = torch.from_numpy(head_w)
    pt_state_dict['head.conv.bias'] = torch.from_numpy(head_b)
    print("✅ Mapped final LayerNorm and Head.")

    pt_model.load_state_dict(pt_state_dict)
    print("🎉 P-Model weight transfer complete!")
    return pt_model


# --- S-Model Transfer Function ---
def transfer_weights_s_direct(tf_weights, pt_model):
    """
    Transfers weights from the TF S-model to the PyTorch S-model
    using a DIRECT, NON-FUSED mapping (like transfer_weights_p).
    """
    print("🚀 Starting DIRECT weight transfer for S-Model...")
    pt_state_dict = pt_model.state_dict()

    # Get sorted TF layer names
    conv_names = get_tf_layer_names(tf_weights, 'conv1d')
    bn_names = get_tf_layer_names(tf_weights, 'batch_normalization')
    ln_names = get_tf_layer_names(tf_weights, 'layer_normalization')
    mha_names = get_tf_layer_names(tf_weights, 'multi_head_attention')
    dense_names = get_tf_layer_names(tf_weights, 'dense')

    tf_conv_idx, tf_bn_idx = 0, 0

    # List all modules that are ConvF1Blocks
    all_convf1_blocks = [
        "conv1", "conv2", "conv3",
        "extra_pre.0", "extra_post.0",
        "extra_pre.1", "extra_post.1",
        "extra_pre.2", "extra_post.2",
        "extra_pre.3", "extra_post.3",
    ]

    for block_name in all_convf1_blocks:
        for layer_num in range(1, 4): # Each block has 3 conv/bn pairs
            # Map Conv1D
            pt_conv_key = f"{block_name}.conv{layer_num}"
            kernel, bias = map_conv1d_weights(tf_weights[conv_names[tf_conv_idx]])
            pt_state_dict[f"{pt_conv_key}.weight"] = torch.from_numpy(kernel)
            pt_state_dict[f"{pt_conv_key}.bias"] = torch.from_numpy(bias)
            
            # Map BatchNorm
            pt_bn_key = f"{block_name}.bn{layer_num}"
            gamma, beta, mean, var = map_batchnorm_weights(tf_weights[bn_names[tf_bn_idx]])
            pt_state_dict[f"{pt_bn_key}.weight"] = torch.from_numpy(gamma)
            pt_state_dict[f"{pt_bn_key}.bias"] = torch.from_numpy(beta)
            pt_state_dict[f"{pt_bn_key}.running_mean"] = torch.from_numpy(mean)
            pt_state_dict[f"{pt_bn_key}.running_var"] = torch.from_numpy(var)

            tf_conv_idx += 1
            tf_bn_idx += 1
    print("✅ Mapped all Conv/BN blocks directly.")

    # --- Load remaining layers (PatchEncoder, Transformers, Head) ---
    # This part is the same as the fused function
    
    pe_name = get_tf_layer_names(tf_weights, 'patch_encoder')[0]
    pe_proj_w, pe_proj_b = map_dense_weights(tf_weights[pe_name][0:2])
    pe_pos_emb = tf_weights[pe_name][2]
    pt_state_dict['encoder.projection.weight'] = torch.from_numpy(pe_proj_w)
    pt_state_dict['encoder.projection.bias'] = torch.from_numpy(pe_proj_b)
    pt_state_dict['encoder.position_embedding.weight'] = torch.from_numpy(pe_pos_emb)
    print("✅ Mapped Patch Encoder.")

    tf_ln_idx, tf_dense_idx = 0, 0
    for i in range(4):
        ln1_w, ln1_b = map_layernorm_weights(tf_weights[ln_names[tf_ln_idx]])
        pt_state_dict[f'transformers.{i}.norm1.weight'], pt_state_dict[f'transformers.{i}.norm1.bias'] = torch.from_numpy(ln1_w), torch.from_numpy(ln1_b)
        
        mha_pt_weights = map_keras_mha_weights(tf_weights[mha_names[i]])
        for key, val in mha_pt_weights.items():
            pt_state_dict[f'transformers.{i}.attn.{key}'] = val
        
        ln2_w, ln2_b = map_layernorm_weights(tf_weights[ln_names[tf_ln_idx+1]])
        pt_state_dict[f'transformers.{i}.norm2.weight'], pt_state_dict[f'transformers.{i}.norm2.bias'] = torch.from_numpy(ln2_w), torch.from_numpy(ln2_b)
        
        mlp_fc1_w, mlp_fc1_b = map_dense_weights(tf_weights[dense_names[tf_dense_idx]])
        pt_state_dict[f'transformers.{i}.mlp.fc1.weight'], pt_state_dict[f'transformers.{i}.mlp.fc1.bias'] = torch.from_numpy(mlp_fc1_w), torch.from_numpy(mlp_fc1_b)
        
        mlp_fc2_w, mlp_fc2_b = map_dense_weights(tf_weights[dense_names[tf_dense_idx+1]])
        pt_state_dict[f'transformers.{i}.mlp.fc2.weight'], pt_state_dict[f'transformers.{i}.mlp.fc2.bias'] = torch.from_numpy(mlp_fc2_w), torch.from_numpy(mlp_fc2_b)
        
        tf_ln_idx += 2
        tf_dense_idx += 2
    print(f"✅ Mapped Transformer sub-layers.")

    final_ln_w, final_ln_b = map_layernorm_weights(tf_weights[ln_names[-1]])
    pt_state_dict['norm.weight'], pt_state_dict['norm.bias'] = torch.from_numpy(final_ln_w), torch.from_numpy(final_ln_b)
    
    head_w, head_b = map_conv1d_weights(tf_weights['picker_S'])
    pt_state_dict['head.conv.weight'] = torch.from_numpy(head_w)
    pt_state_dict['head.conv.bias'] = torch.from_numpy(head_b)
    print("✅ Mapped final LayerNorm and Head.")

    pt_model.load_state_dict(pt_state_dict)
    print("🎉 S-Model DIRECT weight transfer complete!")
    return pt_model


def main():
    """Main conversion function using pickle files"""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    model_dir = repo_root / "ModelPS"

    print("Loading weights from pickle files...")
    with open(model_dir / "Sweights.pkl", "rb") as f:
        Sweights = pickle.load(f)
    print(f"Loaded {len(Sweights)} S model layers")

    modelS_pt_direct = EQCCTModelS()

    print("\nTransferring weights to PyTorch S-Model (Direct Method)...")
    modelS_pt_direct = transfer_weights_s_direct(Sweights, modelS_pt_direct)

    torch.save(modelS_pt_direct.state_dict(), repo_root / "modelS_pytorch_direct.pth")

    print("\nWeight transfer complete!")