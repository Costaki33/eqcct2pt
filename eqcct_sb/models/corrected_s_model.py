import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from eqcct_sb.models.predictor_pt_fixed import (
    ConvF1Block, Patches, PatchEncoder, StochasticDepth, 
    TransformerMLP, OutputHead, KerasMHA,
    transformer_layers, projection_dim, num_heads, 
    stochastic_depth_rate, patch_size, num_patches, patch_dim
)

class EQCCTModelS_Corrected(nn.Module):
    """
    Corrected S Model that exactly matches TensorFlow's flow.
    The key insight is that in TF, the pre-conv modifies encoded_patches 
    BEFORE it's used as the identity for the skip connection.
    """
    def __init__(self):
        super().__init__()
        # Initial conv blocks
        self.conv1 = ConvF1Block(3, 10)
        self.conv2 = ConvF1Block(10, 20)
        self.conv3 = ConvF1Block(20, 40)
        
        # Patch processing
        self.patch = Patches(patch_size)
        self.encoder = PatchEncoder(num_patches, projection_dim, patch_dim)
        
        # Transformer components for each layer
        self.pre_convs = nn.ModuleList([ConvF1Block(40, 40) for _ in range(transformer_layers)])
        self.norm1s = nn.ModuleList([nn.LayerNorm(projection_dim, eps=1e-6) for _ in range(transformer_layers)])
        self.attns = nn.ModuleList([KerasMHA(embed_dim=projection_dim, num_heads=num_heads, key_dim=40) 
                                   for _ in range(transformer_layers)])
        self.post_convs = nn.ModuleList([ConvF1Block(40, 40) for _ in range(transformer_layers)])
        self.drop_path1s = nn.ModuleList([StochasticDepth(stochastic_depth_rate * (i / transformer_layers)) 
                                         for i in range(transformer_layers)])
        
        self.norm2s = nn.ModuleList([nn.LayerNorm(projection_dim, eps=1e-6) for _ in range(transformer_layers)])
        self.mlps = nn.ModuleList([TransformerMLP(projection_dim, dropout_rate=0.1) 
                                  for _ in range(transformer_layers)])
        self.drop_path2s = nn.ModuleList([StochasticDepth(stochastic_depth_rate * (i / transformer_layers)) 
                                         for i in range(transformer_layers)])
        
        # Output processing
        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)
        self.head = OutputHead(in_channels=1)

    def forward(self, x):
        # x shape: (B, 6000, 3)
        x = x.transpose(1, 2)  # (B, 3, 6000)
        
        # Initial convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Reshape for patching
        x = x.unsqueeze(2).permute(0, 3, 2, 1)  # (B, 6000, 1, 40)
        
        # Create patches and encode
        x = self.patch(x)    # (B, 150, 1600)
        encoded_patches = self.encoder(x)  # (B, 150, 40)
        
        # Transformer blocks - matching TF exactly
        for i in range(transformer_layers):
            # 1. Apply pre-conv to encoded_patches (this modifies it!)
            encoded_patches = encoded_patches.transpose(1, 2)  # (B, 40, 150)
            encoded_patches = self.pre_convs[i](encoded_patches)
            encoded_patches = encoded_patches.transpose(1, 2)  # (B, 150, 40)
            
            # 2. Now encoded_patches has been updated by pre-conv
            # Use it for normalization and as identity
            x1 = self.norm1s[i](encoded_patches)
            
            # 3. Multi-head attention
            attention_output = self.attns[i](x1)
            
            # 4. Apply post-conv to attention output
            attention_output = attention_output.transpose(1, 2)  # (B, 40, 150)
            attention_output = self.post_convs[i](attention_output)
            attention_output = attention_output.transpose(1, 2)  # (B, 150, 40)
            
            # 5. Skip connection 1 (using the pre-conv modified encoded_patches)
            attention_output = self.drop_path1s[i](attention_output)
            x2 = encoded_patches + attention_output
            
            # 6. Layer norm 2
            x3 = self.norm2s[i](x2)
            
            # 7. MLP
            x3 = self.mlps[i](x3)
            
            # 8. Skip connection 2
            x3 = self.drop_path2s[i](x3)
            encoded_patches = x2 + x3  # Update encoded_patches for next iteration
        
        # Final processing
        representation = self.norm(encoded_patches)  # (B, 150, 40)
        
        # Reshape to original sequence length
        representation = representation.reshape(representation.size(0), 6000, 1)  # (B, 6000, 1)
        
        # Apply output head
        return self.head(representation)  # (B, 6000, 1)


def transfer_weights_s_corrected(tf_weights, pt_model):
    """Transfer weights to the corrected S model."""
    print("🚀 Starting weight transfer for Corrected S-Model...")
    pt_state_dict = pt_model.state_dict()
    
    # Get sorted TF layer names
    from loader import get_tf_layer_names, map_conv1d_weights, map_batchnorm_weights, map_layernorm_weights, map_keras_mha_weights, map_dense_weights
    
    conv_names = get_tf_layer_names(tf_weights, 'conv1d')
    bn_names = get_tf_layer_names(tf_weights, 'batch_normalization')
    ln_names = get_tf_layer_names(tf_weights, 'layer_normalization')
    mha_names = get_tf_layer_names(tf_weights, 'multi_head_attention')
    dense_names = get_tf_layer_names(tf_weights, 'dense')
    
    tf_conv_idx, tf_bn_idx, tf_ln_idx, tf_dense_idx = 0, 0, 0, 0
    
    # 1. Initial Conv Blocks (same as before)
    for block_num in range(1, 4):
        for layer_num in range(1, 4):
            pt_conv_key = f'conv{block_num}.conv{layer_num}'
            kernel, bias = map_conv1d_weights(tf_weights[conv_names[tf_conv_idx]])
            pt_state_dict[f'{pt_conv_key}.weight'] = torch.from_numpy(kernel)
            pt_state_dict[f'{pt_conv_key}.bias'] = torch.from_numpy(bias)
            
            pt_bn_key = f'conv{block_num}.bn{layer_num}'
            gamma, beta, mean, var = map_batchnorm_weights(tf_weights[bn_names[tf_bn_idx]])
            pt_state_dict[f'{pt_bn_key}.weight'] = torch.from_numpy(gamma)
            pt_state_dict[f'{pt_bn_key}.bias'] = torch.from_numpy(beta)
            pt_state_dict[f'{pt_bn_key}.running_mean'] = torch.from_numpy(mean)
            pt_state_dict[f'{pt_bn_key}.running_var'] = torch.from_numpy(var)
            
            tf_conv_idx += 1
            tf_bn_idx += 1
    
    print("✅ Mapped initial Conv blocks")
    
    # 2. Patch Encoder
    pe_name = get_tf_layer_names(tf_weights, 'patch_encoder')[0]
    pe_proj_w, pe_proj_b = map_dense_weights(tf_weights[pe_name][0:2])
    pe_pos_emb = tf_weights[pe_name][2]
    pt_state_dict['encoder.projection.weight'] = torch.from_numpy(pe_proj_w)
    pt_state_dict['encoder.projection.bias'] = torch.from_numpy(pe_proj_b)
    pt_state_dict['encoder.position_embedding.weight'] = torch.from_numpy(pe_pos_emb)
    
    print("✅ Mapped Patch Encoder")
    
    # 3. Transformer Blocks
    for i in range(4):
        # Pre-conv block
        for layer_num in range(1, 4):
            pt_conv_key = f'pre_convs.{i}.conv{layer_num}'
            kernel, bias = map_conv1d_weights(tf_weights[conv_names[tf_conv_idx]])
            pt_state_dict[f'{pt_conv_key}.weight'] = torch.from_numpy(kernel)
            pt_state_dict[f'{pt_conv_key}.bias'] = torch.from_numpy(bias)
            
            pt_bn_key = f'pre_convs.{i}.bn{layer_num}'
            gamma, beta, mean, var = map_batchnorm_weights(tf_weights[bn_names[tf_bn_idx]])
            pt_state_dict[f'{pt_bn_key}.weight'] = torch.from_numpy(gamma)
            pt_state_dict[f'{pt_bn_key}.bias'] = torch.from_numpy(beta)
            pt_state_dict[f'{pt_bn_key}.running_mean'] = torch.from_numpy(mean)
            pt_state_dict[f'{pt_bn_key}.running_var'] = torch.from_numpy(var)
            
            tf_conv_idx += 1
            tf_bn_idx += 1
        
        # LayerNorm 1
        ln1_w, ln1_b = map_layernorm_weights(tf_weights[ln_names[tf_ln_idx]])
        pt_state_dict[f'norm1s.{i}.weight'] = torch.from_numpy(ln1_w)
        pt_state_dict[f'norm1s.{i}.bias'] = torch.from_numpy(ln1_b)
        
        # Multi-head attention
        mha_pt_weights = map_keras_mha_weights(tf_weights[mha_names[i]])
        for key, val in mha_pt_weights.items():
            pt_state_dict[f'attns.{i}.{key}'] = val
        
        # Post-conv block
        for layer_num in range(1, 4):
            pt_conv_key = f'post_convs.{i}.conv{layer_num}'
            kernel, bias = map_conv1d_weights(tf_weights[conv_names[tf_conv_idx]])
            pt_state_dict[f'{pt_conv_key}.weight'] = torch.from_numpy(kernel)
            pt_state_dict[f'{pt_conv_key}.bias'] = torch.from_numpy(bias)
            
            pt_bn_key = f'post_convs.{i}.bn{layer_num}'
            gamma, beta, mean, var = map_batchnorm_weights(tf_weights[bn_names[tf_bn_idx]])
            pt_state_dict[f'{pt_bn_key}.weight'] = torch.from_numpy(gamma)
            pt_state_dict[f'{pt_bn_key}.bias'] = torch.from_numpy(beta)
            pt_state_dict[f'{pt_bn_key}.running_mean'] = torch.from_numpy(mean)
            pt_state_dict[f'{pt_bn_key}.running_var'] = torch.from_numpy(var)
            
            tf_conv_idx += 1
            tf_bn_idx += 1
        
        # LayerNorm 2 and MLP
        ln2_w, ln2_b = map_layernorm_weights(tf_weights[ln_names[tf_ln_idx+1]])
        pt_state_dict[f'norm2s.{i}.weight'] = torch.from_numpy(ln2_w)
        pt_state_dict[f'norm2s.{i}.bias'] = torch.from_numpy(ln2_b)
        
        mlp_fc1_w, mlp_fc1_b = map_dense_weights(tf_weights[dense_names[tf_dense_idx]])
        pt_state_dict[f'mlps.{i}.fc1.weight'] = torch.from_numpy(mlp_fc1_w)
        pt_state_dict[f'mlps.{i}.fc1.bias'] = torch.from_numpy(mlp_fc1_b)
        
        mlp_fc2_w, mlp_fc2_b = map_dense_weights(tf_weights[dense_names[tf_dense_idx+1]])
        pt_state_dict[f'mlps.{i}.fc2.weight'] = torch.from_numpy(mlp_fc2_w)
        pt_state_dict[f'mlps.{i}.fc2.bias'] = torch.from_numpy(mlp_fc2_b)
        
        tf_ln_idx += 2
        tf_dense_idx += 2
    
    print(f"✅ Mapped {len(mha_names)} Transformer blocks")
    
    # 4. Final LayerNorm and Head
    final_ln_w, final_ln_b = map_layernorm_weights(tf_weights[ln_names[-1]])
    pt_state_dict['norm.weight'] = torch.from_numpy(final_ln_w)
    pt_state_dict['norm.bias'] = torch.from_numpy(final_ln_b)
    
    head_w, head_b = map_conv1d_weights(tf_weights['picker_S'])
    pt_state_dict['head.conv.weight'] = torch.from_numpy(head_w)
    pt_state_dict['head.conv.bias'] = torch.from_numpy(head_b)
    
    print("✅ Mapped final LayerNorm and Head")
    
    pt_model.load_state_dict(pt_state_dict)
    print("🎉 Corrected S-Model weight transfer complete!")
    return pt_model


if __name__ == "__main__":
    import pickle

    from eqcct_sb.paths import MODELPS_DIR
    from eqcct_sb.reference import predictor_tf

    mp = MODELPS_DIR

    # Load weights
    with open(mp / "Sweights.pkl", 'rb') as f:
        Sweights = pickle.load(f)
    
    # Create and transfer to corrected model
    modelS_corrected = EQCCTModelS_Corrected()
    modelS_corrected = transfer_weights_s_corrected(Sweights, modelS_corrected)
    
    # Save the corrected model
    torch.save(modelS_corrected.state_dict(), 'modelS_corrected.pth')
    print("\n✅ Corrected model saved as 'modelS_corrected.pth'")
    
    # Test the corrected model
    print("\n=== Testing Corrected S Model ===")
    _, modelS_tf = predictor_tf.load_eqcct_model(
        str(mp / 'test_trainer_024.h5'), str(mp / 'test_trainer_021.h5'))
    
    np.random.seed(42)
    test_input = np.random.rand(1, 6000, 3).astype('float32')
    
    # TF output
    tf_output = modelS_tf(test_input, training=False).numpy()
    
    # PT output
    modelS_corrected.eval()
    with torch.no_grad():
        pt_input = torch.from_numpy(test_input)
        pt_output = modelS_corrected(pt_input).numpy()
    
    print(f"TF output: mean={tf_output.mean():.6f}, std={tf_output.std():.6f}, max={tf_output.max():.6f}")
    print(f"PT output: mean={pt_output.mean():.6f}, std={pt_output.std():.6f}, max={pt_output.max():.6f}")
    
    # Check specific values
    print(f"\nFirst 10 TF values: {tf_output[0, :10, 0]}")
    print(f"First 10 PT values: {pt_output[0, :10, 0]}")
    
    # Find peaks
    tf_peak_idx = np.argmax(tf_output)
    pt_peak_idx = np.argmax(pt_output)
    print(f"\nTF peak at index {tf_peak_idx}: value={tf_output.flat[tf_peak_idx]:.6f}")
    print(f"PT peak at index {pt_peak_idx}: value={pt_output.flat[pt_peak_idx]:.6f}")
    
    try:
        np.testing.assert_allclose(tf_output, pt_output, rtol=1e-3, atol=1e-3)
        print("\n✅ SUCCESS: The outputs match!")
    except AssertionError:
        diff = np.abs(tf_output - pt_output)
        print(f"\n⚠️  Differences remain: max={diff.max():.6f}, mean={diff.mean():.6f}")
        rel_diff = diff / (np.abs(tf_output) + 1e-8)
        print(f"Max relative difference: {rel_diff.max():.2f}")