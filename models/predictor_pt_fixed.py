import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Model parameters (matching TensorFlow)
w1 = 6000
w2 = 3
drop_rate = 0.2
stochastic_depth_rate = 0.1
positional_emb = False
conv_layers = 4
num_classes = 1
input_shape = (6000, 3)
image_size = 6000
patch_size = 40
num_patches = image_size // patch_size  # 150
projection_dim = 40
num_heads = 4
patch_dim = 40 * 1 * patch_size 
transformer_units = [
    projection_dim,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4

class ConvF1Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=11, dropout_rate=0.1):
        super().__init__()
        # Re-introducing BatchNorm layers for the direct transfer
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(in_channels, eps=0.001, momentum=0.01) # TF-compatible settings

        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(in_channels, eps=0.001, momentum=0.01)

        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out, approximate='tanh')

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.gelu(out, approximate='tanh')
        
        out = out + identity

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.gelu(out, approximate='tanh')

        out = self.dropout(out)
        return out

# All other classes remain the same - just import them
from models.predictor_pt import (
    Patches,
    PatchEncoder,
    StochasticDepth,
    TransformerMLP,
    TransformerBlock,
    OutputHead,
    KerasMHA,
)

# Fixed S Model with correct BatchNorm settings
class EQCCTModelS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvF1Block(3, 10)
        self.conv2 = ConvF1Block(10, 20)
        self.conv3 = ConvF1Block(20, 40)

        self.patch = Patches(patch_size)
        self.encoder = PatchEncoder(num_patches, projection_dim, patch_dim)

        self.extra_pre = nn.ModuleList([ConvF1Block(40, 40) for _ in range(transformer_layers)])
        self.extra_post = nn.ModuleList([ConvF1Block(40, 40) for _ in range(transformer_layers)])

        self.transformers = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads, drop_prob=stochastic_depth_rate * (i / transformer_layers))
            for i in range(transformer_layers)
        ])

        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)
        self.head = OutputHead(in_channels=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        x = self.patch(x)
        x = self.encoder(x)

        for i in range(transformer_layers):
            # 1. Apply the pre-convolutional block
            x_pre_conv = x.transpose(1, 2)
            x_pre_conv = self.extra_pre[i](x_pre_conv).transpose(1, 2)
            x = x_pre_conv

            # --- Start of Manual Transformer Block Logic ---
            identity = x
            
            # 2. Layer Norm -> Attention
            x_norm1 = self.transformers[i].norm1(x)
            attention_output = self.transformers[i].attn(x_norm1)

            # 3. Apply the post-attention convolutional block
            attention_output_post_conv = attention_output.transpose(1, 2)
            attention_output_post_conv = self.extra_post[i](attention_output_post_conv).transpose(1, 2)
            
            # 4. First skip connection (Identity + Processed Attention Output)
            x = identity + self.transformers[i].drop_path1(attention_output_post_conv)

            # 5. MLP and second skip connection
            identity2 = x
            x_norm2 = self.transformers[i].norm2(x)
            x_mlp = self.transformers[i].mlp(x_norm2)
            x = identity2 + self.transformers[i].drop_path2(x_mlp)
            # --- End of Manual Transformer Block Logic ---

        # --- Final Layers ---
        x = self.norm(x)
        x = x.reshape(x.size(0), 6000, 1)
        return self.head(x)


# Fixed P Model with correct BatchNorm settings
class EQCCTModelP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvF1Block(3, 10)
        self.conv2 = ConvF1Block(10, 20)
        self.conv3 = ConvF1Block(20, 40)

        self.patch = Patches(patch_size)
        self.encoder = PatchEncoder(num_patches, projection_dim, patch_dim)

        self.transformer = nn.Sequential(*[
            TransformerBlock(projection_dim, num_heads, 
                           drop_prob=stochastic_depth_rate * (i / transformer_layers)) 
            for i in range(transformer_layers)
        ])
        
        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)
        self.head = OutputHead(in_channels=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        x = self.patch(x)
        x = self.encoder(x)
        x = self.transformer(x)
        x = self.norm(x)

        x = x.reshape(x.size(0), 6000, 1)
        return self.head(x)


# Test script
if __name__ == "__main__":
    import pickle
    from pathlib import Path

    from conversion.transfer_weights_legacy import (
        transfer_weights_p,
        transfer_weights_s_direct,
    )
    from models import predictor_pt_p as arch_p
    from reference import predictor_tf

    from paths import MODELPS_DIR, REPO_ROOT

    repo_root = REPO_ROOT
    model_dir = MODELPS_DIR

    print("Loading pickled TF weight dicts (see conversion/transfer_weights_legacy.py)...")

    with open(model_dir / "Sweights.pkl", "rb") as f:
        Sweights = pickle.load(f)

    with open(model_dir / "Pweights.pkl", "rb") as f:
        Pweights = pickle.load(f)

    # State dict keys (attn.o, mlp.fc1, …) match predictor_pt_p, not predictor_pt.TransformerBlock.
    modelS_fixed = arch_p.EQCCTModelS()
    modelS_fixed = transfer_weights_s_direct(Sweights, modelS_fixed)

    modelP_fixed = arch_p.EQCCTModelP()
    modelP_fixed = transfer_weights_p(Pweights, modelP_fixed)

    torch.save(modelS_fixed.state_dict(), repo_root / "modelS_fixed_bn.pth")
    torch.save(modelP_fixed.state_dict(), repo_root / "modelP_fixed_bn.pth")

    print("\n✅ Fixed models saved (repo root).")

    print("\n=== Testing Fixed S Model ===")
    _, modelS_tf = predictor_tf.load_eqcct_model(
        str(model_dir / "test_trainer_024.h5"),
        str(model_dir / "test_trainer_021.h5"),
    )
    
    np.random.seed(42)
    test_input = np.random.rand(1, 6000, 3).astype('float32')
    
    # TF output
    tf_output = modelS_tf(test_input, training=False).numpy()
    
    # PT output
    modelS_fixed.eval()
    with torch.no_grad():
        pt_input = torch.from_numpy(test_input)
        pt_output = modelS_fixed(pt_input).numpy()
    
    print(f"TF output: mean={tf_output.mean():.6f}, std={tf_output.std():.6f}")
    print(f"PT output: mean={pt_output.mean():.6f}, std={pt_output.std():.6f}")
    
    try:
        np.testing.assert_allclose(tf_output, pt_output, rtol=1e-4, atol=1e-4)
        print("\n✅ SUCCESS: The outputs match!")
    except AssertionError:
        diff = np.abs(tf_output - pt_output)
        print(f"\n⚠️  Still some differences: max={diff.max():.6f}, mean={diff.mean():.6f}")
        print("But this should be much closer than before!")