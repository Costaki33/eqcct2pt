import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from models.predictor_pt import (
    ConvF1Block,
    Patches,
    PatchEncoder,
    TransformerBlock,
    OutputHead,
)

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

# To help with the vector transposition
class TokenConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ConvF1Block(40, 40)

    def forward(self, x):          # x: (B, 150, 40) <- 150 patches made out of 40 samples 
        x = x.transpose(1, 2)      # (B, 40, 150)
        x = self.block(x)          # conv over token axis
        return x.transpose(1, 2)   # back to (B, 150, 40)


class EQCCTModelS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvF1Block(3, 10)
        self.conv2 = ConvF1Block(10, 20)
        self.conv3 = ConvF1Block(20, 40)

        # Patching on (B, 40, 1, 6000)
        self.patches = Patches(patch_size)
        self.encoder = PatchEncoder(num_patches, projection_dim, patch_dim)

        # For each transformer layer, do ConvF1Block before and after attention 
        self.pre = nn.ModuleList([TokenConv() for _ in range(transformer_layers)])
        self.post = nn.ModuleList([TokenConv() for _ in range(transformer_layers)])


        self.transformers = nn.ModuleList([
            TransformerBlock(
                            dim=projection_dim, 
                            num_heads=num_heads, 
                            drop_prob=stochastic_depth_rate * (i / transformer_layers)) for i in range(transformer_layers)])

        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)
        self.head = OutputHead(in_channels=1)
    
    def forward(self, x):                       # x: (B, 6000, 3)
        x = x.transpose(1, 2)                   # x: (B, 3, 6000)

        # Go through the 3 Conv Blocks 
        x = self.conv1(x)                       # (B, 10, 6000)
        x = self.conv2(x)                       # (B, 20, 6000)
        x = self.conv3(x)                       # (B, 40, 6000)

        # Patch and Embedding Layer 
        x = x.unsqueeze(2).permute(0, 3, 2, 1)  # (B, 6000, 1, 40) -> treat as H=6000, W=1, C=40
        x = self.patches(x)                     # (B, 150, 1600)
        x = self.encoder(x)                     # (B, 150, 40)

        # 4 Transformer blocks 
        for i in range (transformer_layers):
            x = self.pre[i](x)                  # Pre-ConvF1Block (B, 150, 40)

            y = self.transformers[i].norm1(x)   # Normalization layer 
            y = self.transformers[i].attn(y)    # (B, 150, 40) - Multihead attention layer 

            y = self.post[i](y)                 # Post-ConvF1Block (B, 150, 40)
            x = x + self.transformers[i].drop_path1(y) # Strochastic Depth Dropout + add original input 

            z = self.transformers[i].norm2(x)   # 2nd Normalization layer
            z = self.transformers[i].mlp(z)     # MLP 
            x = x + self.transformers[i].drop_path2(z)

        #  
        x = self.norm(x)
        x = x.reshape(x.size(0), 6000, 1)
        return self.head(x)

