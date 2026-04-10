import math
from re import X
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

# Residual 1D conv block 
class ConvF1Block(nn.Module): 
    def __init__(self, in_channels, out_channels, kernal_size=11, dropout_rate=0.1):
        super().__init__() 
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernal_size, padding=kernal_size // 2, bias=True)
        self.bn1 = nn.BatchNorm1d(in_channels, eps=0.001, momentum=0.01)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(in_channels, in_channels, kernal_size, padding=kernal_size // 2, bias=True)
        self.bn2 = nn.BatchNorm1d(in_channels, eps=0.001, momentum=0.01)
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv1d(in_channels, out_channels, kernal_size, padding=kernal_size // 2, bias=True)
        self.bn3 = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01)
        self.act3 = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x): # Defines the forward pass of data through the n.n.
        # x = (batch_size, channels, 6000)
        residual = x 
        # First layer 
        x = self.act1(self.bn1(self.conv1(x)))
 
        # Second layer 
        x = self.act2(self.bn2(self.conv2(x)))
        x = residual + x  # Add layer 

        # Third layer 
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        return x 

# Splits time series into patches. (6000, 3) -> input for Transformer 
class Patches(nn.Module): # nn.Module is the foundational building block for building N.N's in PT
    def __init__(self, patch_size): 
        super().__init__()
        self.patch_size = patch_size 
    
    def forward(self, images): 
        # Input: (batch, 6000, 1, 40)
        B = images.size(0)
        P = self.patch_size 

        patches = images.unfold(1, P, P)

        # reorder so that every patch lies in contiguous memory
        # (B, n_patches, P, 1, 40)
        patches = patches.permute(0, 1, 4, 2, 3).contiguous()

        # flatten the last three dims → (B, n_patches, P*1*40 = 1600)
        patches = patches.view(B, patches.size(1), -1)
        return patches
    
# Projects each path to a linear projection and adds positional embeddings 
class PatchEncoder(nn.Module): 
    def __init__(self, num_patches, projection_dim, patch_dim):
        super().__init__()
        self.projection = nn.Linear(patch_dim, projection_dim) # Projects each path to a linear projection 
        self.position_embedding = nn.Embedding(num_patches, projection_dim) # Adds positional embeddings 

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = self.projection(x) + self.position_embedding(positions)
        return x 
    
# Drops paths randomly during training 
class StochasticDepth(nn.Module): 
    def __init__(self, drop_prob): 
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x): 
        if not self.training or self.drop_prob == 0.0: 
            return x 
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return (x / keep_prob) * binary_tensor


# To help with the vector transposition
class TokenConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ConvF1Block(40, 40)

    def forward(self, x):          # x: (B, 150, 40) <- 150 patches made out of 40 samples 
        x = x.transpose(1, 2)      # (B, 40, 150)
        x = self.block(x)          # conv over token axis
        return x.transpose(1, 2)   # back to (B, 150, 40)
    

# TransformerBlock mimics the for loop logic in the create_cct_modelX()'s 
class TransformerBlock(nn.Module): 
    def __init__(self, dim=40, num_heads=4, drop_prob=0.1, sd=0.0): 
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = KerasMHA(embed_dim=dim, num_heads=num_heads, head_dim=40, attn_drop=drop_prob, proj_drop=0.0)
        self.drop_path1 = StochasticDepth(sd) if sd > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)
        self.act = nn.GELU()
        self.dp1 = nn.Dropout(drop_prob)
        self.dp2 = nn.Dropout(drop_prob)
        self.drop_path2 = StochasticDepth(sd) if sd > 0 else nn.Identity()
    
    def mlp(self, x): 
        x = self.dp1(self.act(self.fc1(x)))
        x = self.dp2(self.act(self.fc2(x)))
        return x 

    
    def forward(self, x): 
        y = self.attn(self.norm1(x))
        x = x + self.drop_path1(y)
        z = self.mlp(self.norm2(x))
        x = x + self.drop_path2(z)
        return x 
        
# Mimics the LayerNorm and the reshaping done in load_eqcct_model
class OutputHead(nn.Module):
    def __init__(self, in_channels=1, kernel_size=15): 
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.activation = nn.Sigmoid()
    
    def forward(self, x): 
        x = x.transpose(1, 2)
        x = self.conv(x)
        return self.activation(x).transpose(1, 2)
    
class KerasMHA(nn.Module):
    """
    Faithful re-implementation of tf.keras.layers.MultiHeadAttention
    with   key_dim = 40,  num_heads = 4,  embed_dim = 40.
    Internal hidden size = key_dim * num_heads = 160.
    """
    def __init__(self, embed_dim=40, num_heads=4, head_dim=40, attn_drop=0.1, proj_drop=0.0):
        super().__init__()  
        inner = num_heads * head_dim 
        self.heads = num_heads
        self.head_dim = head_dim
        self.q = nn.Linear(embed_dim, inner, bias=True)
        self.k = nn.Linear(embed_dim, inner, bias=True)
        self.v = nn.Linear(embed_dim, inner, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner, embed_dim, bias=True)   
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x): 
        B, N, C = x.shape 
        def split(t): 
            t = t.view(B, N, self.heads, self.head_dim).transpose(1, 2)
            return t 
        q = split(self.q(x))
        k = split(self.k(x))
        v = split(self.v(x))

        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        y = attn @ v                               # (B, h, N, d)
        y = y.transpose(1, 2).reshape(B, N, -1)    # (B, N, 160)
        y = self.proj_drop(self.proj(y))           # (B, N, 40)
        return y

class EQCCTModelP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvF1Block(3, 10)
        self.conv2 = ConvF1Block(10, 20)
        self.conv3 = ConvF1Block(20, 40)

        self.patch = Patches(patch_size)
        self.encoder = PatchEncoder(num_patches, projection_dim, patch_dim)

        self.transformer = nn.Sequential(*[TransformerBlock(projection_dim, num_heads, drop_prob=stochastic_depth_rate * (i / transformer_layers)) for i in range(transformer_layers)])
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

def sd_at(i, total, rate): 
    return 0.0 if total <= 1 else rate * (i / (total - 1))

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
        self.extra_pre = nn.ModuleList([TokenConv() for _ in range(transformer_layers)])
        self.extra_post = nn.ModuleList([TokenConv() for _ in range(transformer_layers)])


        self.transformers = nn.ModuleList([
            TransformerBlock(
                            dim=projection_dim, 
                            num_heads=num_heads, 
                            drop_prob=0.1, 
                            sd=sd_at(i, transformer_layers, stochastic_depth_rate)) for i in range(transformer_layers)])

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
            x = self.extra_pre[i](x)               # Pre-ConvF1Block (B, 150, 40)
            residual = x 
            y = self.transformers[i].attn(
                self.transformers[i].norm1(x))     # Normalization layer & KerasMHA (B, 150, 40) - Multihead attention layer 
            y = self.extra_post[i](x)              # Post-ConvF1Block (B, 150, 40)
            x = residual + self.transformers[i].drop_path1(y)  # StochasticDepth + Add 

            residual = x 
            z = self.transformers[i].mlp(self.transformers[i].norm2(x)) # Normalization layer & MultiLayerPerceptron
            x = residual + self.transformers[i].drop_path2(z)   # Strochastic Depth Dropout + add original input
 
        x = self.norm(x)
        x = x.reshape(x.size(0), 6000, 1)
        return self.head(x)

