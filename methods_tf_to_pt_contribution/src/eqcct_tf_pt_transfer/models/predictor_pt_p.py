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

# Residual 1D conv block 
class ConvF1Block(nn.Module): 
    def __init__(self, in_channels, out_channels, kernal_size=11, dropout_rate=0.1):
        super().__init__() 
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernal_size, padding=kernal_size // 2)
        self.bn1 = nn.BatchNorm1d(in_channels, eps=0.001)

        self.conv2 = nn.Conv1d(in_channels, in_channels, kernal_size, padding=kernal_size // 2)
        self.bn2 = nn.BatchNorm1d(in_channels, eps=0.001)

        self.conv3 = nn.Conv1d(in_channels, out_channels, kernal_size, padding=kernal_size // 2)
        self.bn3 = nn.BatchNorm1d(out_channels, eps=0.001)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x): # Defines the forward pass of data through the n.n.
        # x = (batch_size, channels, 6000)

        # First layer 
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)

        # Second layer 
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.gelu(out)
        out = out + x  # Residual connection 

        # Third layer 
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.gelu(out)
        out = self.dropout(out)

        return out 

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
        if not self.training or self.drop_prob == 0.: 
            return x 
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return (x / keep_prob) * binary_tensor


# MLP with GeLU activations and dropout
# Matches predictor_tf.mlp(): each Dense uses activation=tf.nn.gelu before Dropout.
class TransformerMLP(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x
    
# TransformerBlock mimics the for loop logic in the create_cct_modelX()'s 
class TransformerBlock(nn.Module): 
    def __init__(self, dim, num_heads, drop_prob=0.1): 
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn  = KerasMHA(embed_dim=dim, num_heads=num_heads, key_dim=40)
        # self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.drop_path1 = StochasticDepth(drop_prob)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = TransformerMLP(dim, dropout_rate=0.1)
        self.drop_path2 = StochasticDepth(drop_prob)

    def forward(self, x): 
        identity = x 
        x = self.norm1(x)
        attn_out = self.attn(x) # Attention head with skip connection 
        x = identity + self.drop_path1(attn_out) 

        # MLP with skip connection 
        identity = x 
        x = self.norm2(x)
        x = identity + self.drop_path2(self.mlp(x))
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
    def __init__(self, embed_dim=40, num_heads=4, key_dim=40):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.key_dim    = key_dim
        self.inner_dim  = num_heads * key_dim          # 160
        self.scale      = 1.0 / math.sqrt(key_dim)

        self.q = nn.Linear(embed_dim, self.inner_dim, bias=True)
        self.k = nn.Linear(embed_dim, self.inner_dim, bias=True)
        self.v = nn.Linear(embed_dim, self.inner_dim, bias=True)
        self.o = nn.Linear(self.inner_dim, embed_dim, bias=True)

    # helper
    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.key_dim).transpose(1, 2)  # (B,H,T,D)

    def _merge(self, x):
        B, H, T, D = x.shape
        return x.transpose(1, 2).reshape(B, T, H * D)                      # (B,T,160)

    def forward(self, x):
        q = self._split(self.q(x))
        k = self._split(self.k(x))
        v = self._split(self.v(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        ctx = torch.matmul(weights, v)            # (B,H,T,D)
        ctx = self._merge(ctx)                    # (B,T,160)
        return self.o(ctx)    
    

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

            # 3. Apply the post-attention convolutional block (THE CRITICAL FIX)
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


