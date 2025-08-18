# --------------------------------------------------------
# Transformer-Based Semantic Segmentation Framework
# Author: Teerapong Panboonyuen (Kao)
# Description: Advanced Transformer decoder architecture with positional encoding,
# multi-scale patch embedding, and feed-forward networks for remote sensing segmentation.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding for feature maps"""
    def __init__(self, embed_dim, height, width):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, height, width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, C, H, W)
        return x + self.pos_embed

class PatchEmbedding(nn.Module):
    """Patchify input image and embed into a vector space"""
    def __init__(self, in_channels=3, embed_dim=256, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: B x C x H x W
        x = self.proj(x)  # B x embed_dim x H/P x W/P
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B x N x C, where N=H*W patches
        x = self.norm(x)
        return x, H, W

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: N x B x C (note: MultiheadAttention expects seq_len x batch x embed_dim)
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x

class TransformerDecoderSeg(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, embed_dim=256, num_heads=8, num_layers=6, patch_size=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.pos_embed = None  # Initialized later dynamically

        # Build multiple transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Decoder head: ConvTranspose to upscale feature map to original resolution
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x: B x C x H x W
        B = x.size(0)

        # Patch embedding + normalization
        x, H, W = self.patch_embed(x)  # x: B x N x C
        N, C = x.shape[1], x.shape[2]

        # Initialize or resize positional encoding if necessary
        if (self.pos_embed is None) or (self.pos_embed.pos_embed.shape[2] != H) or (self.pos_embed.pos_embed.shape[3] != W):
            self.pos_embed = PositionalEncoding2D(C, H, W).to(x.device)

        # Add positional encoding (convert x to B x C x H x W to add 2D pos embed)
        x_reshaped = x.transpose(1, 2).reshape(B, C, H, W)
        x_reshaped = self.pos_embed(x_reshaped)
        x = x_reshaped.flatten(2).transpose(1, 2)  # Back to B x N x C

        # Prepare for transformer (seq_len x batch x embed_dim)
        x = x.transpose(0, 1)

        # Pass through transformer decoder layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.transpose(0, 1)  # Back to B x N x C

        # Reshape to feature map
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Decode & upsample to input resolution
        logits = self.decoder_head(x)  # B x num_classes x (H*patch) x (W*patch)

        return logits