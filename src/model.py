# --------------------------------------------------------
# Transformer-Based Semantic Segmentation Framework
# Author: Teerapong Panboonyuen (Kao)
# Description: Simplified Transformer decoder architecture for segmentation.
# --------------------------------------------------------

import torch
import torch.nn as nn

class TransformerDecoderSeg(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)                      # B x C x H x W
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)        # (HW) x B x C
        x = self.transformer(x)                  # (HW) x B x C
        x = x.permute(1, 2, 0).reshape(B, C, H, W)
        return self.decoder(x)                   # B x num_classes x H x W