# --------------------------------------------------------
# Transformer-Based Semantic Segmentation Framework
# Author: Teerapong Panboonyuen (Kao)
# Description: Training loop for transformer-based semantic segmentation model.
# --------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from model import TransformerDecoderSeg
from utils import save_checkpoint

def train(cfg, logger):
    dataset = SegmentationDataset(cfg['data']['train'])
    dataloader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)

    model = TransformerDecoderSeg(**cfg['model']).to(cfg['training']['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg['training']['epochs']):
        model.train()
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(cfg['training']['device']), masks.to(cfg['training']['device'])

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Loss: {avg_loss:.4f}")

        if (epoch + 1) % cfg['training']['save_freq'] == 0:
            save_checkpoint(model, optimizer, epoch+1, cfg['output_dir'], logger)