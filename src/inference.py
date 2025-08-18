# --------------------------------------------------------
# Transformer-Based Semantic Segmentation Framework
# Author: Teerapong Panboonyuen (Kao)
# Description: Inference script for applying a trained transformer decoder model to remote sensing images.
# --------------------------------------------------------

import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from model import TransformerDecoderSeg
from utils import load_checkpoint, save_predictions

def run_inference(cfg, logger):
    dataset = SegmentationDataset(cfg['data']['test'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = TransformerDecoderSeg(**cfg['model']).to(cfg['inference']['device'])
    load_checkpoint(model, cfg['inference']['checkpoint'], logger)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(cfg['inference']['device'])
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())

    save_predictions(all_preds, cfg['inference']['output_dir'], logger)