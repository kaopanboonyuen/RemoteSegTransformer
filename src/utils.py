# --------------------------------------------------------
# Transformer-Based Semantic Segmentation Framework
# Author: Teerapong Panboonyuen (Kao)
# Description: Utility functions for logging, checkpointing, metrics, etc.
# --------------------------------------------------------

import os
import torch
import logging
from datetime import datetime

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("segmentation")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(output_dir, "log.txt"))
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    return logger

def save_checkpoint(model, optimizer, epoch, output_dir, logger):
    path = os.path.join(output_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    logger.info(f"Saved checkpoint to {path}")

def load_checkpoint(model, checkpoint_path, logger):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model_state_dict'])
    logger.info(f"Loaded model from {checkpoint_path}")
    return model

def save_predictions(preds, output_dir, logger):
    os.makedirs(output_dir, exist_ok=True)
    for i, pred in enumerate(preds):
        path = os.path.join(output_dir, f"pred_{i}.png")
        Image.fromarray(pred.squeeze().astype('uint8')).save(path)
    logger.info(f"Saved {len(preds)} predictions.")