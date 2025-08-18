# --------------------------------------------------------
# Transformer-Based Semantic Segmentation Framework
# Author: Teerapong Panboonyuen (Kao)
# Description: Entry point for training or inference using a transformer decoder on remote sensing imagery.
# --------------------------------------------------------

import argparse
import yaml
from train import train
from inference import run_inference
from utils import setup_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='train')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger = setup_logger(config['output_dir'])

    if args.mode == 'train':
        train(config, logger)
    else:
        run_inference(config, logger)

if __name__ == '__main__':
    main()