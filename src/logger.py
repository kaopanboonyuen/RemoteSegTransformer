# --------------------------------------------------------
# Swin Transformer Logger Utility (Refactored)
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Refactored by Teerapong Panboonyuen (Kao)
# --------------------------------------------------------

import os
import sys
import logging
import functools
from termcolor import colored

@functools.lru_cache()
def create_logger(output_dir: str, dist_rank: int = 0, name: str = ''):
    """
    Creates and configures a logger instance with both console and file output.

    Args:
        output_dir (str): Directory where log file will be saved.
        dist_rank (int): Distributed rank (used to control logging for master process).
        name (str): Logger name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Log format definitions
    base_format = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_format = (
        colored('[%(asctime)s %(name)s]', 'green') +
        colored(' (%(filename)s:%(lineno)d)', 'yellow') +
        ': %(levelname)s %(message)s'
    )

    # Console logger for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(fmt=color_format, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # File logger for all ranks
    log_file = os.path.join(output_dir, f'log_rank{dist_rank}.txt')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=base_format, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger