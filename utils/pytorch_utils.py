"""
Utility functions for PyTorch operations.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Disclaimer: This code has been modified from many sources.
"""

import random
import os
import sys
import math
import multiprocessing
from functools import reduce, partial
from subprocess import call

import numpy as np

import torch
from torch.nn import functional as F
import torch.cuda as tcuda


# -------------------------------------------------------------------- #
#                           Reproducibility
# -------------------------------------------------------------------- #
### Ref: https://pytorch.org/docs/stable/notes/randomness.html

class SeedAll:
    def __init__(self, seed, logger=None) -> None:
        """Set seed for reproducibility
        Args:
            seed (_type_, optional): Random seed. Defaults to None.
            logger (_type_, optional): Logger object. Defaults to None.
        """
        self.seed = seed
        
        self.logger = logger
        
        ### Set seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.numpy_generator = np.random.default_rng(seed=self.seed)

        ### Set PYTHONHASHSEED
        os.environ['PYTHONHASHSEED'] = str(self.seed)

        self.torch_generator = torch.manual_seed(self.seed) ### Set this one is enough

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if self.logger is None:
            print(f'Using random seed: {self.seed}')
        else:
            self.logger.info(f'Using random seed: {self.seed}')


def seed_all(seed=None, logger=None) -> torch._C.Generator:
    """
    Set seed for reproducibility
    Ref: https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed is None:
        seed = 2024

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch_generator = torch.manual_seed(seed) ### Set this one is enough

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    if logger is None:
        print(f'Using random seed: {seed}')
    else:
        logger.info(f'Using random seed: {seed}')

    return torch_generator


def seed_worker(worker_id: int) -> None:
    """
    Set seed for workers in DataLoader
    Ref: https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    Args:
        worker_id (): Received worker_id. Ex: 0, 1, 2

    Returns: None

    For example:
    ------------
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # print(f"{worker_id=}")
    # print(f"{torch.initial_seed()=}")
    # print(f"{worker_seed=}")

