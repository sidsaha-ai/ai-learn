"""
This file contains various utility functions.
"""
from torch import Tensor


def print_loss(epoch: int, loss: Tensor) -> None:
    """
    Function to print loss for an epoch.
    """
    print(f'{epoch=}, loss={loss.item():.4f}')
