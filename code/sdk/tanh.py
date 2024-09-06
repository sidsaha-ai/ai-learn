"""
This class represents the Tanh layer.
"""

import torch
from torch import Tensor


class Tanh:
    """
    Class to represent the Tanh layer.
    """

    def __call__(self, x: Tensor) -> Tensor:
        return torch.tanh(x)
