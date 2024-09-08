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

    def parameters(self) -> list:
        """
        Returns the parameters of this layer.
        """
        return []

    @property
    def num_parameters(self) -> int:
        """
        Returns the number of parameters in this layer.
        """
        return sum(p.nelement() for p in self.parameters())
