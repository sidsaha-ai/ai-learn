"""
This class represents the Tanh layer.
"""

import torch
from torch import Tensor


class Tanh:
    """
    Class to represent the Tanh layer.
    """

    def __init__(self) -> None:
        super().__init__()

        self.output = None

    def __call__(self, x: Tensor) -> Tensor:
        out = torch.tanh(x)

        self.output = out
        return out

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
