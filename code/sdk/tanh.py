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
        self.input = None
        self.input_grad = None

        self.training = True  # can be turned off during inference

    def __call__(self, x: Tensor) -> Tensor:
        self.input = x.requires_grad_(True)  # ensure input is trackable
        out = torch.tanh(x)

        self.output = out
        
        # register hook to capture the gradient during the backward pass
        if self.training:
            out.register_hook(self.save_gradient)
        return out

    def save_gradient(self, grad: Tensor) -> None:
        """
        Hook to save the gradient of the output during the backward pass.
        """
        self.input_grad = grad

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
