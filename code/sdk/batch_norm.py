"""
This class represents the batch norm layer.
"""

import torch
from torch import Tensor


class BatchNorm:
    """
    Class for the batch norm layer.
    """
    def __init__(
            self,
            *,
            num_features: int,     # the number of input features
            eps: float = 1e-5,     # the epsilon to add to avoid division by zero
            momentum: float = 0.1  # the momentum to use to udpate the weights of this layer
    ) -> None:
        self.eps: float = eps
        self.momentum: float = momentum

        size: tuple[int, int] = (1, num_features)

        self.gamma = torch.ones(size)
        self.beta = torch.zeros(size)
        self.gamma.requires_grad = True
        self.beta.requires_grad = True

        self.training = True  # should be changed to False during prediction

        # the final mean and variance to use during prediction
        self.mean = torch.zeros(size)
        self.var = torch.ones(size)

        # output
        self.output = None

    def parameters(self) -> list:
        """
        Returns the parameters list of this layer.
        """
        return [self.gamma, self.beta]

    @property
    def num_parameters(self) -> int:
        """
        Returns the number of parameters in this layer.
        """
        return sum(p.nelement() for p in self.parameters())

    def __call__(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(0, keepdim=True) if self.training else self.mean
        var = inputs.var(0, keepdim=True) if self.training else self.var

        outputs = self.gamma * ((inputs - mean) / torch.sqrt(var + self.eps)) + self.beta

        # update the mean and var of the class.
        if self.training:
            with torch.no_grad():
                self.mean = ((1 - self.momentum) * self.mean) + (self.momentum * mean)
                self.var = ((1 - self.momentum) * self.var) + (self.momentum * var)

        self.output = outputs
        return outputs

    def to_gpu(self) -> None:
        """
        Moves the tensors to the GPU, if available.
        """
        if not torch.backends.mps.is_available():
            return

        device = torch.device('mps')

        self.gamma.to(device)
        self.beta.to(device)
        self.mean.to(device)
        self.var.to(device)
