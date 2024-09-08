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

        self.training = True  # should be changed to False during prediction

        # the final mean and variance to use during prediction
        self.mean = torch.ones(size)
        self.var = torch.zeros(size)

    def parameters(self) -> list:
        """
        Returns the parameters list of this layer.
        """
        return [self.gamma, self.beta]

    def __call__(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(0, keepdim=True) if self.training else self.mean
        var = inputs.var(0, keepdim=True) if self.training else self.var

        outputs = self.gamma * ((inputs - mean) / torch.sqrt(var + self.eps)) + self.beta

        # update the mean and var of the class.
        if self.training:
            with torch.no_grad():
                self.mean = ((1 - self.momentum) * self.mean) + (self.momentum * self.mean)
                self.var = ((1 - self.momentum) * self.var) + (self.momentum * self.var)

        return outputs
    
    def to_gpu(self) -> None:
        if not torch.backends.mps.is_available():
            return
        
        device = torch.device('mps')
        
        self.gamma.to(device)
        self.beta.to(device)
        self.mean.to(device)
        self.var.to(device)
