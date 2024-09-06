"""
This class represents a linear layer of the neural network.
"""
import torch
from torch import Tensor


class Linear:
    """
    Class to represent a linear layer.
    """

    def __init__(
            self,
            *,
            in_features: int,   # the number of input features
            out_features: int,  # the number of output features
            nonlinearity: str,  # the non-linearity that will be applied after this linear layer (optional)
            bias: bool = True,    # whether this layer should have a bias or not.
    ) -> None:
        super().__init__()

        size: tuple[int, int] = (in_features, out_features)
        self.weights = torch.empty(size, dtype=torch.float, requires_grad=True)

        # init the weights
        if nonlinearity:
            torch.nn.init.kaiming_normal_(self.weights, nonlinearity=nonlinearity)
        else:
            torch.nn.init.normal_(self.weights, mean=0, std=1)

        self.bias = None
        if bias:
            self.bias = torch.randn(
                out_features, dtype=torch.float, requires_grad=True
            ) * 0.01  # multiply near-zero to squash the bias

    def parameters(self) -> list:
        """
        Returns the parameters of this layer.
        """
        return [self.weights, self.bias] if self.bias is not None else [self.weights]

    def __call__(self, inputs: Tensor) -> Tensor:
        res = inputs @ self.weights
        if self.bias is not None:
            res += self.bias

        return res
    
    @property
    def in_features(self) -> int:
        return self.weights.shape[0]

    @property
    def out_features(self) -> int:
        return self.weights.shape[1]
