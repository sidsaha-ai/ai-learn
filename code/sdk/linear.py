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
            in_features: int,     # the number of input features
            out_features: int,    # the number of output features
            nonlinearity: str,    # the non-linearity that will be applied after this linear layer (optional)
            bias: bool = True,    # whether this layer should have a bias or not.
    ) -> None:
        super().__init__()

        size: tuple[int, int] = (in_features, out_features)
        self.weights = torch.empty(size, dtype=torch.float)

        # init the weights
        """
        Knowledge:
        For layers that will have to go through a non-linearity, it is good to initiliaze
        the weights so that their variance generally constant so that gradients do not explode
        or vanish.

        For last layer (where there is no nonlinearity), it is better to keep weights small, so
        that the initial loss is not very high. If the initial loss is very high, then
        gradient could be very high, and gradient descent updates might overshoot the local
        minima.
        """
        if nonlinearity:
            torch.nn.init.kaiming_normal_(self.weights, nonlinearity=nonlinearity)
        else:
            self.weights = torch.randn(
                size, dtype=torch.float) * 0.01

        self.weights.requires_grad = True

        self.bias = None
        if bias:
            self.bias = torch.randn(
                out_features, dtype=torch.float) * 0.01  # multiply near-zero to squash the bias
            self.bias.requires_grad = True
        
        self.output = None

    def parameters(self) -> list:
        """
        Returns the parameters of this layer.
        """
        return [self.weights, self.bias] if self.bias is not None else [self.weights]

    def __call__(self, inputs: Tensor) -> Tensor:
        res = inputs @ self.weights
        if self.bias is not None:
            res += self.bias

        self.output = res
        return res

    def to_gpu(self) -> None:
        """
        Moves the tensors to the GPU, if available.
        """
        if not torch.backends.mps.is_available():
            return

        device = torch.device('mps')

        self.weights.to(device)
        if self.bias is not None:
            self.bias.to(device)

    @property
    def in_features(self) -> int:
        """
        Returns the number of input features to this layer.
        """
        return self.weights.shape[0]

    @property
    def out_features(self) -> int:
        """
        Returns the numnber of output features from this layer.
        """
        return self.weights.shape[1]

    @property
    def num_parameters(self) -> int:
        """
        Returns the number of parameters from this layer.
        """
        return sum(p.nelement() for p in self.parameters())
