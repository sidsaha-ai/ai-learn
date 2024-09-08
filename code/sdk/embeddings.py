"""
This class can be used to create embeddings for the inputs.
"""

import torch
from torch import Tensor


class Embedding:
    """
    Class to represent Embeddings.
    """

    def __init__(
            self,
            *,
            num_embeddings: int,
            embedding_dim: int,
    ) -> None:
        super().__init__()

        size: tuple[int, int] = (num_embeddings, embedding_dim)
        self.weights: Tensor = torch.empty(
            size, dtype=torch.float,
        )

        # init with a normal distribution with a mean of 0 and a standard deviation of 1.
        torch.nn.init.normal_(self.weights, mean=0, std=1)
        self.weights.requires_grad = True

    def parameters(self) -> list:
        """
        Returns the parameters.
        """
        return [self.weights]

    def __getitem__(self, index) -> Tensor:
        return self.weights[index]

    def __matmul__(self, other: Tensor) -> Tensor:
        """
        Implements the @ operator like `embedding @ other`.
        """
        return self.weights @ other

    def __rmatmul__(self, other: Tensor) -> Tensor:
        """
        Implements the @ operator like `other @ embedding`.
        """
        return other @ self.weights

    @property
    def shape(self) -> torch.Size:
        """
        Adds a property to get the shape of the underlying tensor.
        """
        return self.weights.shape

    @property
    def num_parameters(self) -> int:
        """
        Returns the number of parameters in this layer.
        """
        return sum(p.nelement() for p in self.parameters())

    def view(self, size: tuple[int, int]) -> Tensor:
        """
        Applies the view method on the underlying tensor.
        """
        return self.weights.view(size)

    def to_gpu(self) -> None:
        """
        Moves the tensors to the GPU, if available.
        """
        if not torch.backends.mps.is_available():
            return

        device = torch.device('mps')
        self.weights.to(device)
