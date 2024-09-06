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
            size, dtype=torch.float, requires_grad=True,
        )

        # init with a normal distribution with a mean of 0 and a standard deviation of 1.
        torch.nn.init.normal_(self.weights, mean=0, std=1)

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
    def shape(self) -> tuple[int, int]:
        return self.weights.shape
    
    def view(self, size: tuple[int, int]) -> Tensor:
        return self.weights.view(size)
    