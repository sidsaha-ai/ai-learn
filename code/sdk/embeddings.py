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

        self.training = True  # to turn off training during inference
        self.input = None
        self.output = None

    def parameters(self) -> list:
        """
        Returns the parameters.
        """
        return [self.weights]

    def __call__(self, inputs: Tensor) -> Tensor:
        self.input = inputs
        self.output = self.weights[inputs]

        return self.output

    def to_gpu(self) -> None:
        """
        Moves the tensors to the GPU, if available.
        """
        if not torch.backends.mps.is_available():
            return

        device = torch.device('mps')
        self.weights.to(device)
