"""
This class can be used to create embeddings for the inputs.
"""

import torch
from torch import Tensor

class Embedding:

    def __init__(
            self,
            *,
            num_embeddings: int,
            embedding_dim: int,
    ) -> None:
        super().__init__()

        size: tuple[int, int] = (num_embeddings, embedding_dim)
        self.embeddings: Tensor = torch.empty(
            size, dtype=torch.float, requires_grad=True,
        )

        # init with a uniform distribution
        torch.nn.init.uniform_(self.embeddings, a=-0.1, b=0.1)
