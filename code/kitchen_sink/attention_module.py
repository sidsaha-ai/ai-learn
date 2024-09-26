"""
This implements a module for the self-attention.
"""

import math

import torch


class SelfAttentionV1(torch.nn.Module):
    """
    The self attention module.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        size = (in_dim, out_dim)

        self.query_weight = torch.nn.Parameter(torch.rand(size))
        self.key_weight = torch.nn.Parameter(torch.rand(size))
        self.value_weight = torch.nn.Parameter(torch.rand(size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Makes the forward pass.
        """
        query = inputs @ self.query_weight
        key = inputs @ self.key_weight
        value = inputs @ self.value_weight

        attn_scores = query @ key.T
        attn_weights = torch.nn.functional.softmax(
            attn_scores / math.sqrt(key.shape[-1]), dim=-1,
        )

        outputs = attn_weights @ value

        return outputs


class SelfAttentionV2(torch.nn.Module):
    """
    The V2 of the self-attention module.
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.query_weight = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.key_weight = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.value_weight = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Makes the forward pass.
        """
        query = self.query_weight(inputs)
        key = self.key_weight(inputs)
        value = self.value_weight(inputs)

        attn_scores = query @ key.T

        # mask the attention scores
        attn_scores = attn_scores.masked_fill(
            ~torch.tril(torch.ones_like(attn_scores)).bool(), -torch.inf,
        )
        attn_weights = torch.nn.functional.softmax(
            attn_scores / math.sqrt(key.shape[-1]), dim=-1,
        )

        outputs = attn_weights @ value

        return outputs
