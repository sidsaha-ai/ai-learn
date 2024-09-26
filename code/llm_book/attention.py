"""
The first version of attention mechanism.
"""

import math

import torch


class Attention(torch.nn.Module):
    """
    The self attention module.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.w_query = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.w_key = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.w_value = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies the attention mechanism.
        """
        query = self.w_query(inputs)
        key = self.w_key(inputs)
        value = self.w_value(inputs)

        attn_score = query @ key.T

        # apply masking to attention scores
        attn_score = attn_score.masked_fill(
            ~torch.tril(torch.ones_like(attn_score)).bool(), -torch.inf,
        )
        attn_weights = torch.nn.functional.softmax(
            attn_score / math.sqrt(key.shape[-1]), dim=-1,
        )

        outputs = attn_weights @ value

        return outputs


def main():
    """
    The main method to test the attention module.
    """
    inputs = [
        [0.43, 0.15, 0.89],  # your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ]

    inputs = torch.tensor(inputs)

    in_dim: int = inputs.shape[1]
    out_dim: int = 2

    attention = Attention(in_dim, out_dim)

    res = attention(inputs)
    print(res)

if __name__ == '__main__':
    main()
