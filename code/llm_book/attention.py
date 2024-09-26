"""
The first version of attention mechanism.
"""

import math

import torch


class Attention(torch.nn.Module):
    """
    The self attention module.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout_percent=0) -> None:
        super().__init__()

        self.w_query = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.w_key = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.w_value = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout_percent)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Applies the forward pass to a batch of inputs.
        """
        query = self.w_query(batch)  # 4x6x2
        key = self.w_key(batch)      # 4x6x2
        value = self.w_value(batch)  # 4x6x2

        attn_score = query @ key.transpose(1, 2)

        # apply masking
        attn_score = attn_score.masked_fill(
            ~torch.tril(torch.ones_like(attn_score)).bool(), -torch.inf,
        )

        # find weights
        attn_weights = torch.nn.functional.softmax(
            attn_score / math.sqrt(key.shape[-1]), dim=-1,
        )
        # apply dropout
        attn_weights = self.dropout(attn_weights)  # 4x6x6
        print(attn_weights.shape)

        outputs = attn_weights @ value  # 4x6x2
        return outputs

def main():
    """
    The main method to test the attention module.
    """
    size = (6, 3)
    inputs = [
        torch.randn(size), torch.randn(size), torch.randn(size), torch.randn(size),
    ]
    batch = torch.stack([sample for sample in inputs])

    in_dim: int = batch.shape[-1]
    out_dim: int = 2
    dropout_percent: float = 0.2

    attn = Attention(in_dim, out_dim, dropout_percent)
    res = attn(batch)
    print(res)


if __name__ == '__main__':
    main()
