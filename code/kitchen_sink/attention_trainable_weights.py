"""
This script does an exercise to find the attention weights with trainable weights.
"""

import math

import torch

from kitchen_sink.attention_module import SelfAttention


def inputs_data() -> torch.Tensor:
    """
    Returns the input data.
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

    return inputs


def main_one_input():
    """
    The main method to find the context vector for one input.
    """
    inputs = inputs_data()
    hidden_dim: int = 2

    x2 = inputs[1]                               # 1x3

    torch.manual_seed(123)  # reproducibility

    # instantiate the trainable weights
    size = (inputs.shape[1], hidden_dim)  # 3x2
    wq2 = torch.rand(size)  # query weights      # 3x2
    wk2 = torch.rand(size)  # key weights        # 3x2
    wv2 = torch.rand(size)  # value weights      # 3x2

    # compute the query for x2
    q2 = x2 @ wq2                                # 1x2

    # compute the keys and values for all inputs with respect to x2
    k = inputs @ wk2                             # 6x2
    v = inputs @ wv2                             # 6x2

    # compute attention scores
    omega = q2 @ k.T                             # 1x6

    # compute attention weights
    # before applying the softmax, scale omega.
    alpha = torch.nn.functional.softmax(         # 1x6
        omega / math.sqrt(k.shape[-1]), dim=0,
    )

    # find the context vector
    z2 = alpha @ v
    print(z2)


def main():
    """
    The main method to find context vectors for all inputs.
    """
    inputs = inputs_data()                                 # 6x3
    hidden_dim: int = 2

    torch.manual_seed(123)  # reproducibility

    # instantiate the trainable weights
    # NOTE: there is no need to use individual weights
    # for each input row. We use one set of weights
    # for all input rows.
    size = (inputs.shape[1], hidden_dim)                            # 3x2
    wq = torch.rand(size)                                  # 3x2
    wk = torch.rand(size)                                  # 3x2
    wv = torch.rand(size)                                  # 3x2

    # compute queries for all inputs
    q = inputs @ wq                                        # 6x2

    # compute keys for all inputs
    k = inputs @ wk                                        # 6x2

    # compute values for all inputs
    v = inputs @ wv                                        # 6x2

    # compute attention scores
    omega = q @ k.T                                        # 6x6

    # compute attention weights
    alpha = torch.nn.functional.softmax(                   # 6x6
        omega / math.sqrt(k.shape[-1]), dim=-1,  # across the last dimension
    )

    # compute context vectors
    z = alpha @ v                                          # 6x2

    print(z)


def main_with_attention_module():
    """
    Main method that finds self-attention using the module.
    """
    inputs = inputs_data()
    hidden_dim: int = 2

    torch.manual_seed(123)

    m = SelfAttention(inputs.shape[1], hidden_dim)
    z = m(inputs)

    print(z)


if __name__ == '__main__':
    main_one_input()
    print()

    main()
    print()

    main_with_attention_module()
