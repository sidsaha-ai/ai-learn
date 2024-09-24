"""
This script does an exercise to find the attention weights with trainable weights.
"""

import torch


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

def main():
    inputs = inputs_data()

    x2 = inputs[1]

    torch.manual_seed(123)  # reproducibility
    
    # instantiate the trainable weights
    size = (inputs.shape[1], 2)
    wq2 = torch.rand(size)  # query weights
    wk2 = torch.rand(size)  # key weights
    wv2 = torch.rand(size)  # value weights

    # compute the query for x2
    q2 = x2 @ wq2

    # compute the keys and values for all inputs with respect to x2
    k = inputs @ wk2
    v = inputs @ wv2

    # compute attention scores
    omega = q2 @ k.T
    print(omega)

if __name__ == '__main__':
    main()
