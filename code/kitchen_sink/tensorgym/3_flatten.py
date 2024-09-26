"""
Script for exercise https://tensorgym.com/exercises/1
"""
import torch


def flatten(x: torch.Tensor) -> torch.Tensor:
    """
    The exercise function to be implemented.
    """
    m = torch.nn.Flatten(0)
    y = m(x)
    return y

if __name__ == '__main__':
    print('=== Test Case 1 ====')
    inputs = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    inputs = torch.tensor(inputs)
    res = flatten(inputs)
    print(res)

    print('=== Test Case 2 ===')
    inputs = [
        [1, 0],
        [0, 1],
    ]
    inputs = torch.tensor(inputs)
    res = flatten(inputs)
    print(res)
