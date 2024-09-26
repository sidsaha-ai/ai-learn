"""
This is the answer to exercise https://tensorgym.com/exercises/2
"""

import torch


def solution(inputs: torch.Tensor) -> torch.Tensor:
    """
    The exercise function to be implemented.
    """
    num_rows: int = 2
    num_cols: int = int(inputs.shape[0] / num_rows)
    inputs = inputs.reshape(num_rows, num_cols)

    res = inputs.transpose(0, 1)

    return res


if __name__ == '__main__':
    print('=== Test Case 1 ===')
    inputs = [1, 2, 3, 4]
    inputs = torch.tensor(inputs)
    res = solution(inputs)
    print(res)
    print()

    print('=== Test Case 2 ===')
    inputs = [1, 2, 3, 4, 5, 6]
    inputs = torch.tensor(inputs)
    res = solution(inputs)
    print(res)
    print()

    print('=== Test Case 3 ===')
    inputs = [1, 0, 0, 1]
    inputs = torch.tensor(inputs)
    res = solution(inputs)
    print(res)
    print()
