"""
This is the answer to exercise https://tensorgym.com/exercises/2
"""

import torch


def solution(x: torch.Tensor) -> torch.Tensor:
    """
    The exercise function to be implemented.
    """
    num_rows: int = 2
    num_cols: int = int(x.shape[0] / num_rows)
    x = x.reshape(num_rows, num_cols)

    result = x.transpose(0, 1)

    return result


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
