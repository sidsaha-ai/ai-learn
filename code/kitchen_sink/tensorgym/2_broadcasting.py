"""
Exercise https://tensorgym.com/exercises/14
"""

import torch


def fill_tensor_with_value(x: torch.Tensor, value: int) -> torch.Tensor:
    return x * 0 + value


def main():
    print('=== Test Case 1 ===')
    t = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    t = torch.tensor(t)
    value = 10

    res = fill_tensor_with_value(t, value)
    print(res)

    print('==== Test Case 2 ===')
    t = [
        [0, -1],
        [-2, -3],
    ]
    t = torch.tensor(t)
    value = 5
    res = fill_tensor_with_value(t, value)
    print(res)


if __name__ == '__main__':
    main()
