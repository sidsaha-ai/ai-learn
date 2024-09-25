"""
Excerise: https://tensorgym.com/exercises/0
"""

import torch

def add_tensors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

def main():
    print('==== Test case 1 ====')
    t1 = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    t2 = [
        [7, 8, 9],
        [10, 11, 12],
    ]
    t1 = torch.tensor(t1, dtype=torch.int)
    t2 = torch.tensor(t2, dtype=torch.int)

    res = add_tensors(t1, t2)
    print(res)

    print('=== Test Case 2 ===')
    t1 = [
        [1, 0],
        [0, 1],
    ]
    t2 = [
        [1, 2],
        [3, 4],
    ]
    t1 = torch.tensor(t1, dtype=torch.int)
    t2 = torch.tensor(t2, dtype=torch.int)
    res = add_tensors(t1, t2)
    print(res)


if __name__ == '__main__':
    main()
