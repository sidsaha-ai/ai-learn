"""
This implements the solution to the exercise https://tensorgym.com/exercises/8
"""
import math

import torch


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    This is the solution implementation to the exercise.
    """
    score = query @ key.T
    score = score / math.sqrt(query.shape[-1])

    weights = torch.nn.functional.softmax(score, dim=-1)

    return weights @ value


def check_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    """
    Checks whether two tensors are equal.
    """
    if x.shape != y.shape:
        return False

    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            el1 = f'{x[row, col].item():.4f}'
            el2 = f'{y[row, col].item():.4f}'
            if el1 != el2:
                return False

    return True


def main():
    """
    The main function that runs the test cases.
    """
    cases = []

    # test case 1
    query = [
        [1, 2],
        [3, 4],
    ]
    key = [
        [1, 1],
        [0, 0],
    ]
    value = [
        [2, 2],
        [3, 3],
    ]
    outputs = [
        [2.1070, 2.1070],
        [2.0070, 2.0070],
    ]
    cases.append({
        'query': torch.tensor(query, dtype=torch.float),
        'key': torch.tensor(key, dtype=torch.float),
        'value': torch.tensor(value, dtype=torch.float),
        'outputs': torch.tensor(outputs, dtype=torch.float),
    })

    # test case 2
    query = [
        [1, 0, 1],
        [0, 1, 1],
    ]
    key = [
        [1, 1, 0],
        [0, 0, 1],
    ]
    value = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    outputs = [
        [2.5, 3.5, 4.5],
        [2.5, 3.5, 4.5],
    ]
    cases.append({
        'query': torch.tensor(query, dtype=torch.float),
        'key': torch.tensor(key, dtype=torch.float),
        'value': torch.tensor(value, dtype=torch.float),
        'outputs': torch.tensor(outputs, dtype=torch.float),
    })

    for ix, c in enumerate(cases):
        res = attention(
            c.get('query'), c.get('key'), c.get('value'),
        )
        message = f'PASS: test case {ix + 1}' if check_equal(res, c.get('outputs')) else f'FAIL: test case {ix + 1}'
        print(message)


if __name__ == '__main__':
    main()
