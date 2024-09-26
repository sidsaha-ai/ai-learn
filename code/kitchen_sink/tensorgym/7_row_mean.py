"""
Exercise solution for https://tensorgym.com/exercises/4
"""
import torch


def solution(x: torch.Tensor) -> torch.Tensor:
    """
    The function for the solution to be implemented.
    """
    means = torch.mean(x, dim=1, dtype=torch.float)
    max_col = torch.argmax(means)
    return max_col


if __name__ == '__main__':
    cases = []

    # test case 1
    inputs = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    outputs = 2
    cases.append({
        'inputs': torch.tensor(inputs),
        'outputs': torch.tensor(outputs),
    })

    # test case 2
    inputs = [
        [10, 20],
        [30, 40],
        [50, 60],
    ]
    outputs = 2
    cases.append({
        'inputs': torch.tensor(inputs),
        'outputs': torch.tensor(outputs),
    })

    for ix, c in enumerate(cases):
        res = solution(c.get('inputs'))
        message = f'PASS: test case {ix + 1}' if torch.equal(res, c.get('outputs')) else f'FAIL: test case {ix + 1}'
        print(message)
