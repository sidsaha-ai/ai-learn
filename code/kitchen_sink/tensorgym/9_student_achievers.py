"""
This contains the solution for the exercise https://tensorgym.com/exercises/6
"""
import torch

def solution(scores: torch.Tensor) -> torch.Tensor:
    """
    This implements the solution to the exercise.
    """
    norm_scores = torch.nn.functional.softmax(scores, dim=0)
    res = torch.sum(norm_scores > 0.3, dim=0)
    return res


if __name__ == '__main__':
    cases = []
    # test case 1
    inputs = [
        [0.2, 0.8],
        [1.0, 1.2],
        [0.5, 0.6],
    ]
    outputs = [1, 2]
    cases.append({
        'inputs': torch.tensor(inputs),
        'outputs': torch.tensor(outputs),
    })

    # test case 2
    inputs = [
        [0.3, 0.5, 0.4],
        [0.7, 0.4, 0.6],
        [0.4, 0.6, 0.5],
    ]
    outputs = [2, 3, 3]
    cases.append({
        'inputs': torch.tensor(inputs),
        'outputs': torch.tensor(outputs),
    })

    for ix, c in enumerate(cases):
        res = solution(c.get('inputs'))
        message = f'PASS: test case {ix + 1}' if torch.equal(res, c.get('outputs')) else f'FAIL: test case {ix + 1}'
        print(message)