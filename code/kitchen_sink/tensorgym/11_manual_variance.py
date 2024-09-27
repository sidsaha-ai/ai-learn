"""
This solves the exercise problem https://tensorgym.com/exercises/10
"""
import torch


def solution(data: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the exercise solution.
    """
    data = data.float()
    mean = torch.mean(data)

    var = torch.sum(torch.pow((data - mean), 2)) / data.shape[0]
    return var


def main():
    """
    Main function that runs the test cases.
    """
    cases = []

    # test case 1
    inputs = [1, 2, 3, 4, 5]
    outputs = 2
    cases.append({
        'inputs': torch.tensor(inputs),
        'outputs': torch.tensor(outputs),
    })

    # test case 2
    inputs = [3, 3, 3, 3, 3]
    outputs = 0
    cases.append({
        'inputs': torch.tensor(inputs),
        'outputs': torch.tensor(outputs),
    })

    for ix, c in enumerate(cases):
        res = solution(c.get('inputs'))
        message = f'PASS: test case {ix + 1}' if torch.equal(res, c.get('outputs')) else f'FAIL: test case {ix + 1}'
        print(message)


if __name__ == '__main__':
    main()
