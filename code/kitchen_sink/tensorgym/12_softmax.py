"""
This implements the solution for the exercise https://tensorgym.com/exercises/7
"""
import torch


def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    The implementation of the solution for the exercise.
    """
    logits = logits - torch.max(logits, dim=1, keepdim=True).values  # for numerical stability
    return torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)


def main():
    """
    The main function running the test cases.
    """
    cases = []

    # test case 1
    inputs = [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
    ]
    inputs = torch.tensor(inputs)
    outputs = torch.nn.functional.softmax(inputs, dim=1)
    cases.append({
        'inputs': inputs,
        'outputs': outputs,
    })

    # test case 2
    inputs = [
        [10000.0, 10000.0, 9999.0],
        [9999.0, 9998.0, 9997.0],
    ]
    inputs = torch.tensor(inputs)
    outputs = torch.nn.functional.softmax(inputs, dim=1)
    cases.append({
        'inputs': inputs,
        'outputs': outputs,
    })

    for ix, c in enumerate(cases):
        res = softmax(c.get('inputs'))
        message = f'PASS: test case {ix + 1}' if torch.equal(res, c.get('outputs')) else f'FAIL: test case {ix + 1}'
        print(message)


if __name__ == '__main__':
    main()
