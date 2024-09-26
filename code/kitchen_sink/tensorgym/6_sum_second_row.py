"""
This is the solution to exercise https://tensorgym.com/exercises/3
"""
import torch


def solution(inputs: torch.Tensor) -> torch.Tensor:
    """
    This implements the solution function to the exercise.
    """
    outputs = inputs[::2]  # get every second row
    outputs = torch.sum(outputs, dim=1)
    return outputs


if __name__ == '__main__':
    # test case 1
    matrix = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
    ]
    result = [3, 11, 19]

    matrix = torch.tensor(matrix)
    result = torch.tensor(result)

    res = solution(matrix)

    message = 'PASS: test case 1' if torch.equal(res, result) else 'FAIL: test case 1'
    print(message)

    # test case 2
    matrix = [
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
    ]
    result = [2, 6]

    matrix = torch.tensor(matrix)
    result = torch.tensor(result)

    res = solution(matrix)

    message = 'PASS: test case 2' if torch.equal(res, result) else 'FAIL: test case 2'
    print(message)
