"""
Script for the exercise https://tensorgym.com/exercises/18
"""
import torch


def drop_column(x: torch.Tensor) -> torch.Tensor:
    """
    The exercise function to be implemented.
    """
    # find the number of zeros in each row.
    num_zeros = torch.sum(x == 0, dim=0)

    # find the column with the most zeros
    target_col = torch.argmax(num_zeros).item()
    result = torch.cat((x[:, 0:target_col], x[:, target_col + 1:]), dim=1)
    return result


if __name__ == '__main__':
    # Test Case 1
    inputs = [
        [1, 0, 3],
        [4, 5, 6],
    ]
    outputs = [
        [1, 3],
        [4, 6],
    ]
    inputs = torch.tensor(inputs)
    outputs = torch.tensor(outputs)

    res = drop_column(inputs)

    message = 'PASS: test case 1' if torch.equal(res, outputs) else 'FAIL: test case 1'
    print(message)

    # Test Case 2
    inputs = [
        [1, 0, 0],
        [0, 5, 0],
    ]
    outputs = [
        [1, 0],
        [0, 5],
    ]

    inputs = torch.tensor(inputs)
    outputs = torch.tensor(outputs)

    res = drop_column(inputs)

    message = 'PASS: test case 2' if torch.equal(res, outputs) else 'FAIL: test case 2'
    print(message)

    # Test Case 3
    inputs = [
        [0, 0, 7, 8],
        [0, 0, 1, 5],
        [3, 0, 5, 0],
        [9, 5, 4, 0],
    ]
    outputs = [
        [0, 7, 8],
        [0, 1, 5],
        [3, 5, 0],
        [9, 4, 0],
    ]

    inputs = torch.tensor(inputs)
    outputs = torch.tensor(outputs)

    res = drop_column(inputs)

    message = 'PASS: test case 3' if torch.equal(res, outputs) else 'FAIL: test case 3'
    print(message)
