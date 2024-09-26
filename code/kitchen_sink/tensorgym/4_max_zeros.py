"""
Script for the exercise https://tensorgym.com/exercises/18
"""
import torch


def drop_column(inputs: torch.Tensor) -> torch.Tensor:
    """
    The exercise function to be implemented.
    """
    # find the number of zeros in each row.
    num_zeros = torch.sum(inputs == 0, dim=0)

    # find the column with the most zeros
    target_col = torch.argmax(num_zeros).item()
    res = torch.cat((inputs[:, 0:target_col], inputs[:, target_col + 1:]), dim=1)
    return res

if __name__ == '__main__':
    print('=== Test Case 1 ===')
    inputs = [
        [1, 0, 3],
        [4, 5, 6],
    ]
    inputs = torch.tensor(inputs)
    res = drop_column(inputs)
    print(res)
    print()

    print('=== Test Case 2 ===')
    inputs = [
        [1, 0, 0],
        [0, 5, 0],
    ]
    inputs = torch.tensor(inputs)
    res = drop_column(inputs)
    print(res)
    print()

    print('=== Test Case 3 ===')
    inputs = [
        [0, 0, 7, 8],
        [0, 0, 1, 5],
        [3, 0, 5, 0],
        [9, 5, 4, 0],
    ]
    inputs = torch.tensor(inputs)
    res = drop_column(inputs)
    print(res)
    print()
