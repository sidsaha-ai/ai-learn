"""
This is the solution to exercise https://tensorgym.com/exercises/3
"""
import torch

def solution(matrix: torch.Tensor) -> torch.Tensor:
    """
    This implements the solution function to the exercise.
    """
    res = matrix[::2]  # get every second row
    res = torch.sum(res, dim=1)
    return res

if __name__ == '__main__':
    print('=== Test Case 1 ===')
    matrix = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
    ]
    matrix = torch.tensor(matrix)
    res = solution(matrix)
    print(res)
    print()

    print('=== Test Case 2 ===')
    matrix = [
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
    ]
    matrix = torch.tensor(matrix)
    res = solution(matrix)
    print(res)
    print()
