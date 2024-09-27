"""
This solves the exercise problem https://tensorgym.com/exercises/19
"""
import torch


def rotate(A: torch.Tensor, axes: str) -> torch.Tensor:
    """
    This function rotates the supplied tensor once along the axes passed.
    """
    if axes == 'x':
        return torch.rot90(A, 1, (1, 2))

    if axes == 'y':
        return torch.rot90(A, 1, (0, 2))

    if axes == 'z':
        return torch.rot90(A, 1, (0, 1))

    raise Exception('axes is incorrect')


def generate_rotations(A: torch.Tensor) -> list[torch.Tensor]:
    """
    This functions generates all the rotations of the tensor supplied.
    """
    rotations: list[torch.Tensor] = []

    current = A
    for _ in range(4):
        for _ in range(4):
            for _ in range(4):
                rotations.append(current)
                current = rotate(current, 'z')
            current = rotate(current, 'y')
        current = rotate(current, 'x')

    return rotations


def optimal_rotation(A: torch.tensor, B: torch.Tensor) -> torch.Tensor:
    """
    This function implements the solution to this exercise.
    """
    max_dot_val = float('-inf')
    rotations: list[torch.Tensor] = generate_rotations(A)

    for rot in rotations:
        dot_prod = torch.sum(rot * B).item()
        max_dot_val = max(max_dot_val, dot_prod)

    return torch.tensor(max_dot_val)


def main():
    """
    The main function that runs the test cases.
    """
    size = (3, 3, 3)
    A = torch.randint(0, size[0] * size[1], size)
    B = torch.randint(0, size[0] * size[1], size)
    res = optimal_rotation(A, B)
    print(res)


if __name__ == '__main__':
    main()
