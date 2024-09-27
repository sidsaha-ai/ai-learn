"""
This implements the solution for the exercise https://tensorgym.com/exercises/13
"""
import torch

class SimpleMLP(torch.nn.Module):
    """
    This implements a simple MLP with 3 layers.
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.in_dim, self.in_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.in_dim // 2, self.in_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.in_dim // 2, self.out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass.
        """
        return self.model(x)


def compute_simple_mlp(x: torch.Tensor) -> torch.Tensor:
    """
    The exercise implementation function.
    """
    torch.manual_seed(0)

    in_dim: int = x.shape[1]
    out_dim: int = 2

    model = SimpleMLP(in_dim, out_dim)
    res = model(x)

    res = res.detach().clone()
    return res


if __name__ == '__main__':
    cases = []

    # test case 1
    inputs = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    ]
    inputs = torch.tensor(inputs)
    outputs = [
        [-0.0439, -0.1204],
    ]
    outputs = torch.tensor(outputs)
    cases.append({
        'inputs': inputs,
        'outputs': outputs,
    })

    # test case 2
    inputs =[
        [0.8, 0,7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    ]
    inputs = torch.tensor(inputs)
    outputs = [
        [0.5470, 0.6105],
    ]
    outputs = torch.tensor(outputs)
    cases.append({
        'inputs': inputs,
        'outputs': outputs,
    })

    for ix, c in enumerate(cases):
        res = compute_simple_mlp(c.get('inputs'))
        print(res)
        message = f'PASS: test case {ix + 1}' if torch.equal(res, c.get('outputs')) else f'FAIL: test case {ix + 1}'
        print(message)
