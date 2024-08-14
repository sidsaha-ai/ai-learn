"""
This defines the neural network class.
"""
from torch import Tensor, nn


class HousePricesNN(nn.Module):
    """
    The neural network class the defines the architecture and the forward pass mechanism.
    """

    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(33, 500)
        self.l2 = nn.Linear(500, 1000)
        self.l3 = nn.Linear(1000, 200)
        self.l4 = nn.Linear(200, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the mechanism of the forward pass.
        """
        y = self.l1(x).clamp(min=0)
        y = self.l2(y).clamp(min=0)
        y = self.l3(y).clamp(min=0)
        y = self.l4(y)
        
        return y
