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

        # define the layers, the input layer has 333 features
        self.layers: list = [
            nn.Linear(94, 666),
            nn.Linear(666, 333),
            nn.Linear(333, 166),
            nn.Linear(166, 83),
            nn.Linear(83, 40),
            nn.Linear(40, 20),
            nn.Linear(20, 10),
            nn.Linear(10, 5),
        ]
        self.output_layer = nn.Linear(5, 1)
        self.activate = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the mechanism of the forward pass.
        """
        for layer in self.layers:
            x = self.activate(layer(x))
        x = self.output_layer(x)
        return x
