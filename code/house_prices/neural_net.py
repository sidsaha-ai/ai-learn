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
            nn.Linear(333, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
        ]
        self.output_layer = nn.Linear(256, 1)
        self.activate = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the mechanism of the forward pass.
        """
        for layer in self.layers:
            x = self.activate(layer(x).clamp(min=0))
        x = self.output_layer(x)
        return x
