"""
Implements the sequential container class to hold all the layers.
"""

from typing import Any
from torch import Tensor

class Sequential:

    def __init__(self, layers: list) -> None:
        super().__init__()

        self.layers = layers
        
        self.input = None
        self.output = None
        self.training = True
    
    def __call__(self, input: Tensor) -> Tensor:
        """
        Calls all the layers.
        """
        self.input = input

        x = input
        for layer in self.layers:
            x = layer(x)
        
        self.output = x

        return self.output
    
    def parameters(self) -> list:
        """
        Returns the parameters of all the layers.
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __setattr__(self, name, value) -> None:
        if name == 'training':
            for layer in self.layers:
                layer.training = value
        
        super().__setattr__(name, value)
