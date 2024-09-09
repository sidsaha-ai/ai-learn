"""
Flatten class implementation.
"""
from torch import Tensor

class Flatten:

    def __init__(self) -> None:
        super().__init__()
        
        self.input = None
        self.output = None
        self.training = True
    
    def __call__(self, inputs: Tensor) -> Tensor:
        self.input = inputs
        self.output = inputs.view(inputs.shape[0], -1)
        
        return self.output
    
    def parameters(self) -> list:
        return []
