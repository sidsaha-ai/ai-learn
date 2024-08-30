"""
This contains the model implementation for the N-gram character model.
"""

import string

import torch
from torch import Tensor
from torch.nn import functional as F


class NGramModel:
    """
    This is the model implementation for an N-gram character model.
    """
    
    def __init__(self, input_words: list) -> None:
        self.input_words: list = input_words
        print(f'Num of words: {len(self.input_words)}')
    
    def train(self, num_epcohs: int) -> None:
        """
        This method will train the model.
        """
        print(f'{num_epcohs=}')
