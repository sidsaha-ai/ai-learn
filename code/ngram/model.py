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

        # map each letter to a number
        self.ltoi: dict = {}
        self.itol: dict = {}
        self._make_mappings()
    
    def _make_mappings(self) -> None:
        letters: list = ['.'] + list(string.ascii_lowercase)

        for ix, l in enumerate(letters):
            self.ltoi[l] = ix
            self.itol[ix] = l

    def train(self, num_epcohs: int) -> None:
        """
        This method will train the model.
        """
        print(f'{num_epcohs=}')

    def predict(self) -> None:
        """
        This methods predicts a word from the trained model.
        """
        return
