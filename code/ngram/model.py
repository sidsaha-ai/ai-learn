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

    def __init__(self, input_words: list, batch_size: int) -> None:
        self.input_words: list = input_words
        self.batch_size = batch_size

        # map each letter to a number
        self.ltoi: dict = {}
        self.itol: dict = {}
        self._make_mappings()

        # make input and output n-grams
        ngrams: list[tuple] = self._make_ngrams()
        for n_gram in ngrams:
            print(f'{n_gram}')
    
    def _make_mappings(self) -> None:
        """
        This method makes mappings between each letter and a corresponding number and vice-versa.
        """
        letters: list = ['.'] + list(string.ascii_lowercase)

        for ix, l in enumerate(letters):
            self.ltoi[l] = ix
            self.itol[ix] = l
        
    def _make_ngrams(self) -> list[tuple]:
        """
        Creates n-gram inputs and targets. For e.g., let's say a word is "emma", then for a batch size of 2,
        the n-grams would be like - 
        [
            (.., e),
            (.e, m),
            (em, m),
            (mm, a),
            (ma, .)
        ]
        """
        res: list[tuple] = []

        for word in self.input_words[0:2]:
            word = '.' * self.batch_size + word + '.'
            for i in range(len(word) - self.batch_size):
                inputs = word[i:i + self.batch_size]
                targets = word[i + self.batch_size]
                res.append(
                    (inputs, targets),
                )
        
        return res

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
