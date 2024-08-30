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

        self.input_words = self.input_words[0:10]  # TODO: remove this

        # make input and targets
        inputs, targets = self._make_inputs_and_targets()
        print(f'{inputs.shape=}')
        print(f'{targets.shape=}')
    
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

        for word in self.input_words:
            # pad appropriately at the beginning with dots
            word = ('.' * self.batch_size) + word + '.'

            for i in range(len(word) - self.batch_size):
                inputs = word[i:i + self.batch_size]
                targets = word[i + self.batch_size]
                res.append(
                    (inputs, targets),
                )
        
        return res
    
    def _make_inputs_and_targets(self) -> tuple[Tensor, Tensor]:
        """
        Creates an input and output tensor with integer mappings of letters.
        """
        ngrams: list[tuple] = self._make_ngrams()
        inputs: list = []
        targets: list = []

        for input_ngram, target_letter in ngrams:
            inputs.append(
                [self.ltoi.get(l) for l in input_ngram],
            )
            targets.append(
                self.ltoi.get(target_letter),
            )

        t_inputs: Tensor = torch.tensor(inputs)
        t_targets: Tensor = torch.tensor(targets)

        return t_inputs, t_targets    

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
