"""
This file implements the trigram model using neural network.
"""

import string
import torch
from torch import Tensor


class TrigramNN:

    def __init__(self, input_words: list) -> None:
        self.input_words: list = input_words
        
        # mapping between a character and a corresponding integer
        self.ltoi: dict = {}
        self.itol: dict = {}
        self._make_char_int_mappings()
    
    def _make_char_int_mappings(self) -> None:
        letters: list = ['.'] + list(string.ascii_lowercase)

        for ix, l in enumerate(letters):
            self.ltoi[l] = ix
            self.itol[ix] = l
    
    def _make_trigrams(self) -> tuple[list, list]:
        """
        Make trigrams with 2 input letters and 1 target letter.
        """
        inputs: list = []  # 2-letter inputs
        targets: list = []  # corresponding outputs

        for word in self.input_words:
            word = f'..{word}.'
            
            for l1, l2, l3 in zip(word, word[1:], word[2:]):
                inputs.append(f'{l1}{l2}')
                targets.append(l3)
        
        return inputs, targets

    def train(self) -> None:
        """
        This method trains the neural network.
        """
        inputs, targets = self._make_trigrams()

        ix = 0
        for input, target in zip(inputs, targets):
            ix += 1
            if ix >= 10:
                break
            print(f'{input} : {target}')
