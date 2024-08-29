"""
This file implements the trigram model using neural network.
"""

import string
import torch
from torch import Tensor
from torch.nn import functional as F


class TrigramNN:

    def __init__(self, input_words: list) -> None:
        self.input_words: list = input_words
        
        # mapping between a character and a corresponding integer
        self.ltoi: dict = {}
        self.itol: dict = {}
        self._make_char_int_mappings()

        # the inputs and targets tensor for training the neural network
        self.inputs: Tensor = None
        self.targets: Tensor = None
    
    def _make_char_int_mappings(self) -> None:
        letters: list = ['.'] + list(string.ascii_lowercase)

        for ix, l in enumerate(letters):
            self.ltoi[l] = ix
            self.itol[ix] = l
    
    def _make_training_data(self) -> None:
        """
        Make trigrams with 2 input letters and 1 target letter.
        """
        l1_inputs: list = []  # first letter input
        l2_inputs: list = []  # second letter input
        targets: list = []  # corresponding targets

        for word in self.input_words:
            word = f'..{word}.'
            
            for l1, l2, l3 in zip(word, word[1:], word[2:]):
                l1_inputs.append(self.ltoi.get(l1))
                l2_inputs.append(self.ltoi.get(l2))
                targets.append(self.ltoi.get(l3))
        
        # make targets as one-hot encoded tensor
        self.targets = F.one_hot(
            torch.tensor(targets), num_classes=len(self.ltoi),
        )

        # make l1_input and l2_input as one hot encoded tensors
        t_l1_inputs: Tensor = F.one_hot(
            torch.tensor(l1_inputs), num_classes=len(self.ltoi),
        )
        t_l2_inputs: Tensor = F.one_hot(
            torch.tensor(l2_inputs), num_classes=len(self.ltoi),
        )
        # concat both the tensor to make the input tensor
        self.inputs = torch.cat(
            (t_l1_inputs, t_l2_inputs), dim=1,
        )

    def train(self) -> None:
        """
        This method trains the neural network.
        """
        self._make_training_data()