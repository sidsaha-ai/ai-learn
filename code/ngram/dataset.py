"""
This builds the train, dev, and test dataset.
"""

import torch
from torch import Tensor

from ngram.encoder import Encoder
import random

class Dataset:
    """
    This class builds the dataset.
    """

    def __init__(self, *, input_words: list[str], context_length: int) -> None:
        super().__init__()

        self.input_words = input_words
        self.context_length = context_length

        self.train_inputs, self.train_targets = None, None
        self.dev_inputs, self.dev_targets = None, None
        self.test_inputs, self.test_targets = None, None

        self.build()
    
    def make_tuples(self) -> tuple[list, list]:
        inputs, targets = [], []

        for word in self.input_words:
            letters = ['.'] * self.context_length + list(word) + ['.']

            for ix in range(0, len(letters) - self.context_length):
                current_inputs = letters[ix:ix + self.context_length]
                current_targets = letters[ix + self.context_length]

                inputs.append(current_inputs)
                targets.append(current_targets)
        
        return inputs, targets
    
    def make_data(self) -> tuple[list, list]:
        inputs, targets = self.make_tuples()

        encoder = Encoder()
        inputs = [
            [encoder.encode(letter) for letter in current_input]
            for current_input in inputs
        ]
        targets = [
            encoder.encode(letter) for letter in targets
        ]

        return inputs, targets

    def build(self) -> None:
        inputs, targets = self.make_data()

        assert len(inputs) == len(targets)

        indexes = [ix for ix in range(0, len(inputs))]
        random.shuffle(indexes)

        # training set 80%
        train_end = int(0.8 * len(indexes))
        dev_end = int(0.9 * len(indexes))

        # training set
        self.train_inputs = torch.tensor(inputs[0:train_end])
        self.train_targets = torch.tensor(targets[0:train_end])

        # dev set
        self.dev_inputs = torch.tensor(inputs[train_end:dev_end])
        self.dev_targets = torch.tensor(targets[train_end:dev_end])

        # test set
        self.test_inputs = torch.tensor(inputs[dev_end:])
        self.test_targets = torch.tensor(targets[dev_end:])
