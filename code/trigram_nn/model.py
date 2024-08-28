"""
This file implements the trigram model using neural network.
"""

import torch
from torch import Tensor


class TrigramNN:

    def __init__(self, input_words: list) -> None:
        self.input_words: list = input_words
        print(f'Num input words: {len(self.input_words)}')
        