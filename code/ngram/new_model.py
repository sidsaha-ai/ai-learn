"""
Build the model by leveraging the SDK.
"""
import string

import matplotlib.pyplot as plt
import torch
from ngram.dataset import Dataset
from ngram.encoder import Encoder
from sdk.batch_norm import BatchNorm
from sdk.embeddings import Embedding
from sdk.linear import Linear
from sdk.tanh import Tanh
from torch import Tensor
from torch.nn import functional as F


class NewNgramModel:
    """
    Model class that leverages the SDK.
    """
    def __init__(self, input_words: list[str], context_length: int) -> None:
        self.input_words: list[str] = input_words
        self.context_length: int = context_length

        self.encoder = Encoder()
        self.dataset = Dataset(
            input_words=input_words, context_length=context_length,
        )

        print(f'{self.dataset.train_inputs.shape=}')
        print(f'{self.dataset.train_targets.shape=}')

        print(f'{self.dataset.dev_inputs.shape=}')
        print(f'{self.dataset.dev_targets.shape=}')

        print(f'{self.dataset.test_inputs.shape=}')
        print(f'{self.dataset.test_targets.shape=}')
