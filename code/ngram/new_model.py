"""
Build the model by leveraging the SDK.
"""
import string

import matplotlib.pyplot as plt
import torch
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

        print(self.encoder.ltoi)
        print(self.encoder.itol)