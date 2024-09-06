"""
Build the model by leveraging the SDK.
"""
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from torch.nn import functional as F

from sdk.embeddings import Embedding
from sdk.linear import Linear
from sdk.batch_norm import BatchNorm
from sdk.tanh import Tanh
import string

from ngram.encoder import Encoder


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