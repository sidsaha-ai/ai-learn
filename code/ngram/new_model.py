"""
Build the model by leveraging the SDK.
"""

from ngram.dataset import Dataset
from ngram.encoder import Encoder


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
