"""
This is the tokenizer class that we will use to train a BPE tokenizer on our data and then use it to feed to the neural net.
"""

from novels_generator.code.constants import SpecialTokens, Hyperparamters
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class BPETokenizer:
    """
    BPE Tokenizer that we will train for the model.
    """

    def __init__(self) -> None:
        self.tokenizer = Tokenizer(BPE(unk_token=SpecialTokens.UNKNOWN))

    def train(self, book_texts: list, vocab_size: int = 40000) -> None:
        """
        Method to take the entire texts of the books and train the tokenizer.
        """
        # pre-process
        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            special_tokens=[
                SpecialTokens.UNKNOWN,
                SpecialTokens.CHAPTER_NAME_START, SpecialTokens.CHAPTER_NAME_END,
                SpecialTokens.HEADING_START, SpecialTokens.HEADING_END,
                SpecialTokens.PARAGRAPH_START, SpecialTokens.PARAGRAPH_END,
                SpecialTokens.END,
                SpecialTokens.PAD,
            ],
            vocab_size=vocab_size,
        )

        # train
        self.tokenizer.train_from_iterator(book_texts, trainer=trainer)

    def encode(self, text: str):
        """
        Method to return the encoding of the passed text based on the tokenizer.
        """
        return self.tokenizer.encode(text)

    def encode_into_sequences(self, text: str, context_length: int = Hyperparamters.CONTEXT_LENGTH) -> list:
        """
        Method to encode the text into sequences of the context length.
        """
        encoded = self.tokenizer.encode(text)

        sequences = []
        for ix in range(0, len(encoded.ids), context_length):
            s = encoded.ids[ix:ix + context_length]
            if len(s) < context_length:
                # pad if smaller than context length
                s += [self.tokenizer.token_to_id(SpecialTokens.PAD)] * (context_length - len(s))
            sequences.append(s)

        return sequences