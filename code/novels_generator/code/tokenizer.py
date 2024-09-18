"""
This is the tokenizer class that we will use to train a BPE tokenizer on our data and then use it to feed to the neural net.
"""

import os

from novels_generator.code.constants import Hyperparamters, SpecialTokens
from novels_generator.code.epub_reader import EPubReader
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

    def train(self, book_texts: list, vocab_size: int = None) -> None:
        """
        Method to take the entire texts of the books and train the tokenizer.
        """
        vocab_size = vocab_size or Hyperparamters.VOCAB_SIZE

        # pre-process
        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            special_tokens=[
                SpecialTokens.UNKNOWN,
                SpecialTokens.CHAPTER_NAME_START, SpecialTokens.CHAPTER_NAME_END,
                SpecialTokens.HEADING_START, SpecialTokens.HEADING_END,
                SpecialTokens.PARAGRAPH_START, SpecialTokens.PARAGRAPH_END,
                SpecialTokens.START, SpecialTokens.END,
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
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decodes the token IDs back to text.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def encode_into_sequences(self, text: str, context_length: int = None) -> list:
        """
        Method to encode the text into sequences of the context length.
        """
        context_length = context_length or Hyperparamters.CONTEXT_LENGTH

        encoded = self.tokenizer.encode(text)

        sequences = []
        for ix in range(0, len(encoded.ids), context_length):
            s = encoded.ids[ix:ix + context_length]
            if len(s) < context_length:
                # pad if smaller than context length
                s += [self.tokenizer.token_to_id(SpecialTokens.PAD)] * (context_length - len(s))
            sequences.append(s)

        return sequences


class BPETokenizerUtils:
    """
    Utils class to init and train the tokenizer.
    """

    @classmethod
    def read_train_books(cls):
        """
        Read all the training books.
        """
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(path, 'data')
        path = os.path.join(path, 'train')

        book_contents = []
        reader = EPubReader()

        for f in os.listdir(path):
            if not f.endswith('.epub'):
                continue

            filepath = os.path.join(path, f)
            content = reader.read(filepath)
            if not content:
                continue

            book_contents.append(content)

        return book_contents

    @classmethod
    def init(cls) -> BPETokenizer:
        """
        Initializes the tokenizer, trains it, and returns it.
        """
        tokenizer = BPETokenizer()
        tokenizer.train(cls.read_train_books())

        return tokenizer
