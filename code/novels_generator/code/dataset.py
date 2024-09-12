"""
This is the Dataset class that contains the entire training dataset.
"""

import os

import torch
from novels_generator.code.epub_reader import EPubReader
from novels_generator.code.tokenizer import BPETokenizer
from torch import Tensor
from torch.utils.data import Dataset


class BooksDataset(Dataset):
    """
    This is the books dataset.
    """

    def __init__(self, tokenizer: BPETokenizer) -> None:
        """
        The constructor takes the already trained tokenizer.
        """
        super().__init__()

        self.tokenizer = tokenizer

        # the full dataset. the dataset is an array of all the tokens of all the training books
        self.data = []
        self.build()

    def read_books(self) -> list:
        """
        Reads all the training books and returns the contents in a list.
        """
        books = []
        reader = EPubReader()

        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(path, 'data')
        path = os.path.join(path, 'train')

        for f in os.listdir(path):
            if not f.endswith('.epub'):
                continue

            filepath = os.path.join(path, f)
            book_content = reader.read(filepath)
            if book_content:
                books.append(book_content)

        return books

    def build(self) -> None:
        """
        Builds the internals of this dataset.
        """
        # read all the books
        books = self.read_books()

        # tokenize into sequences each book.
        for book_content in books:
            sequences = self.tokenizer.encode_into_sequences(book_content)
            self.data += sequences

    def __len__(self) -> int:
        """
        Returns the total length of the data.
        """
        return len(self.data)

    def __getitem__(self, ix) -> Tensor:
        """
        Returns the item at passed index from the data.
        """
        return torch.tensor(self.data[ix])
