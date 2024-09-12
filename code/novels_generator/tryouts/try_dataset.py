"""
This script is for testing Dataset.
"""

import os

from novels_generator.code.dataset import BooksDataset
from novels_generator.code.epub_reader import EPubReader
from novels_generator.code.tokenizer import BPETokenizer


def read_books() -> list:
    """
    This reads all the epub books and returns their contents.
    """
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(path, 'data')
    path = os.path.join(path, 'train')

    book_texts: list = []
    reader = EPubReader()

    for f in os.listdir(path):
        if not f.endswith('.epub'):
            continue

        filepath = os.path.join(path, f)
        book_content = reader.read(filepath)
        if book_content:
            book_texts.append(book_content)

    return book_texts

def main():
    tokenizer = BPETokenizer()
    tokenizer.train(read_books())

    dataset = BooksDataset(tokenizer)
    print(f'Total Length: {len(dataset)}')

    print(f'Index 10: {dataset[10]}, Shape: {dataset[10].shape}')
    print(f'Index 50: {dataset[50]}, Shape: {dataset[50].shape}')


if __name__ == '__main__':
    main()
