"""
Experiment Objective
====================

The objective of this experiement is to arrive at a vocab size to use in the BPE Tokenizer. The size of the vocabulary determines how many
unique tokens will be used to represent the text. A smaller vocabulary size will break down more words in sub-words and could help in
better generalization, but can lose context about potential rare and important words. A larger vocabulary size will keep more words
intact, but might increase model complexity.

Experiment Setup
================

Let's try with the following vocab sizes -

1. (Small) 10K tokens
2. (Small) 15K tokens
3. (Small) 20K tokens
4. (Medium) 30K tokensx
5. (Medium) 40K tokens
6. (Medium) 50K tokens
7. (Large) 60K tokens
8. (Large) 80K tokens
9. (Large) 100K tokens

What to measure
===============

For each vocab size, let's measure the following:

1. Token Count - Pick one book (and potentially a subset of books), encode, and compare the number of tokens in
each setup to gauge compression efficiency.

2. UNK Token Count - Pick an unknown book (not used in training), encode, and compare the number of `<UNK>` tokens in each setup to see how well
rare words are handled.

3. Token Diversity - After each encoding, count how many distinct tokens were used in the book to understand the
diversity and utility of the vocabulary.

5. UNK Token Positioning - Analyze where `<UNK>` tokens occur to determine if any critical or common terms are being represented by `<UNK>`.
"""

import os

from matplotlib import pyplot as plt
from novels_generator.code.constants import SpecialTokens
from novels_generator.code.epub_reader import EPubReader
from novels_generator.code.tokenizer import BPETokenizer


def read_train_books() -> dict[str, str]:
    """
    Read all the training books.
    """
    books_data: dict[str, str] = {}  # name vs content

    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(path, 'data')
    path = os.path.join(path, 'train')

    reader = EPubReader()

    for f in os.listdir(path):
        if not f.endswith('.epub'):
            continue

        filepath = os.path.join(path, f)
        book_content = reader.read(filepath)
        if not book_content:
            continue

        books_data[f] = book_content

    return books_data


def read_test_book() -> tuple[str, str]:
    """
    Read a test book.
    """
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(path, 'data')
    path = os.path.join(path, 'test')

    reader = EPubReader()

    for book_name in os.listdir(path):
        if not book_name.endswith('.epub'):
            continue

        filepath = os.path.join(path, book_name)
        book_content = reader.read(filepath)
        if not book_content:
            continue

        return book_name, book_content


def build_tokenizers(books_data: dict[str, str]) -> dict[int, BPETokenizer]:
    """
    Builds tokenizers of different vocab sizes.
    """
    res = {}

    vocab_sizes: list = [10000, 15000, 20000, 30000, 40000, 50000, 60000, 80000, 100000]
    for size in vocab_sizes:
        t = BPETokenizer()
        t.train(books_data.values(), vocab_size=size)

        res[size] = t

    return res


def expt_unknown_tokens(tokenizers: dict) -> None:
    """
    Runs the experiment on number of unknown tokens.
    """
    x = []
    num_unknown_tokens = []

    book_name, book_content = read_test_book()

    for vocab_size, tokenizer in tokenizers.items():
        encoded_book = tokenizer.encode(book_content)
        x.append(vocab_size)
        num_unknown_tokens.append(
            list(encoded_book.tokens).count(SpecialTokens.UNKNOWN),
        )

    plt.plot(x, num_unknown_tokens, label=book_name)
    plt.xlabel('Vocab size')
    plt.ylabel('Num unknown tokens')
    plt.title('Num unknown tokens vs Vocab size')
    plt.legend()
    plt.show()


def _encode_books(tokenizers: dict[int, BPETokenizer], books_data: dict[str, str]) -> dict:
    data: dict = {}

    for tokenizer_size, tokenizer in tokenizers.items():
        current_data: dict = {}

        for book_name, book_content in books_data.items():
            encoded_book_content = tokenizer.encode(book_content)
            current_data[book_name] = encoded_book_content

        data[tokenizer_size] = current_data
    
    return data


def expt_token_count(tokenizers: dict[int, BPETokenizer], books_data: dict[str, str]) -> None:
    """
    Runs the experiment for token count.
    """
    data: dict = _encode_books(tokenizers, books_data)

    x = list(data.keys())
    book_names = list(data.get(x[0]).keys())

    for name in book_names:
        num_tokens = [
            len(d.get(name).tokens) for _, d in data.items()
        ]
        plt.plot(x, num_tokens, label=name)

    plt.xlabel('Vocab size')
    plt.ylabel('Num tokens')
    plt.title('Num tokens vs Vocab sizes')
    plt.legend()
    plt.show()


def expt_token_diversity(tokenizers: dict[int, BPETokenizer], books_data: dict[str, str]) -> None:
    """
    Runs the experiment to view token diversity for each book.
    """
    data: dict = _encode_books(tokenizers, books_data)

    x = list(data.keys())
    book_names = list(data.get(x[0]).keys())

    for name in book_names:
        num_unique_tokens = [
            len(list(set(list(d.get(name).tokens)))) for _, d in data.items()
        ]
        plt.plot(x, num_unique_tokens, label=name)

    plt.xlabel('Vocab size')
    plt.ylabel('Num unique tokens')
    plt.title('Num unique tokens vs Vocab sizes')
    plt.legend()
    plt.show()


def main():
    """
    The main function where the execution starts.
    """
    books_data: dict[str, str] = read_train_books()
    tokenizers: dict[int, BPETokenizer] = build_tokenizers(books_data)

    expt_token_count(tokenizers, books_data)
    expt_unknown_tokens(tokenizers)
    expt_token_diversity(tokenizers, books_data)

    # Check out the viz. in the results folder.


if __name__ == '__main__':
    main()
    # pylint: disable=pointless-string-statement
    """
    Experiment Results
    ==================
    From the results, it looks like 30K - 40K is a good enough vocab size. The number of tokens,
    the number of unique tokens, and the number of unknown tokens all stabilize after this threshold. So,
    there is no point in increased the complexity by going beyond a vocab size of 40K.

    Result - Let's use a vocab size of 40K.
    """
    # pylint: enable=all
