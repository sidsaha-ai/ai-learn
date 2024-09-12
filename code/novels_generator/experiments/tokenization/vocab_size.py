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

2. UNK Token Count - Pick the same book, encode, and compare the number of `<UNK>` tokens in each setup to see how well
rare words are handled.

3. Token Diversity - After each encoding, count how many distinct tokens were used in the book to understand the
diversity and utility of the vocabulary.

5. UNK Token Positioning - Analyze where `<UNK>` tokens occur to determine if any critical or common terms are being represented by `<UNK>`.
"""

from novels_generator.code.tokenizer import BPETokenizer
from matplotlib import pyplot as plt
import os
from novels_generator.code.epub_reader import EPubReader

def read_books() -> dict[str, str]:
    books_data: dict[str, str] = {}  # name vs content

    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(path, 'data')

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

def build_tokenizers(books_data: dict[str, str]) -> dict[str, BPETokenizer]:
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

def plot_vocab_size_vs_num_tokens(data: dict) -> None:
    x = list(data.keys())
    book_names = list(data.get(x[0]).keys())

    for name in book_names:
        num_tokens = [
            d.get(name) for _, d in data.items()
        ]
        plt.plot(x, num_tokens, label=name)
    
    plt.xlabel('Vocab size')
    plt.ylabel('Num tokens')
    plt.title('Num tokens vs Vocab sizes')
    plt.legend()
    plt.show()

def expt_token_count(tokenizers: dict[str, BPETokenizer], books_data: dict[str, str]) -> None:
    data: dict = {}

    for tokenizer_size, tokenizer in tokenizers.items():
        current_data: dict = {}

        for book_name, book_content in books_data.items():
            encoded_book_content = tokenizer.encode(book_content)
            num_tokens = len(encoded_book_content.tokens)
            current_data[book_name] = num_tokens
        
        data[tokenizer_size] = current_data
    
    # visualization in multiple ways
    plot_vocab_size_vs_num_tokens(data)
    
def main():
    books_data: dict[str, str] = read_books()
    tokenizers: dict[str, BPETokenizer] = build_tokenizers(books_data)

    expt_token_count(tokenizers, books_data)


if __name__ == '__main__':
    main()
