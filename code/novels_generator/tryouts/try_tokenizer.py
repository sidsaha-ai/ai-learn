"""
Test script to try out the BPE Tokenizer class.
"""
import os

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
    """
    The main method to start the execution of this script.
    """
    book_texts: list = read_books()

    tokenizer = BPETokenizer()
    tokenizer.train(book_texts)

    # First case
    text: str = 'Hi, how are you doing?'
    output = tokenizer.encode(text)
    sequences = tokenizer.encode_into_sequences(text, context_length=5)
    print(f'Tokens: {output.tokens}')
    print(f'IDs: {output.ids}')
    print(f'Sequences: {sequences}')

    print()

    text = 'She thought to herself how much she wanted how much she wanted him, even though he gets on her nerves. She wanted to take him!'
    output = tokenizer.encode(text)
    sequences = tokenizer.encode_into_sequences(text, context_length=10)
    print(f'Tokens: {output.tokens}')
    print(f'IDs: {output.ids}')
    print(f'Sequences: {sequences}')

    print()

    text = "Hi, I'm Sid, the owner of this project."
    output = tokenizer.encode(text)
    sequences = tokenizer.encode_into_sequences(text, context_length=3)
    print(f'Tokens: {output.tokens}')
    print(f'IDs: {output.ids}')
    print(f'Sequences: {sequences}')


if __name__ == '__main__':
    main()
