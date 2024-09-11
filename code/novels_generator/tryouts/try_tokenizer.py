import os

from novels_generator.code.epub_reader import EPubReader
from novels_generator.code.tokenizer import BPETokenizer


def read_books() -> list:
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(path, 'data')

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
    book_texts: list = read_books()

    tokenizer = BPETokenizer()
    tokenizer.train(book_texts)

    output = tokenizer.encode('Hi, how are you doing?')
    print(output.tokens)
    print(output.ids)

    print()

    output = tokenizer.encode('My name is Siddharth Saha!')
    print(output.tokens)
    print(output.ids)


if __name__ == '__main__':
    main()
