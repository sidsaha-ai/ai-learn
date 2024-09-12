"""
Contains test script to try the EPUB reader.
"""
import argparse
import os

from novels_generator.code.epub_reader import EPubReader


def main(book_name: str):
    """
    Main method where the execution starts for this script.
    """
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(path, 'data')

    files = [f for f in os.listdir(path) if f.endswith('.epub')]
    filepath = os.path.join(path, files[0])
    if book_name:
        for f in files:
            if book_name in f:
                filepath = os.path.join(path, f)
                break

    reader = EPubReader()
    book_text = reader.read(filepath)

    print(book_text)
    print(f'Book: {os.path.basename(filepath)}. Book Length: {len(book_text):,}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--book_name', type=str,
    )
    args = parser.parse_args()

    main(args.book_name)
