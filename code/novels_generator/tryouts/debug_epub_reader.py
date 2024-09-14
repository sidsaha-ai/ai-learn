"""
This script is used to debug the reading of epubs.
"""

import argparse
import os

from novels_generator.code.epub_reader import EPubReader


def main(book_name: str):
    """
    The main method that does the execution.
    """
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(path, 'data')
    path = os.path.join(path, 'train')

    files = [f for f in os.listdir(path) if f.endswith('.epub')]

    for f in files:
        if book_name is not None and book_name not in f:
            continue

        filepath = os.path.join(path, f)

        reader = EPubReader()
        print(f'=== Book {f} ===')
        reader.debug_chapter_names(filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--book_name', type=str,
    )
    args = parser.parse_args()

    main(args.book_name)
