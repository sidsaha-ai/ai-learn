"""
This contains a class that reads an ePub book.
"""
import ebooklib
from ebooklib import epub

import os


class EPubReader:

    def __init__(self) -> None:
        super().__init__()
    
    def read(self, filepath: str) -> str:
        """
        Accpets a file path of an epub file, reads, and returns its content.
        """
        book = epub.read_epub(filepath)
        text: list = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT and item.is_chapter():
                text.append(
                    item.get_content().decode('utf-8'),
                )
        
        return ' '.join(text)


if __name__ == '__main__':
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(path, 'data')

    files = [f for f in os.listdir(path) if f.endswith('.epub')]
    filepath = os.path.join(path, files[0])

    reader = EPubReader()
    book = reader.read(filepath)

    print(f'Book length: {len(book):,}')
