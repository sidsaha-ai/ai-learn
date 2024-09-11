"""
This contains a class that reads an ePub book.
"""
import os

import bs4
import ebooklib
from ebooklib import epub

from novels_generator.code.constants import SpecialTokens


class EPubReader:
    """
    Class to read epub files.
    """

    def clean(self, content) -> str:
        """
        Processes the content of each chapter.
        """
        soup = bs4.BeautifulSoup(content, features='xml')
        section = soup.body.section
        text_parts = []

        for el in section.contents:
            if isinstance(el, bs4.element.Tag):
                match el.name:
                    # add special tokens so that tokenizers can handle them.
                    case 'h1':
                        text_parts.append(SpecialTokens.CHAPTER)
                        text_parts.append('\n\n')
                        text_parts.append(el.get_text(strip=True))
                        text_parts.append('\n\n')
                    case 'h2':
                        text_parts.append(SpecialTokens.HEADING)
                        text_parts.append(el.get_text(strip=True))
                        text_parts.append('\n\n')
                    case 'p':
                        text_parts.append(SpecialTokens.PARAGRAPH)
                        text_parts.append(el.get_text(strip=True))
                        text_parts.append('\n')

            elif isinstance(el, bs4.element.NavigableString):
                text_parts.append(el)

        return ''.join(text_parts)

    def preprocess(self, content) -> str:
        """
        Preprocess the content before cleaning.
        """
        soup = bs4.BeautifulSoup(content, 'html.parser')

        # remove class attributes from all tags
        for tag in soup.find_all(True):  # `True` argument finds all tags
            if 'class' in tag.attrs:
                del tag.attrs['class']

        # handle span tags within p tags
        for p in soup.find_all('p'):
            for span in p.find_all('span'):
                span.unwrap()

        return str(soup)

    def read(self, epub_filepath: str) -> str:
        """
        Accpets a file path of an epub file, reads, and returns its content.
        """
        book = epub.read_epub(epub_filepath)
        text: list = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT and item.is_chapter() and 'chapter' in item.get_name().lower():
                content = item.get_content().decode('utf-8')
                content = self.preprocess(content)
                content = self.clean(content)
                text.append(content)

        # token to mark novel end
        text.append(SpecialTokens.END)

        return ''.join(text)


if __name__ == '__main__':

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(path, 'data')

    files = [f for f in os.listdir(path) if f.endswith('.epub')]
    filepath = os.path.join(path, files[0])

    reader = EPubReader()
    book_text = reader.read(filepath)

    print(f'Book: {os.path.basename(filepath)}. Book length: {len(book_text):,}')
