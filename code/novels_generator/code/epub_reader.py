"""
This contains a class that reads an ePub book.
"""
import warnings

import bs4
import ebooklib
from ebooklib import epub
from novels_generator.code.constants import SpecialTokens

warnings.filterwarnings('ignore')


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
                        text_parts.append(SpecialTokens.CHAPTER_NAME_START)
                        text_parts.append(el.get_text(strip=True))
                        text_parts.append(SpecialTokens.CHAPTER_NAME_END)
                    case 'h2':
                        text_parts.append(SpecialTokens.HEADING_START)
                        text_parts.append(el.get_text(strip=True))
                        text_parts.append(SpecialTokens.HEADING_END)
                    case 'p':
                        text_parts.append(SpecialTokens.PARAGRAPH_START)
                        text_parts.append(el.get_text(strip=True))
                        text_parts.append(SpecialTokens.PARAGRAPH_END)

            elif isinstance(el, bs4.element.NavigableString):
                if el.strip():
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

    def is_chapter(self, item) -> bool:
        """
        Decides whether the item in the book is a chapter to be read or not.
        """
        if item.get_type() != ebooklib.ITEM_DOCUMENT or not item.is_chapter():
            return False

        item_name: str = item.get_name()

        if 'chapter' in item_name.lower():
            return True

        if 'part' in item_name.lower():
            return True

        return False

    def read(self, epub_filepath: str) -> str:
        """
        Accpets a file path of an epub file, reads, and returns its content.
        """
        book = epub.read_epub(epub_filepath)
        text: list = []

        for item in book.get_items():
            if self.is_chapter(item):
                content = item.get_content().decode('utf-8')
                content = self.preprocess(content)
                content = self.clean(content)
                text.append(content)

        # token to mark novel end
        text.append(SpecialTokens.END)

        return ''.join(text)

    # Debug methods
    # =============
    def debug_chapter_names(self, epub_filepath: str) -> str:
        """
        Prints all the chapter names that will be considered while reading.
        """
        book = epub.read_epub(epub_filepath)
        chapter_names = []
        non_chapter_names = []

        for item in book.get_items():
            if self.is_chapter(item):
                chapter_names.append(item.get_name())
            else:
                non_chapter_names.append(item.get_name())

        print('--- Chapter Names ---')
        for chapter in chapter_names:
            print(chapter)
        print('--- Non Chapter Names ---')
        for non_chapter in non_chapter_names:
            print(non_chapter)
