from novels_generator.code.epub_reader import EPubReader

import os

def main():
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(path, 'data')

    files = [f for f in os.listdir(path) if f.endswith('.epub')]
    filepath = os.path.join(path, files[0])

    reader = EPubReader()
    book_text = reader.read(filepath)

    print(book_text)
    print(f'Book: {os.path.basename(filepath)}. Book Length: {len(book_text):,}')


if __name__ == '__main__':
    main()
