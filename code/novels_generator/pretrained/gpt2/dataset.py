import os

from novels_generator.code.epub_reader import EPubReader
from novels_generator.pretrained.gpt2.tokenizer import BooksTokenizer
from torch.utils.data import Dataset
from torch import Tensor

class BooksDataset(Dataset):

    def __init__(self, tokenizer: BooksTokenizer, folder: str = 'train') -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.folder = folder

        self.data = []
        self.build()
    
    def read_books(self) -> list:
        books = []
        reader = EPubReader()

        path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(path, 'data')
        path = os.path.join(path, self.folder)

        for f in os.listdir(path):
            if not f.endswith('.epub'):
                continue

            filepath = os.path.join(path, f)
            book_content = reader.read(filepath)
            if book_content:
                books.append(book_content)
        
        return books
    
    def build(self) -> None:
        # read all books
        books = self.read_books()

        for book_content in books:
            sequences = self.tokenizer.encode_into_sequences(book_content)
            self.data += sequences
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, ix) -> Tensor:
        return self.data[ix]
