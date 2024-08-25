import torch
import string

class TrigramCountingModel:

    def __init__(self, input_words: list) -> None:
        self.input_words: list = input_words

        self.ltoi: dict = {}
        self.itol: dict = {}
        self._make_mappings()

        print(f'{self.ltoi=}')
        print(f'{self.itol=}')
    
    def _make_mappings(self) -> None:
        # function to make mappings between letters and integers
        letters: dict = ['.'] + list(string.ascii_lowercase)

        for index, letter in enumerate(letters):
            self.ltoi[letter] = index
            self.itol[index] = letter
    