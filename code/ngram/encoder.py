"""
This file contains the letter encoding mapper class.
"""

import string

class Encoder:

    def __init__(self) -> None:
        super().__init__()

        self.ltoi: dict = {}
        self.itol: dict = {}

        universe: list = ['.'] + list(string.ascii_lowercase)
        for index, letter in enumerate(universe):
            self.ltoi[letter] = index
            self.itol[index] = letter
    
    def encoded_val(self, letter: str) -> int:
        return self.ltoi.get(letter)
    
    def decoded_val(self, code: int) -> str:
        return self.itol.get(code)
