"""
This file contains the letter encoding mapper class.
"""

import string


class Encoder:
    """
    Encoder to encode letters to integers and integers back to letters.
    """

    def __init__(self) -> None:
        super().__init__()

        self.ltoi: dict = {}
        self.itol: dict = {}

        universe: list = ['.'] + list(string.ascii_lowercase)
        for index, letter in enumerate(universe):
            self.ltoi[letter] = index
            self.itol[index] = letter

    def encode(self, letter: str) -> int:
        """
        Returns the encoded value of the letter.
        """
        return self.ltoi.get(letter)

    def decode(self, code: int) -> str:
        """
        Returns the decoded letter value.
        """
        return self.itol.get(code)
