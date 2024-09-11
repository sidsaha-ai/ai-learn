"""
This implements an encoder.
"""


class Encoder:
    """
    Encoder class to encode and decode.
    """

    def __init__(self, vocab: list[str]) -> None:
        super().__init__()

        self.vocab = vocab

        self.build()

    def build(self) -> None:
        """
        Build the encoder and decoder dictionaries.
        """
        self.ltoi: dict = {}
        self.itol: dict = {}

        for ix, el in enumerate(self.vocab):
            self.ltoi[el] = ix
            self.itol[ix] = el

    def encode(self, el: str) -> int:
        """
        Method to encode a letter to encoded integer.
        """
        return self.ltoi.get(el)

    def decode(self, ix: int) -> str:
        """
        Method to decode an integer to the letter.
        """
        return self.itol.get(ix)
