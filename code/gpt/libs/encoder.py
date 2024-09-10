"""
This implements an encoder.
"""

class Encoder:

    def __init__(self, vocab: list[str]) -> None:
        super().__init__()

        self.vocab = vocab
        
        self.build()
    
    def build(self) -> None:
        self.ltoi: dict = {}
        self.itol: dict = {}

        for ix, el in enumerate(self.vocab):
            self.ltoi[el] = ix
            self.itol[ix] = el
    
    def encode(self, el: str) -> int:
        return self.ltoi.get(el)

    def decode(self, ix: int) -> str:
        return self.itol.get(ix)
