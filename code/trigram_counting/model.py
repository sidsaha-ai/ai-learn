import torch
import string
from torch import Tensor

class TrigramCountingModel:

    def __init__(self, input_words: list) -> None:
        self.input_words: list = input_words

        self.ltoi: dict = {}
        self.itol: dict = {}
        self._make_mappings()

        # init a 3D model with zeros
        size: int = len(self.ltoi)
        self.model: Tensor = torch.zeros((size, size, size), dtype=torch.float)
    
    def _make_mappings(self) -> None:
        # function to make mappings between letters and integers
        letters: dict = ['.'] + list(string.ascii_lowercase)

        for index, letter in enumerate(letters):
            self.ltoi[letter] = index
            self.itol[index] = letter
    
    def train(self) -> None:
        """
        Method to train the model.
        """
        size: int = len(self.ltoi)
        counts_model: Tensor = torch.zeros((size, size, size), dtype=torch.int)

        # make the trigrams
        for word in self.input_words:
            # two delimiters in the beginning and one delimiter at the end
            word = f'..{word}.'
            for l1, l2, l3 in zip(word, word[1:], word[2:]):
                l1_index: int = self.ltoi.get(l1)
                l2_index: int = self.ltoi.get(l2)
                l3_index: int = self.ltoi.get(l3)

                counts_model[l1_index, l2_index, l3_index] += 1
        
        # now that the counts are created, take the probability of the counts
        # across the 3rd dimension (which represents the counts)
        self.model = counts_model.float().div(counts_model.sum(dim=2, keepdims=True))
        self.model[torch.isnan(self.model)] = 0
    
    def predict(self) -> str:
        word: str = ''
        l1: str = '.'  # the first letter of the trigram
        l2: str = '.'  # the second letter of the trigram

        while True:
            ix1: int = self.ltoi.get(l1)
            ix2: int = self.ltoi.get(l2)
            
            pred_ix: int = torch.multinomial(
                self.model[ix1, ix2], num_samples=1, replacement=True,
            ).item()
            pred_l: str = self.itol.get(pred_ix)
            
            if pred_l == '.':
                break

            word = f'{word}{pred_l}'

            # change current trigram
            l1 = l2
            l2 = pred_l
        
        return word

    def loss(self) -> float:
        """
        This method finds the loss across the input data.
        """
        loss: float = 0
        num: int = 0

        for word in self.input_words:
            word = f'..{word}.'

            for l1, l2, l3 in zip(word, word[1:], word[2:]):
                ix1: int = self.ltoi.get(l1)
                ix2: int = self.ltoi.get(l2)
                ix3: int = self.ltoi.get(l3)

                loss += torch.log(self.model[ix1, ix2, ix3])
                num += 1
        
        loss = (-1) * loss  # negative log
        loss = loss / num

        return loss