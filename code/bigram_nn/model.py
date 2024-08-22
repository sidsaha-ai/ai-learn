import string

import torch
from torch.nn import functional as F


class BigramNN:
    
    def __init__(self, input_words: list) -> None:
        super().__init__()
        self.input_words = input_words

        self.ltoi: dict = {}  # map of letter to integer
        self.itol: dict = {}  # map of integer to letter

        self._make_ltoi()
    
    def _make_ltoi(self) -> None:
        letters: list = ['.']

        letters += list(string.ascii_lowercase)

        for index, letter in enumerate(letters):
            self.ltoi[letter] = index
            self.itol[index] = letter
    
    def _make_inputs_and_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        inputs: list = []
        targets: list = []

        for word in self.input_words:
            word = f'.{word}.'  # add delimiter
            
            for l1, l2 in zip(word, word[1:]):
                inputs.append(self.ltoi.get(l1))
                targets.append(self.ltoi.get(l2))
        
        t_inputs: torch.Tensor = torch.tensor(inputs)
        t_targets: torch.Tensor = torch.tensor(targets)
        
        return t_inputs, t_targets
    
    def train(self) -> None:
        """
        This trains the model based on the `input_words`.
        """
        # make a list of input characters (to integers) that represents the first letter of the bigram
        # make a list of target characters (to integers) that represents the second letter of the bigram
        inputs: torch.Tensor = None
        targets: torch.Tensor = None
        inputs, targets = self._make_inputs_and_targets()

        # in the `inputs` and `targets`, the indices actually represent letters and we are going to predict letters
        # as output. Letters can be considered to be categorical data, in that sense (and not numerical data).
        # so, we will one-hot encode the inputs and targets
        
        # `num_classes` in the number of classes possible, so the universe of letters
        inputs = F.one_hot(inputs, num_classes=len(self.ltoi))
        targets = F.one_hot(targets, num_classes=len(self.ltoi))
        print(f'{inputs.dtype=}')
        print(f'{inputs.shape=}')
        print(f'{targets.dtype=}')
        print(f'{targets.shape=}')
