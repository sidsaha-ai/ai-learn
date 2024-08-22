"""
This file contains the class the defines the bigram language model.
"""
import string
import torch


class BigramLM:
    """
    This class defines the model for the Bigram Language Model (LM).
    """

    def __init__(self, input_words: list) -> None:
        super().__init__()
        self.input_words = input_words

        self.ltoi: dict = {}  # map of letter to integer
        self.itol: dict = {}  # corresponding map of integer to letter
        self.model: torch.Tensor = None  # the model tensor

        self._make_stoi()
        self._init_tensor()

    def _make_stoi(self) -> None:
        """
        Makes a mapping between each letter of the alphabet and an integer. This is required for
        indexing into the tensors for corresponding letters.
        """
        # the first letter is the delimiter character
        letters: list = ['.']

        # add all lowercase letters
        letters += list(string.ascii_lowercase)

        for index, letter in enumerate(letters):
            self.ltoi[letter] = index
            self.itol[index] = letter
        
    def _init_tensor(self) -> None:
        size: int = len(self.ltoi)  # the number of letters

        # init the model with zeros
        self.model = torch.zeros((size, size), dtype=torch.int)

    def train(self) -> None:
        """
        This method trains the model and creates the tensor with the probabilities on pair-wise characters.
        """
        # count bigrams and add count to the model tensor
        for word in self.input_words:
            # add dot delimiter to thew w
            word = f'.{word}.'

            # iterate over the bigrams and increment count over the model tensor
            for l1, l2 in zip(word, word[1:]):
                l1_index: int = self.ltoi.get(l1)
                l2_index: int = self.ltoi.get(l2)
                self.model[l1_index, l2_index] += 1
        
        print(self.model)

    def predict(self) -> str:
        """
        This method predicts a word based on the trained model.
        """
        return
