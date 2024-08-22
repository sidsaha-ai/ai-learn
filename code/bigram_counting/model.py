"""
This file contains the class the defines the bigram language model.
"""
import string


class BigramLM:
    """
    This class defines the model for the Bigram Language Model (LM).
    """

    def __init__(self, input_words: list) -> None:
        super().__init__()
        self.input_words = input_words

        self.ltoi: dict = {}  # map of letter to integer
        self.itol: dict = {}  # corresponding map of integer to letter

        self._make_stoi()

    def _make_stoi(self) -> None:
        """
        Makes a mapping between each letter of the alphabet and an integer. This is required for
        indexing into the tensors for corresponding letters.
        """
        # the first letter is the delimiter character
        letters: list = ['.']
        
        # add all lowercase letters
        letters += [letter for letter in string.ascii_lowercase]
        
        for index, letter in enumerate(letters):
            self.ltoi[letter] = index
            self.itol[index] = letter
        
    
    def train(self) -> None:
        """
        This method trains the model and creates the tensor with the probabilities on pair-wise characters.
        """