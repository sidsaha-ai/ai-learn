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
        self._init_model_tensor()

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

    def _init_model_tensor(self) -> None:
        size: int = len(self.ltoi)  # the number of letters

        # init the model with zeros
        self.model = torch.zeros((size, size), dtype=torch.float64)

    def _make_counts_model(self) -> torch.Tensor:
        """
        Counts the bigram and makes and returns a tensor containing counts of the bigrams.
        """
        size = len(self.ltoi)
        counts_model: torch.Tensor = torch.zeros((size, size), dtype=torch.int)

        for word in self.input_words:
            # add dot delimiter to the word
            word = f'.{word}.'

            # iterate over the bigrams and increment count in the counts model
            for l1, l2 in zip(word, word[1:]):
                l1_index: int = self.ltoi.get(l1)
                l2_index: int = self.ltoi.get(l2)
                counts_model[l1_index, l2_index] += 1

        return counts_model

    def _populate_model_tensor(self, counts_model: torch.Tensor) -> None:
        """
        Convert the counts_model to probabilities row-wise.
        """
        # size: int = len(self.ltoi)
        # for row in range(size):
        #     for col in range(size):
        #         self.model[row, col] = counts_model[row, col] / counts_model[row].sum()

        # a shorter way to write the above for-loops
        self.model = counts_model.float().div(counts_model.sum(dim=1, keepdims=True))

    def train(self) -> None:
        """
        This method trains the model and creates the tensor with the probabilities on pair-wise characters.
        """
        # init an intermediate integer tensor for keeping counts
        counts_model: torch.Tensor = self._make_counts_model()

        # now convert the counts to probabilities row-wise
        # we do it row-wise because each row corresponds to an input letter and the columns
        # correspond to next letter. so, we are looking for the probability of the next
        # letter (column) given an input letter (row).
        self._populate_model_tensor(counts_model)

    def loss(self) -> float:
        """
        This method finds the loss of the model.
        """
        # for language models, we generally use negative log likelihood.

        loss: float = 0
        num: int = 0

        # iterate over all the words of the dataset (training data in this case, but, in real-world we will use testing dataset)
        # find the bigrams and take the probabilities of those bigrams.
        for word in self.input_words:
            word = f'.{word}.'

            for l1, l2 in zip(word, word[1:]):
                l1_index: int = self.ltoi.get(l1)
                l2_index: int = self.ltoi.get(l2)

                loss += torch.log(self.model[l1_index, l2_index])
                num += 1

        loss = (-1) * loss  # negative of the log
        loss = loss / num  # average out the loss

        return loss

    def predict(self) -> str:
        """
        This method predicts a word based on the trained model.
        """
        # start from the first row as it will be dot, continue sampling, and stop on getting a dot
        word: str = ''
        current_letter: str = '.'

        while True:
            # go to the row for the current letter
            current_letter_index: int = self.ltoi.get(current_letter)

            # sample possible next letter
            next_letter_index: int = torch.multinomial(
                self.model[current_letter_index],
                num_samples=1,  # select one letter
                replacement=True,
            ).item()
            next_letter: str = self.itol.get(next_letter_index)

            if next_letter == '.':
                break

            word = f'{word}{next_letter}'
            current_letter = next_letter

        return word
