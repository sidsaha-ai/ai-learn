"""
This contains the model implementation for the N-gram character model.
"""

import string

import torch
from torch import Tensor
from torch.nn import functional as F


class NGramModel:
    """
    This is the model implementation for an N-gram character model.
    """

    def __init__(self, input_words: list, batch_size: int) -> None:
        self.input_words: list = input_words
        self.batch_size = batch_size

        # map each letter to a number
        self.ltoi: dict = {}
        self.itol: dict = {}
        self._make_mappings()

        self.input_words = self.input_words[0:10]  # TODO: remove this

        # make inputs and targets
        self.inputs: Tensor = None
        self.targets: Tensor = None
        self._make_inputs_and_targets()

        # init the embeddings for the input
        self.embeddings: Tensor = None
        self._init_embeddings()

        # neural network layers
        self.weights_1: Tensor = None
        self.bias_1: Tensor = None
        self.weights_2: Tensor = None
        self.bias_2: Tensor = None
    
    def _make_mappings(self) -> None:
        """
        This method makes mappings between each letter and a corresponding number and vice-versa.
        """
        letters: list = ['.'] + list(string.ascii_lowercase)

        for ix, l in enumerate(letters):
            self.ltoi[l] = ix
            self.itol[ix] = l
        
    def _make_ngrams(self) -> list[tuple]:
        """
        Creates n-gram inputs and targets. For e.g., let's say a word is "emma", then for a batch size of 2,
        the n-grams would be like - 
        [
            (.., e),
            (.e, m),
            (em, m),
            (mm, a),
            (ma, .)
        ]
        """
        res: list[tuple] = []

        for word in self.input_words:
            # pad appropriately at the beginning with dots
            word = ('.' * self.batch_size) + word + '.'

            for i in range(len(word) - self.batch_size):
                inputs = word[i:i + self.batch_size]
                targets = word[i + self.batch_size]
                res.append(
                    (inputs, targets),
                )
        
        return res
    
    def _make_inputs_and_targets(self) -> None:
        """
        Creates an input and output tensor with integer mappings of letters.
        """
        ngrams: list[tuple] = self._make_ngrams()
        inputs: list = []
        targets: list = []

        for input_ngram, target_letter in ngrams:
            inputs.append(
                [self.ltoi.get(l) for l in input_ngram],
            )
            targets.append(
                self.ltoi.get(target_letter),
            )

        self.inputs = torch.tensor(inputs)
        self.targets = torch.tensor(targets)
    
    def _init_embeddings(self) -> None:
        """
        This methods inits the embeddings for the input letters that will be trained.
        Embeddings are way to represent the universe of inputs in a "small space" that are trained
        so that "similar inputs" end up nearby in that space. 

        In this character model, the universe of letters has 27 characters. Let's represent them
        with 2 integers. So, we will create a random tensor of shape 27*2. 27 is the universe of letters
        and 2 is the embedding for each letter.
        """
        num_letters: int = len(self.ltoi)
        embedding_size: int = 2  # each letter is represented by 2 integers

        # init a random embedding.
        self.embeddings = torch.randn(
            (num_letters, embedding_size), dtype=torch.float,
        )

    def train(self, num_epcohs: int) -> None:
        """
        This method will train the model.
        """
        # get the embeddings of the inputs
        embs: Tensor = self.embeddings[self.inputs]
        view_size: tuple[int, int] = (
            embs.shape[0], (embs.shape[1] * embs.shape[2]),
        )
        embs = embs.view(view_size)
        
        # layer 1
        self.weights_1 = torch.randn(
            (embs.shape[1], 100), dtype=torch.float,
        )
        self.bias_1 = torch.randn(
            self.weights_1.shape[1], dtype=torch.float,
        )
        l1_logits: Tensor = (embs @ self.weights_1) + self.bias_1

        # apply the
        l1_output: Tensor = F.tanh(l1_logits)

        # layer 2
        self.weights_2 = torch.randn(
            (self.weights_1.shape[1], len(self.ltoi)), dtype=torch.float,
        )
        self.bias_2 = torch.randn(
            self.weights_2.shape[1], dtype=torch.float,
        )
        logits: Tensor = (l1_output @ self.weights_2) + self.bias_2

        # let's find loss
        loss = F.cross_entropy(logits, self.targets)
        print(f'Loss: {loss.item():.4f}')

    def predict(self) -> None:
        """
        This methods predicts a word from the trained model.
        """
        return
