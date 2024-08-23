import string

import torch
from torch.nn import functional as F
from torch import Tensor


class BigramNN:
    
    def __init__(self, input_words: list) -> None:
        super().__init__()
        self.input_words = input_words

        self.ltoi: dict = {}  # map of letter to integer
        self.itol: dict = {}  # map of integer to letter

        self._make_ltoi()

        self.inputs: Tensor = None  # input data for training
        self.targets: Tensor = None  # output data for training

        self.weights: Tensor = None  # the one-layer neural network
    
    def _make_ltoi(self) -> None:
        letters: list = ['.']

        letters += list(string.ascii_lowercase)

        for index, letter in enumerate(letters):
            self.ltoi[letter] = index
            self.itol[index] = letter
    
    def _make_inputs_and_targets(self) -> tuple[Tensor, Tensor]:
        inputs: list = []
        targets: list = []

        for word in self.input_words:
            word = f'.{word}.'  # add delimiter
            
            for l1, l2 in zip(word, word[1:]):
                inputs.append(self.ltoi.get(l1))
                targets.append(self.ltoi.get(l2))
        
        self.inputs: Tensor = torch.tensor(inputs)
        self.targets: Tensor = torch.tensor(targets)

        # in the `inputs` and `targets`, the indices actually represent letters and we are going to predict letters
        # as output. Letters can be considered to be categorical data, in that sense (and not numerical data).
        # so, we will one-hot encode the inputs and targets
        self.inputs = F.one_hot(self.inputs, num_classes=len(self.ltoi))
        self.targets = F.one_hot(self.targets, num_classes=len(self.ltoi))

        # convert the tensors to float for neural net processing
        self.inputs = self.inputs.float()
        self.targets = self.targets.float()
    
    def _pred(self, inputs: Tensor) -> Tensor:
        # Calculates the probabilities based on the current weights
        logits: Tensor = inputs @ self.weights
        probs: Tensor = F.softmax(logits, dim=1)

        return probs
    
    def train(self, num_epochs: int) -> None:
        """
        This trains the model based on the `input_words`.
        """
        # make a list of input characters (to integers) that represents the first letter of the bigram
        # make a list of target characters (to integers) that represents the second letter of the bigram
        self._make_inputs_and_targets()

        # init weights (parameters of the model) with random numbers
        size: tuple[int, int] = (self.inputs.shape[1], self.targets.shape[1])
        self.weights = torch.randn(size, requires_grad=True)

        learning_rate: float = 50

        for epoch in range(num_epochs):
            # make one neural net layer with `weights`
            probs: Tensor = self._pred(self.inputs)

            # find loss
            loss: Tensor = F.nll_loss(
                torch.log(probs), self.targets.argmax(dim=1),
            )
            print(f'{epoch=}, Loss: {loss.item():.4f}')

            # now do gradient descent on the weights
            self.weights.grad = None
            loss.backward()

            # update weights
            with torch.no_grad():
                self.weights += (-learning_rate) * self.weights.grad
    
    def predict(self) -> str:
        """
        Predict a word based on the trained model.
        """
        res: str = ''
        current_letter: str = '.'

        while True:
            # make an input tensor with current letter to predict next letter.
            inputs: Tensor = torch.tensor([self.ltoi.get(current_letter)], dtype=torch.int64)
            inputs = F.one_hot(inputs, num_classes=len(self.ltoi))
            inputs = inputs.float()

            probs: Tensor = self._pred(inputs)

            # pick a sample based on the probability as the next letter
            next_letter_index: int = torch.multinomial(
                probs, num_samples=1, replacement=True,
            ).item()
            next_letter: str = self.itol.get(next_letter_index)
            if next_letter == '.':
                break
            
            res = f'{res}{next_letter}'

        return res
