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

        self.weights: torch.Tensor = None  # the one-layer neural network
    
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

        # in the `inputs` and `targets`, the indices actually represent letters and we are going to predict letters
        # as output. Letters can be considered to be categorical data, in that sense (and not numerical data).
        # so, we will one-hot encode the inputs and targets
        t_inputs = F.one_hot(t_inputs, num_classes=len(self.ltoi))
        t_targets = F.one_hot(t_targets, num_classes=len(self.ltoi))

        # convert the tensors to float for neural net processing
        t_inputs = t_inputs.float()
        t_targets = t_targets.float()
        
        return t_inputs, t_targets
    
    def train(self, num_epochs: int) -> None:
        """
        This trains the model based on the `input_words`.
        """
        # make a list of input characters (to integers) that represents the first letter of the bigram
        # make a list of target characters (to integers) that represents the second letter of the bigram
        inputs: torch.Tensor = None
        targets: torch.Tensor = None
        inputs, targets = self._make_inputs_and_targets()

        # init weights (parameters of the model) with random numbers
        size: tuple[int, int] = (inputs.shape[1], targets.shape[1])
        self.weights = torch.randn(size, requires_grad=True)

        learning_rate: float = 50

        for epoch in range(num_epochs):
            # make one neural net layer with `weights`
            logits: torch.Tensor = inputs @ self.weights
            probs: torch.Tensor = F.softmax(logits, dim=1)

            # find loss (negative log likelihood)
            loss: torch.Tensor = F.nll_loss(
                torch.log(probs), targets.argmax(dim=1),
            )
            print(f'{epoch=}, Loss: {loss.item()}')

            # now do gradient descent on the weights
            self.weights.grad = None
            loss.backward()

            # update weights
            with torch.no_grad():
                self.weights += (-learning_rate) * self.weights.grad
