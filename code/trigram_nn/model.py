"""
This file implements the trigram model using neural network.
"""

import string
import torch
from torch import Tensor
from torch.nn import functional as F


class TrigramNN:

    def __init__(self, input_words: list) -> None:
        self.input_words: list = input_words
        
        # mapping between a character and a corresponding integer
        self.ltoi: dict = {}
        self.itol: dict = {}
        self._make_char_int_mappings()

        # the inputs and targets tensor for training the neural network
        self.inputs: Tensor = None
        self.targets: Tensor = None

        # the weights of the neural network
        self.weights_1: Tensor = None
        self.weights_2: Tensor = None
    
    def _make_char_int_mappings(self) -> None:
        letters: list = ['.'] + list(string.ascii_lowercase)

        for ix, l in enumerate(letters):
            self.ltoi[l] = ix
            self.itol[ix] = l
    
    def _make_training_data(self) -> None:
        """
        Make trigrams with 2 input letters and 1 target letter.
        """
        l1_inputs: list = []  # first letter input
        l2_inputs: list = []  # second letter input
        targets: list = []  # corresponding targets

        for word in self.input_words:
            word = f'..{word}.'
            
            for l1, l2, l3 in zip(word, word[1:], word[2:]):
                l1_inputs.append(self.ltoi.get(l1))
                l2_inputs.append(self.ltoi.get(l2))
                targets.append(self.ltoi.get(l3))
        
        # make targets as one-hot encoded tensor
        self.targets = F.one_hot(
            torch.tensor(targets), num_classes=len(self.ltoi),
        )

        # make l1_input and l2_input as one hot encoded tensors
        t_l1_inputs: Tensor = F.one_hot(
            torch.tensor(l1_inputs), num_classes=len(self.ltoi),
        )
        t_l2_inputs: Tensor = F.one_hot(
            torch.tensor(l2_inputs), num_classes=len(self.ltoi),
        )
        # concat both the tensor to make the input tensor
        self.inputs = torch.cat(
            (t_l1_inputs, t_l2_inputs), dim=1,
        )

        # make the tensors float so that they can be used in training.
        self.inputs = self.inputs.float()
        self.targets = self.targets.float()
    
    def _init_nn(self) -> None:
        """
        Make the neural network weights.
        """
        l1_size: tuple[int, int] = (
            self.inputs.shape[1], self.inputs.shape[1],
        )
        l2_size: tuple[int, int] = (
            l1_size[1], self.targets.shape[1],
        )

        self.weights_1 = torch.randn(l1_size, requires_grad=True)
        self.weights_2 = torch.randn(l2_size, requires_grad=True)
    
    def _pred(self, inputs: Tensor) -> Tensor:
        """
        Find predictions based on the inputs passed and the weights trained.
        """
        # first layer
        layer_output = F.relu(inputs @ self.weights_1)

        # second layer
        logits = layer_output @ self.weights_2

        probs: Tensor = F.softmax(logits, dim=1)
        return probs

    def train(self, num_epochs: int) -> None:
        """
        This method trains the neural network.
        """
        self._make_training_data()
        self._init_nn()

        for epoch in range(num_epochs):
            # forward pass
            probs: Tensor = self._pred(self.inputs)
            # find the loss
            loss = F.nll_loss(torch.log(probs), self.targets.argmax(dim=1))
            print(f'Epoch: {epoch}, Loss: {loss:.4f}')

            # backward pass
            learning_rate: float = 5
            self.weights_1.grad = None
            self.weights_2.grad = None
            loss.backward()

            self.weights_1.data += (-learning_rate) * self.weights_1.grad
            self.weights_2.data += (-learning_rate) * self.weights_2.grad
    
    def _generate_input_from_two_letters(self, l1: str, l2: str) -> Tensor:
        l1_input: Tensor = torch.tensor(
            [self.ltoi.get(l1)], dtype=torch.int64,
        )
        l2_input: Tensor = torch.tensor(
            [self.ltoi.get(l2)], dtype=torch.int64,
        )

        # one-hot encode them, concat them, and turn to float
        t_l1_input: Tensor = F.one_hot(l1_input, num_classes=len(self.ltoi))
        t_l2_input: Tensor = F.one_hot(l2_input, num_classes=len(self.ltoi))
        inputs = torch.cat(
            (t_l1_input, t_l2_input), dim=1,
        )
        inputs = inputs.float()

        return inputs

    def predict(self) -> str:
        """
        Generate a prediction from the model.
        """
        res: str = ''
        
        l1: str = '.'
        l2: str = '.'

        while True:
            # make tensors for current input letters
            inputs = self._generate_input_from_two_letters(
                l1, l2,
            )

            probs: Tensor = self._pred(inputs)

            l3_ix: int = torch.multinomial(probs, num_samples=1, replacement=True).item()
            l3: str = self.itol.get(l3_ix)

            res = f'{res}{l3}'

            if l3 == '.':
                break

            l1, l2 = l2, l3
        
        return res
    
    def loss(self) -> float:
        """
        Find the loss across the entire input data.
        """
        loss: float = 0
        num: int = 0

        for word in self.input_words:
            word = f'..{word}.'

            for l1, l2, l3 in zip(word, word[1:], word[2:]):
                inputs: Tensor = self._generate_input_from_two_letters(l1, l2)
                probs: Tensor = self._pred(inputs)

                loss += torch.log(
                    probs[0, self.ltoi.get(l3)],
                )
                num += 1
        
        loss = (-1) * loss  # negative loss
        loss = loss / num  # average negative loss

        return loss
