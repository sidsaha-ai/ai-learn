import string

import torch
from torch.nn import functional as F
from torch import Tensor


class KBigramNN:

    def __init__(self, input_words: list) -> None:
        super().__init__()
        self.input_words = input_words

        self.ltoi: dict = {}
        self.itol: dict = {}
        self.inputs: Tensor = None
        self.targets: Tensor = None
        self.weights: Tensor = None
        self.num = 0

        self._make_ltoi()
    
    def _make_ltoi(self) -> None:
        letters: list = ['.']
        letters += list(string.ascii_lowercase)

        for index, letter in enumerate(letters):
            self.ltoi[letter] = index
            self.itol[index] = letter
        
    def _make_inputs_and_targets(self) -> None:
        inputs, targets = [], []

        for word in self.input_words:
            chs = ['.'] + list(word) + ['.']
            for l1, l2 in zip(chs, chs[1:]):
                inputs.append(self.ltoi.get(l1))
                targets.append(self.ltoi.get(l2))

        self.inputs = torch.tensor(inputs)
        self.targets = torch.tensor(targets)

        self.num = self.inputs.nelement()

        self.inputs = F.one_hot(self.inputs, num_classes=27).float()
    
    def train(self, num_epochs: int) -> None:
        self._make_inputs_and_targets()

        self.weights = torch.randn((27, 27), requires_grad=True)

        for epoch in range(num_epochs):
            logits = self.inputs @ self.weights
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)
            loss = F.nll_loss(
                torch.log(probs), self.targets,
            )
            print(f'{epoch=}, Loss: {loss.item()}')

            self.weights.grad = None
            loss.backward()

            self.weights.data += -50 * self.weights.grad
    
    def predict(self) -> str:
        ix = 0
        out = []

        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ self.weights
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)

            ix = torch.multinomial(probs, num_samples=1, replacement=True).item()
            out.append(
                self.itol.get(ix),
            )
            if ix == 0:
                break
                
        return ''.join(out)
