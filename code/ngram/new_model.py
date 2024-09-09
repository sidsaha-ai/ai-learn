"""
Build the model by leveraging the SDK.
"""

import torch
from ngram.dataset import Dataset
from ngram.encoder import Encoder
from sdk.batch_norm import BatchNorm
from sdk.cross_entropy import CrossEntropy
from sdk.embeddings import Embedding
from sdk.flatten import Flatten
from sdk.linear import Linear
from sdk.plotter import Plotter
from sdk.sequential import Sequential
from sdk.tanh import Tanh
from torch import Tensor
from torch.nn import functional as F


class NewNgramModel:
    """
    Model class that leverages the SDK.
    """
    def __init__(self, input_words: list[str], context_length: int) -> None:
        self.input_words: list[str] = input_words
        self.context_length: int = context_length

        self.encoder = Encoder()
        self.dataset = Dataset(
            input_words=input_words, context_length=context_length,
        )

        # layers
        embedding_dim: int = 10
        num_hidden: int = 100

        self.model = Sequential([
            Embedding(num_embeddings=len(self.encoder.ltoi), embedding_dim=embedding_dim),
            Flatten(),

            Linear(in_features=self.dataset.train_inputs.shape[1] * embedding_dim, out_features=num_hidden, nonlinearity='tanh'),
            BatchNorm(num_features=num_hidden),
            Tanh(),

            Linear(in_features=num_hidden, out_features=num_hidden, nonlinearity='tanh'),
            BatchNorm(num_features=num_hidden),
            Tanh(),

            Linear(in_features=num_hidden, out_features=num_hidden, nonlinearity='tanh'),
            BatchNorm(num_features=num_hidden),
            Tanh(),

            Linear(in_features=num_hidden, out_features=num_hidden, nonlinearity='tanh'),
            BatchNorm(num_features=num_hidden),
            Tanh(),

            Linear(in_features=num_hidden, out_features=num_hidden, nonlinearity='tanh'),
            BatchNorm(num_features=num_hidden),
            Tanh(),

            Linear(in_features=num_hidden, out_features=len(self.encoder.ltoi), nonlinearity=None),
        ])

        self.loss_fn = CrossEntropy()
        self.parameters = self.model.parameters()

    def _lr(self, epoch: int, num_epochs: int) -> float:
        """
        Returns the learning rate.
        """
        decay_percent: int = 90
        return 0.1 if epoch <= (decay_percent * num_epochs) / 100 else 0.001

    def train(self, num_epochs: int) -> None:
        """
        The method trains the neural network.
        """
        self.model.training = True

        losses: list[dict] = []
        for epoch in range(num_epochs):
            inputs_batch, targets_batch = self.dataset.minibatch(batch_percent=1)

            x = self.model(inputs_batch)
            loss = self.loss_fn(x, targets_batch)

            # backpropagation
            for p in self.parameters:
                p.grad = None
            loss.backward()

            lr = self._lr(epoch, num_epochs)
            for p in self.parameters:
                p.data += (-lr) * p.grad

            losses.append(
                {'loss': loss.item(), 'lr': lr},
            )

            # let's see some plots after the first epoch. this is primarily done
            # to identify issues with the initialization of the network.
            if epoch == 1:
                Plotter.plot_activations(self.model)
                Plotter.plot_gradients(self.model)

            if epoch % 100 == 0:
                print(f'#{epoch}, LR: {lr:.4f}, Loss: {loss.item():.4f}')
        print(f'#{epoch}, LR: {lr:.4f}, Loss: {loss.item():.4f}')

        Plotter.plot_losses(losses)

        # after training, check the activations (saturation should be low)
        Plotter.plot_activations(self.model, to_plot=False)

    @torch.no_grad()
    def loss(self, inputs: Tensor, targets: Tensor) -> float:
        """
        Returns the loss over the passed inputs and targets
        """
        self.model.training = False

        x = self.model(inputs)
        loss = self.loss_fn(x, targets).item()

        return loss

    def train_loss(self) -> float:
        """
        Returns the loss over the training dataset.
        """
        return self.loss(self.dataset.train_inputs, self.dataset.train_targets)

    def dev_loss(self) -> float:
        """
        Returns the loss over the dev dataset.
        """
        return self.loss(self.dataset.dev_inputs, self.dataset.dev_targets)

    def test_loss(self) -> float:
        """
        Returns the loss over the test dataset.
        """
        return self.loss(self.dataset.test_inputs, self.dataset.test_targets)

    def generate(self) -> str:
        """
        Generates a name from the trained neural network.
        """
        res: str = ''

        self.model.training = False

        inputs = [self.encoder.encode(letter) for letter in list('.' * self.context_length)]
        while True:
            x = self.model(
                torch.tensor([inputs]),
            )
            probs = F.softmax(x, dim=1)

            output = torch.multinomial(probs, num_samples=1, replacement=True).item()
            output_letter = self.encoder.decode(output)
            res += output_letter
            if output_letter == '.':
                break

            inputs = inputs[1:] + [output]

        return res
