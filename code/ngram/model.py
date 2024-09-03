"""
This contains the model implementation for the N-gram character model.
"""

import string

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch import Tensor
from torch.nn import functional as F


class NGramModel:  # pylint: disable=too-many-instance-attributes
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

        # self.input_words = self.input_words[0:10]

        # make inputs and targets
        self.inputs: Tensor = None
        self.targets: Tensor = None
        self._make_inputs_and_targets()

        # init the embeddings for the input
        self.embeddings: Tensor = None
        # neural network layers
        self.weights_1: Tensor = None
        self.bias_1: Tensor = None
        self.weights_2: Tensor = None
        self.bias_2: Tensor = None
        # parameters of the neural network
        self.parameters = []
        self._init_neural_net()

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
                [self.ltoi.get(letter) for letter in input_ngram],
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

        # while experimenting, we found that for embeddings of size-2, letters
        # were all clustered together, so there was no good learning. So, we are increasing the
        # embeddings size,
        embedding_size: int = 20  # each letter is represented by 20 integers

        # init a random embedding.
        self.embeddings = torch.randn(
            (num_letters, embedding_size), dtype=torch.float, requires_grad=True,
        )

    def _init_neural_net(self) -> None:
        """
        This method inits the layers of the neural network.
        """
        self._init_embeddings()

        # layer 1
        size: tuple[int, int] = (
            self.inputs.shape[1] * self.embeddings.shape[1], 500,
        )
        self.weights_1 = torch.randn(size, dtype=torch.float, requires_grad=True)
        self.bias_1 = torch.randn(self.weights_1.shape[1], dtype=torch.float, requires_grad=True)

        # layer 2
        size = (
            self.weights_1.shape[1], len(self.ltoi),
        )
        self.weights_2 = torch.randn(size, dtype=torch.float, requires_grad=True)
        self.bias_2 = torch.randn(self.weights_2.shape[1], dtype=torch.float, requires_grad=True)

        self.parameters = [
            self.embeddings, self.weights_1, self.bias_1, self.weights_2, self.bias_2,
        ]

    def _mini_batch(self) -> tuple[Tensor, Tensor]:
        """
        This method finds a random mini-batch of the inputs and targets and returns
        them for training.
        """
        input_size: int = self.inputs.shape[0]

        # take 5% of the input size as the minibatch
        percent: int = 5
        minibatch_size = int((percent * input_size) / 100)

        # random permutation of indices
        indices = torch.randperm(self.inputs.size(0))
        batch_indices = indices[0:minibatch_size]

        inputs_minibatch: Tensor = self.inputs[batch_indices]
        targets_minibatch: Tensor = self.targets[batch_indices]

        return inputs_minibatch, targets_minibatch

    def train(self, num_epochs: int) -> None:
        """
        This method will train the model.
        """
        print('Training...')
        
        losses: list = []  # to record the loss during each training

        for epoch in range(num_epochs):
            # ** Know-how **
            # The training loop takes a quite some time, because we process all the inputs
            # at once. Instead of training every loop on the entire input dataset, we can sample
            # from the input dataset to create a "mini-batch" and train an epoch on that mini-batch.
            # The next epoch will train on another random mini-batch.
            inputs_minibatch: Tensor = None
            targets_minibatch: Tensor = None
            inputs_minibatch, targets_minibatch = self._mini_batch()

            # get the embeddings of the inputs
            embs: Tensor = self.embeddings[inputs_minibatch]
            view_size: tuple[int, int] = (
                embs.shape[0], (embs.shape[1] * embs.shape[2]),
            )
            embs = embs.view(view_size)

            # layer 1
            l1_output = F.tanh((embs @ self.weights_1) + self.bias_1)

            # layer 2
            logits: Tensor = (l1_output @ self.weights_2) + self.bias_2

            # let's find loss
            loss = F.cross_entropy(logits, targets_minibatch)
            losses.append(loss.item())

            if epoch % 500 == 0:
                print(f'#{epoch}: Loss: {loss.item():.4f}')

            # back propagation
            for p in self.parameters:
                p.grad = None
            loss.backward()
            lr: float = 0.1
            for p in self.parameters:
                p.data -= lr * p.grad

        print(f'Loss: {loss.item():.4f}')
        self.plot_training_loss(losses)
        self.plot_embeddings()

    def predict(self) -> str:
        """
        This methods predicts a word from the trained model.
        """
        res: str = ''

        # we initially start with dots (according to batch size) and sample from the network.
        input_letters: list = ['.'] * self.batch_size
        input_letters = [self.ltoi.get(letter) for letter in input_letters]

        while True:
            # make embeddings for this input
            embs: Tensor = self.embeddings[input_letters]
            view_size: tuple[int, int] = (
                1, (embs.shape[0] * embs.shape[1]),
            )
            embs = embs.view(view_size)

            # layer 1
            l1_output: Tensor = F.tanh(embs @ self.weights_1) + self.bias_1

            # layer 2
            logits: Tensor = (l1_output @ self.weights_2) + self.bias_2

            # take softmax to convert logits to probabilities
            probs: Tensor = F.softmax(logits, dim=1)

            # sample from the probabilities
            sample = torch.multinomial(probs, num_samples=1, replacement=True).item()

            res += self.itol.get(sample)
            if self.itol.get(sample) == '.':
                break

            # shift input
            input_letters = input_letters[0:len(input_letters) - 1] + [sample]

        return res

    def plot_training_loss(self, losses: list) -> None:
        """
        This method is used for experimentation. This takes the list of losses
        got during training and plots them in a graph to visualize how the losses
        have varied.
        """
        epochs: list = range(1, len(losses) + 1)
        
        # let's plot the log of losses for more discernability.
        losses = torch.log(torch.tensor(losses)).tolist()
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def plot_embeddings(self) -> None:
        """
        This plots the embeddings. This will only work if the embeddings are of 2 dimensions.
        """
        embeddings = self.embeddings.detach().numpy()

        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        reduced_embeddings = tsne.fit_transform(embeddings)

        plt.figure(figsize=(12, 8))
        for letter, index in self.ltoi.items():
            x, y = reduced_embeddings[index, :]

            plt.scatter(x, y)
            plt.annotate(
                letter,
                (x, y),
                textcoords='offset points',
                xytext=(0, 10),
                ha='center',
            )
        
        plt.show()
