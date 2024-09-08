"""
Build the model by leveraging the SDK.
"""

from ngram.dataset import Dataset
from ngram.encoder import Encoder
from sdk.batch_norm import BatchNorm
from sdk.embeddings import Embedding
from sdk.linear import Linear
from sdk.tanh import Tanh
from torch.nn import functional as F
from sdk.plotter import Plotter


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
        self.embeddings = Embedding(
            num_embeddings=len(self.encoder.ltoi), embedding_dim=10,  # each letter is represented by 10 dimensions
        )
        self.l1 = Linear(
            in_features=self.dataset.train_inputs.shape[1] * self.embeddings.shape[1],
            out_features=200,
            nonlinearity='tanh',
        )
        self.bn1 = BatchNorm(num_features=self.l1.out_features)
        self.t1 = Tanh()

        self.l2 = Linear(
            in_features=self.l1.out_features, out_features=100, nonlinearity='tanh',
        )
        self.bn2 = BatchNorm(num_features=self.l2.out_features)
        self.t2 = Tanh()

        self.l3 = Linear(
            in_features=self.l2.out_features, out_features=len(self.encoder.ltoi), nonlinearity=None,
        )

        self.parameters = self.embeddings.parameters() + self.l1.parameters() + self.l2.parameters() + self.l3.parameters()
        print(f'Num parameters: {self.embeddings.num_parameters + self.l1.num_parameters + self.l2.num_parameters + self.l3.num_parameters}')
    
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
        losses: list[dict] = []
        for epoch in range(num_epochs):
            inputs_batch, targets_batch = self.dataset.minibatch()
            embs = self.embeddings[inputs_batch]
            embs = embs.view(
                (embs.shape[0], (embs.shape[1] * embs.shape[2])),
            )

            out = self.t1(self.bn1(self.l1(embs)))
            out = self.t2(self.bn2(self.l2(out)))
            logits = self.l3(out)

            loss = F.cross_entropy(logits, targets_batch)

            # backpropagation
            for p in self.parameters:
                p.grad = None
            loss.backward()

            lr = self._lr(epoch, num_epochs)
            for p in self.parameters:
                p.data += (-lr) * p.grad
            
            losses.append({
                'loss': loss.item(),
                'lr': lr,
            })
            
            if epoch % 100 == 0:
                print(f'#{epoch}, LR: {lr:.4f}, Loss: {loss.item():.4f}')
        print(f'#{epoch}, LR: {lr:.4f}, Loss: {loss.item():.4f}')

        Plotter.plot_losses(losses)
