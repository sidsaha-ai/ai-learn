"""
This describes the Transformer model that will be used for training.
"""

import torch
from novels_generator.code.constants import Hyperparamters
from torch import nn


class BooksTransformerModel(nn.Module):
    """
    The transformer model for training on the novels.
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            vocab_size: int = None,
            embedding_dim: int = None,
            num_heads: int = None,
            num_layers: int = None,
            ff_dim: int = None,
            context_length: int = None,
    ) -> None:
        super().__init__()

        vocab_size = vocab_size or Hyperparamters.VOCAB_SIZE
        embedding_dim = embedding_dim or Hyperparamters.EMBEDDING_SIZE
        num_heads = num_heads or Hyperparamters.SELF_ATTENTION_HEADS
        num_layers = num_layers or Hyperparamters.NUM_LAYERS
        ff_dim = ff_dim or Hyperparamters.FEED_FORWARD_SIZE
        context_length = context_length or Hyperparamters.CONTEXT_LENGTH

        # token embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # positional encoding embeddings
        self.pos_embeddings = nn.Embedding(context_length, embedding_dim)

        # transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=Hyperparamters.DROPOUT,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final linear layer
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the network.
        """
        context_length = x.size(1)
        pos = torch.arange(0, context_length, device=x.device).unsqueeze(0)

        # add both input embeddings and positional embeddings to feed to the network
        x = self.embeddings(x) + self.pos_embeddings(pos)

        # pass through the transformer layers of the network
        x = self.transformer_encoder(x)

        # pass through the last linear layer to produce logits
        logits = self.output_layer(x)

        return logits
