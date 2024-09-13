"""
This describes the Transformer model that will be used for training.
"""

import torch
import torch.nn as nn
from novels_generator.code.constants import Hyperparamters


class BooksTransformerModel(nn.Module):
    """
    The transformer model for training on the novels.
    """

    def __init__(
            self,
            vocab_size: int = Hyperparamters.VOCAB_SIZE,
            embedding_dim: int = Hyperparamters.EMBEDDING_SIZE,
            num_heads: int = Hyperparamters.SELF_ATTENTION_HEADS,
            num_layers: int = Hyperparamters.NUM_LAYERS,
            ff_dim: int = Hyperparamters.FEED_FORWARD_SIZE,
            context_length: int = Hyperparamters.CONTEXT_LENGTH,
    ) -> None:
        super().__init__()

        # token embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # positional encoding embeddings
        self.pos_embeddings = nn.Embedding(context_length, embedding_dim)

        # transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=ff_dim, 
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
