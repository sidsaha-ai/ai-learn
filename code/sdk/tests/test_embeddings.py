import pytest
from torch import Tensor

from sdk.embeddings import Embedding

def test_indexing() -> None:
    """
    Test the indexing operation.
    """
    num_embeddings: int = 2
    embedding_dim: int = 5

    emb = Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
    )

    # select one row
    slice = emb[0]
    assert slice.shape[0] == embedding_dim

    # select one item
    slice = emb[0, 1]
    assert type(slice.item() == float)

    slice = emb[0, [1, 2, 4]]
    assert slice.shape[0] == 3

    # select rows 0 and 1, and colummns 1, 2, and 4
    slice = emb[[0, 1], :][:, [1, 2, 4]]
    assert slice.shape == (2, 3)
