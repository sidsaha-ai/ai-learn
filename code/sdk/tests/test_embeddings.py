import pytest
import torch
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


def test_shape() -> None:
    """
    Test the shape property.
    """
    emb = Embedding(num_embeddings=1, embedding_dim=100)
    assert emb.shape == (1, 100)

    emb = Embedding(num_embeddings=20, embedding_dim=50)
    assert emb.shape == (20, 50)


def test_matmul() -> None:
    """
    Test emb @ tensor.
    """
    emb = Embedding(num_embeddings=27, embedding_dim=10)
    other = torch.randn(
        (10, 100), dtype=torch.float32,
    )
    
    res = emb @ other

    assert res is not None
    assert res.shape == (27, 100)

def test_rmatmul() -> None:
    """
    Test other @ emb.
    """
    other = torch.randn(
        (10, 100), dtype=torch.float32,
    )
    emb = Embedding(num_embeddings=100, embedding_dim=5)

    res = other @ emb

    assert res is not None
    assert res.shape == (10, 5)
