"""
Unit tests for the Tanh layer.
"""

import torch
from sdk.tanh import Tanh
from torch import Tensor
import pytest

@pytest.mark.parametrize(
    'num_rows, num_cols',
    [
        (1, 10),
        (10, 1),
        (10, 10),
    ],
)
def test_call(num_rows: int, num_cols: int) -> None:
    """
    Test the call.
    """
    size = (num_rows, num_cols)
    x: Tensor = torch.randn(size, dtype=torch.float)
    
    tanh = Tanh()
    y = tanh(x)
    
    assert y is not None
    assert y.shape[0] == num_rows
    assert y.shape[1] == num_cols
