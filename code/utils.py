"""
This file contains various utility functions.
"""
from torch import Tensor
import pandas as pd


def print_loss(epoch: int, loss: Tensor) -> None:
    """
    Function to print loss for an epoch.
    """
    print(f'{epoch=}, loss={loss.item():.4f}')


def make_output_csv(df: pd.DataFrame, filepath: str) -> None:
    """
    Make CSV file from the dataframe.
    """
    df.to_csv(
        filepath, index=False, quotechar='"', escapechar='\\', doublequote=False,
    )
    print(f'CSV file written to {filepath}')
