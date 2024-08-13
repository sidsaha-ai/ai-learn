import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import Tensor


class DataHelpers:

    @staticmethod
    def _categorical_cols() -> list:
        return [
            'MSSubClass',
            'MSZoning',
        ]

    @staticmethod
    def _numerical_cols() -> list:
        return [
            'LotFrontage',
        ]

    @classmethod
    def make_data(cls, csv_filepath: str) -> Tensor:
        df: pd.DataFrame = pd.read_csv(csv_filepath)
        if df.empty:
            return None

        categorical_cols: list = cls._categorical_cols()
        numerical_cols: list = cls._numerical_cols()

        return df
