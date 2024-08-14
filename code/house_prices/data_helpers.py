"""
This defines the class for the data helpers that can be used to read the training and the testing
data.
"""
import datetime as dt

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch import Tensor


class DataHelpers:  # pylint: disable=too-few-public-methods
    """
    Class the defines class methods to read the data files and return tensors.
    """

    @staticmethod
    def _categorical_cols() -> list:
        return [
            'MSSubClass',
            'MSZoning',
            'Street',
            'Alley',
            'LotShape',
            'LandContour',
            'Utilities',
            'LotConfig',
            'LandSlope',
            'Neighborhood',
            'Condition1',
            'Condition2',
            'BldgType',
            'HouseStyle',
            'OverallQual',  # test whether to consider this categorical or numerical data
            'OverallCond',  # test whether to consider this categorical or numerical data
            'RoofStyle',
            'RoofMatl',
            'Exterior1st',
            'Exterior2nd',
            'MasVnrType',
            'ExterQual',
            'ExterCond',
            'Foundation',
            'BsmtQual',
            'BsmtCond',
            'BsmtExposure',
            'BsmtFinType1',
            'BsmtFinType2',
            'Heating',
            'HeatingQC',
            'CentralAir',
            'Electrical',
            'KitchenQual',
            'Functional',
            'FireplaceQu',
            'GarageType',
            'GarageFinish',
            'GarageQual',
            'GarageCond',
            'PavedDrive',
            'PoolQC',
            'Fence',
            'MiscFeature',
            'SaleType',
            'SaleCondition',
        ]

    @staticmethod
    def _numerical_cols() -> list:
        return [
            'LotFrontage',
            'LotArea',
            'MasVnrArea',
            'BsmtFinSF1',
            'BsmtFinSF2',
            'BsmtUnfSF',
            'TotalBsmtSF',
            '1stFlrSF',
            '2ndFlrSF',
            'LowQualFinSF',
            'GrLivArea',
            'BsmtFullBath',
            'BsmtHalfBath',
            'FullBath',
            'HalfBath',
            'BedroomAbvGr',
            'KitchenAbvGr',
            'TotRmsAbvGrd',
            'Fireplaces',
            'GarageCars',
            'GarageArea',
            'WoodDeckSF',
            'OpenPorchSF',
            'EnclosedPorch',
            '3SsnPorch',
            'ScreenPorch',
            'PoolArea',
            'MiscVal',
            'YearBuilt',
            'YearRemodAdd',
            'GarageYrBlt',
            'MoSold',
            'YrSold',
        ]

    @staticmethod
    def _age(*, df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
        current_year: int = dt.datetime.now().year

        # add age as target_col
        df[target_col] = df[source_col].apply(
            lambda x: current_year - x if pd.notnull(x) else np.nan
        )
        df = df.drop(columns=[source_col])  # remove source column
        return df

    @staticmethod
    def _months_since_sold(df: pd.DataFrame) -> pd.DataFrame:
        mo_sold_col: str = 'MoSold'
        yr_sold_col: str = 'YrSold'

        def _calculate(row):
            sold_month = row[mo_sold_col]
            sold_year = row[yr_sold_col]

            if pd.isnull(sold_month) or pd.isnull(sold_year):
                return np.nan

            sale_date = dt.datetime(
                year=int(sold_year),
                month=int(sold_month),
                day=1,
            )
            num_days: int = (dt.datetime.now() - sale_date).days
            num_months: int = num_days // 30
            return num_months

        df['MoSinceSold'] = df.apply(_calculate, axis=1)
        df = df.drop(columns=[mo_sold_col, yr_sold_col])

        return df

    @classmethod
    def make_data(cls, csv_filepath: str) -> tuple[Tensor, Tensor]:
        """
        The main function of this class to read the data files and return tensors that can
        be used by the neural network.
        """
        df: pd.DataFrame = pd.read_csv(csv_filepath)
        if df.empty:
            return None

        # drop the rows where `SalePrice` is null
        df = df.dropna(subset=['SalePrice'])

        # drop the ID column, it's not be used in training
        df = df.drop(columns=['Id'])

        # divide into input data and output data
        input_df: pd.DataFrame = df.drop(columns=['SalePrice'])
        output_df: pd.DataFrame = df[['SalePrice']]

        # categorize columns into numerical and categorical
        categorical_cols: list = cls._categorical_cols()
        numerical_cols: list = cls._numerical_cols()

        # for numerical columns, impute to fill missing with 0
        # and apply min-max scaling
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('min_max_scaler', MinMaxScaler()),
        ])

        numerical_input_df = pd.DataFrame(
            numeric_pipeline.fit_transform(input_df[numerical_cols]),
        )

        # one-hot encode categorical columns after imputing
        category_pipeline = Pipeline(steps=[
            # impute None columns with `missing` value
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            # then one-hot encode them
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ])
        categorical_input_df = pd.DataFrame(
            category_pipeline.fit_transform(input_df[categorical_cols]),
        )

        input_df = pd.concat(
            [numerical_input_df, categorical_input_df],
            axis=1,
        )

        # apply pipeline to output also
        output_df = pd.DataFrame(
            numeric_pipeline.fit_transform(output_df[output_df.columns]),
        )

        return (
            torch.tensor(input_df.values, dtype=torch.float32),
            torch.tensor(output_df.values, dtype=torch.float32),
        )
