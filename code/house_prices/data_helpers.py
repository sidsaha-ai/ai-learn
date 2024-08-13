import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import Tensor
import numpy as np

import datetime as dt


class DataHelpers:

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
            'OverallQual',  # TODO: test whether to consider this categorical or numerical data
            'OverallCond',  # TODO: test whether to consider this categorical or numerical data
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
            'MsnVnrArea',
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

            # derived fields
            'Age',
            'RemodelAge',
            'GarageAge',
            'MoSinceSold',
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
    def make_data(cls, csv_filepath: str) -> Tensor:
        df: pd.DataFrame = pd.read_csv(csv_filepath)
        if df.empty:
            return None
        
        # replace `YearBuilt` with the age of the house
        df = cls._age(df=df, source_col='YearBuilt', target_col='Age')
        # replace `YearRemodAdd` with the remodel age of the house
        df = cls._age(df=df, source_col='YearRemodAdd', target_col='RemodelAge')
        # replace `GarageYrBlt` with the age of the garage
        df = cls._age(df=df, source_col='GarageYrBlt', target_col='GarageAge')
        # replace `MoSold` and `YrSold` with the number of months since sold
        df = cls._months_since_sold(df)

        categorical_cols: list = cls._categorical_cols()
        numerical_cols: list = cls._numerical_cols()

        return df
