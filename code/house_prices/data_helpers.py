"""
This defines the class for the data helpers that can be used to read the training and the testing
data.
"""
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch import Tensor


class DataHelpers:  # pylint: disable=too-few-public-methods
    """
    Class the defines class methods to read the data files and return tensors.
    """

    def __init__(self) -> None:
        # for numerical columns, impute to fill missing with 0 and apply min-max scaling
        self.numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('min_max_scaler', MinMaxScaler()),
        ])

        self.category_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ])

        self.output_pipeline = Pipeline(steps=[
            ('min_max_scaler', MinMaxScaler()),
        ])

    def _categorical_cols(self) -> list:
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

    def _numerical_cols(self) -> list:
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

    def make_training_data(self, csv_filepath: str) -> tuple[Tensor, Tensor]:
        """
        The main function of this class to read the data files and return tensors that can
        be used for training of the neural network.
        """
        df: pd.DataFrame = pd.read_csv(csv_filepath)
        if df.empty:
            return None

        # drop the ID column, it's not be used in training
        df = df.drop(columns=['Id'])
        df = df.dropna(subset=['SalePrice'])

        input_df = df.drop(columns=['SalePrice'])
        output_df: pd.DataFrame = df[['SalePrice']]

        # categorize columns into numerical and categorical
        categorical_cols: list = self._categorical_cols()
        numerical_cols: list = self._numerical_cols()

        numerical_input_df = pd.DataFrame(
            self.numeric_pipeline.fit_transform(input_df[numerical_cols]),
            columns=numerical_cols,
        )

        categorical_input_df = pd.DataFrame(
            self.category_pipeline.fit_transform(input_df[categorical_cols]),
            columns=self.category_pipeline.named_steps['onehot'].get_feature_names_out(
                categorical_cols
            ),
        )

        input_df = pd.concat(
            [numerical_input_df, categorical_input_df],
            axis=1,
        )

        output_df = pd.DataFrame(
            self.output_pipeline.fit_transform(output_df[output_df.columns]),
            columns=output_df.columns,
        )

        input_tensor: Tensor = torch.tensor(input_df.values, dtype=torch.float32)
        output_tensor: Tensor = torch.tensor(output_df.values, dtype=torch.float32)

        return (
            input_tensor, output_tensor,
        )

    def make_test_data(self, df: pd.DataFrame) -> Tensor:
        """
        Function to make the tensor data for evaluation.
        """
        if df.empty:
            return None

        df = df.drop(columns=['Id'])

        num_cols: list = self._numerical_cols()
        cat_cols: list = self._categorical_cols()

        numerical_df = pd.DataFrame(
            self.numeric_pipeline.transform(df[num_cols]), columns=num_cols,
        )
        categorical_df = pd.DataFrame(
            self.category_pipeline.transform(df[cat_cols]),
            columns=self.category_pipeline.named_steps['onehot'].get_feature_names_out(cat_cols),
        )
        df = pd.concat(
            [numerical_df, categorical_df], axis=1,
        )

        return torch.tensor(df.values, dtype=torch.float32)
