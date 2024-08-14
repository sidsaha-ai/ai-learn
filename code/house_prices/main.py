"""
The main function to execute the training of the model and running to produce the final output.
"""
import argparse
import uuid

import pandas as pd
import torch
import utils
from house_prices.data_helpers import DataHelpers
from house_prices.neural_net import HousePricesNN
from torch import Tensor, nn, optim


class MainEngine:
    """
    The main class that acts as an executable.
    """

    def __init__(self, train_data_file: str, test_data_file: str, num_epochs: int) -> None:
        self.data_helpers: DataHelpers = DataHelpers()
        self.model: HousePricesNN = None

        self.train_data_file: str = train_data_file
        self.test_data_file: str = test_data_file
        self.num_epochs = num_epochs

    def train(self) -> None:
        """
        Function that trains the neural network.
        """
        inputs: Tensor
        targets: Tensor
        inputs, targets = self.data_helpers.make_training_data(self.train_data_file)
        print(f'{inputs.size()}')
        print(f'{targets.size()}')

        self.model = HousePricesNN(
            input_dimension=inputs.shape[1],
            output_dimension=targets.shape[1],
        )
        loss_fn = nn.MSELoss(reduction='sum')
        learning_rate: float = 1e-4
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            outputs: Tensor = self.model(inputs)
            loss: Tensor = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                utils.print_loss(epoch, loss)

        utils.print_loss('FINAL', loss)

    def test(self) -> pd.DataFrame:
        """
        Function to run the trained neural network to produce predictions.
        """
        df: pd.DataFrame = pd.read_csv(self.test_data_file)
        if df.empty:
            print('Test data is empty, returning...')
            return None

        ids: list = df['Id'].values
        inputs: Tensor = self.data_helpers.make_test_data(df)
        print(f'{inputs.size()}')

        # set model to evaluation mode
        self.model.eval()

        predictions: Tensor
        with torch.no_grad():
            predictions = self.model(inputs)

        predicted_values = predictions.numpy()
        predicted_values = self.data_helpers.output_pipeline.named_steps['min_max_scaler'].inverse_transform(predicted_values)  # pylint: disable=line-too-long  # NOQA

        res_df: pd.DataFrame = pd.DataFrame(
            predicted_values, columns=['SalePrice'],
        )
        res_df = pd.DataFrame({'Id': ids}).join(res_df)
        return res_df

    def output_csv(self, df: pd.DataFrame) -> None:
        """
        Takes the result dataframe and produces the CSV output file.
        """
        filename: str = f'./output_{uuid.uuid4()}.csv'
        if df.empty:
            print('Dataframe is empty, returning')
            return

        utils.make_output_csv(df, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs', required=True, type=int,
    )
    parser.add_argument(
        '--train_data_file', required=True, type=str,
    )
    parser.add_argument(
        '--test_data_file', required=True, type=str,
    )

    args = parser.parse_args()

    engine = MainEngine(
        args.train_data_file, args.test_data_file, args.num_epochs,
    )

    engine.train()
    result_df = engine.test()

    if result_df:
        print('=== Some results ===')
        print(result_df.head(10))
        engine.output_csv(result_df)
