"""
The main function to execute the training of the model and running to produce the final output.
"""
import argparse

import utils
from house_prices.data_helpers import \
    DataHelpers  # pylint: disable=import-error
from house_prices.neural_net import \
    HousePricesNN  # pylint: disable=import-error
from house_prices.rmse import LogRMSELoss  # pylint: disable=import-error
from torch import Tensor, optim


def train(num_epochs: int, train_data_file: str) -> None:
    """
    Function that trains the neural network.
    """
    # make training data
    inputs: Tensor
    targets: Tensor
    inputs, targets = DataHelpers.make_data(train_data_file)
    print(f'{inputs.size()}')
    print(f'{targets.size()}')

    model = HousePricesNN()
    loss_fn = LogRMSELoss()
    learning_rate: float = 0.1
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs: Tensor = model(inputs)
        loss: Tensor = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            utils.print_loss(epoch, loss)

    utils.print_loss('FINAL', loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs', required=True, type=int,
    )
    parser.add_argument(
        '--train_data_file', required=True, type=str,
    )

    args = parser.parse_args()

    train(
        args.num_epochs,
        args.train_data_file,
    )
