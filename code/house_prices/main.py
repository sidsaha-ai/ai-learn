"""
The main function to execute the training of the model and running to produce the final output.
"""
import argparse

import utils
from house_prices.data_helpers import DataHelpers
from house_prices.neural_net import HousePricesNN
from torch import Tensor, nn, optim


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

    model = HousePricesNN(
        input_dimension=inputs.shape[1],
        output_dimension=targets.shape[1],
    )
    loss_fn = nn.MSELoss(reduction='sum')
    learning_rate: float = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
