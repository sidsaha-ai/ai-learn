import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from simple_nn.neural_net import SimpleNeuralNet

import argparse


def _inputs_and_targets() -> tuple[Tensor, Tensor]:
    inputs: Tensor = torch.tensor([
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ], dtype=torch.float32)

    targets: Tensor = torch.tensor([
        [1.0],
        [-1.0],
        [-1.0],
        [1.0],
    ], dtype=torch.float32)

    return (inputs, targets)


def train(num_epochs: int):
    """
    Function that trains the neural network
    """
    inputs, targets = _inputs_and_targets()

    model = SimpleNeuralNet()

    # define the loss function
    loss_fn = nn.MSELoss()

    # define the optimizer
    learning_rate: float = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # before starting back propagation, reset the gradients
        outputs = model(inputs)  # call the model to get the output
        loss = loss_fn(outputs, targets)  # compute the loss basis the output of the model and the targets

        loss.backward()  # back propagate
        optimizer.step()  # update the weights of the neural network

        # print progress
        if (epoch + 1) % 50 == 0:
            print(f'{epoch=}, loss={loss.item():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs', required=True, type=int,
    )

    args = parser.parse_args()

    train(
        args.num_epochs,
    )
