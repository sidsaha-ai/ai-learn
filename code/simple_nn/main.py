"""
The main executable file that will train the neural network.
"""
import argparse

import torch
from simple_nn.neural_net import SimpleNeuralNet  # pylint:disable=import-error
from torch import Tensor, nn, optim


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


def _print_loss(epoch: int, loss: Tensor) -> None:
    print(f'{epoch=}, loss={loss.item():.4f}')


def _gpu(x: Tensor) -> Tensor:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    x = x.to(device)
    return x


def train(num_epochs: int):
    """
    Function that trains the neural network
    """
    inputs, targets = _inputs_and_targets()

    model = SimpleNeuralNet()

    # move the model, input, and output tensors to GPU
    inputs = _gpu(inputs)
    targets = _gpu(targets)
    model = _gpu(model)

    # define the loss function
    loss_fn = nn.MSELoss()

    # define the optimizer
    learning_rate: float = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # (IMP): before starting back propagation, reset the gradients
        optimizer.zero_grad()

        # call the model to get the output
        outputs = model(inputs)

        # compute the loss basis the outputs of the model and the targets
        loss: Tensor = loss_fn(outputs, targets)

        loss.backward()  # back propagate
        optimizer.step()  # update the weights of the neural network

        # print progress
        if (epoch + 1) % 50 == 0:
            _print_loss(epoch, loss)

    _print_loss('FINAL', loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs', required=True, type=int,
    )

    args = parser.parse_args()

    train(
        args.num_epochs,
    )
