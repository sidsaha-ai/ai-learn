"""
The script to run this experiment.
"""
import os

import torch
from novels_generator.code import train
from novels_generator.code.constants import Hyperparamters
from novels_generator.experiments.training.expt_1 import hyperparameters


def main():
    """
    The main function to start experiment execution.
    """
    num_epochs: int = 20

    # override the hyperparameters
    hyperparameters.set_hyperparameters()

    model = train.train_model(num_epochs)

    # save the model
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'model.pth',
    )
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    main()
