"""
This script is to run the experiment.
"""
import os

import torch
from novels_generator.code import train
from novels_generator.experiments.training.expt_3 import hyperparameters


def main():
    """
    The main function to runs the experiment.
    """
    num_epochs: int = 20

    # hyperparameters
    hyperparameters.set_hyperparameters()

    def lr_lambda(epoch):
        # for 20 epochs, train usally and then at 1e-5
        return 1 if epoch < 15 else 1e-1

    model = train.train_model(
        num_epochs, lr_scheduler_type='LambdaLR', lr_lambda=lr_lambda,
    )

    # save the model
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'model.pth',
    )
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    main()
