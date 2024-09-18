"""
This script is to run the experiment.
"""

import os

import torch
from novels_generator.code import train
from novels_generator.experiments.training.expt_4 import hyperparameters


def main():
    """
    The main function to run this experiment.
    """
    num_epochs: int = 30

    # hyperparameters
    hyperparameters.set_hyperparameters()

    def lr_lambda(epoch):
        # base LR in the model is 1e-4.
        # first 5 epochs, LR should be 1e-5. next 20 epochs, LR should be 1e-4. last 5 epochs, LR should be 1e-6.
        if epoch < 5:
            return 1e-1
        if epoch < 25:
            return 1
        return 1e-2

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
