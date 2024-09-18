"""
This script is to run the experiment.
"""

import os

import torch
from novels_generator.code import train
from novels_generator.experiments.training.expt_2 import hyperparameters


def main():
    """
    The main function to start the experiement execution.
    """
    num_epochs: int = 30

    # hyperparameters
    hyperparameters.set_hyperparameters()

    # define the learning rate scheduler
    def lr_lambda(epoch):
        # for 25 epochs, train usually (1e-4) and then muliply learning rate by 1e-2 to make it 1e-6
        return 1 if epoch < 25 else 1e-2

    model = train.train_model(
        num_epochs,
        lr_scheduler_type='LambdaLR',
        lr_lambda=lr_lambda,
    )

    # save the model
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'model.pth',
    )
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    main()
