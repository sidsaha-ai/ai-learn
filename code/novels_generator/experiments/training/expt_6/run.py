"""
This script runs the experiment.
"""

import math
import os

import torch
from novels_generator.code import train
from novels_generator.experiments.training.expt_6 import hyperparameters


def lr_schedule(epoch: int) -> float:
    """
    Learning Rate scheduling multiplier.
    """
    res = 1
    current_epoch = epoch + 1

    warmup_epochs: int = 5
    constant_lr_epochs: int = 5
    total_num_epochs: int = 35

    if current_epoch <= warmup_epochs:
        # learning rate should be 1e-5
        res = 1e-1

    if warmup_epochs < current_epoch <= (warmup_epochs + constant_lr_epochs):
        # learning rate should be 5e-4
        res = 5

    if epoch > (warmup_epochs + constant_lr_epochs):
        # learning rate should be cosine annealing to 1e-6 from 5e-4
        num_cosine_epochs: int = total_num_epochs - warmup_epochs - constant_lr_epochs
        num_elapsed_epochs: int = current_epoch - warmup_epochs - constant_lr_epochs
        min_lr = 1e-6
        max_lr = 5e-4
        base_lr = 1e-4

        res = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * num_elapsed_epochs / num_cosine_epochs))
        res = res / base_lr

    return res


def main():
    """
    The main function to run this experiment.
    """
    total_num_epochs: int = 35

    # hyperparameters
    hyperparameters.set_hyperparameters()

    model = train.train_model(
        total_num_epochs, lr_scheduler_type='LambdaLR', lr_lambda=lr_schedule,
    )

    # save the model
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'model.pth',
    )
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    main()
