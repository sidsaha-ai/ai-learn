"""
This script runs the experiment.
"""

import math

from novels_generator.code import train
from novels_generator.code.constants import Hyperparamters


def main():
    """
    The main function to run this experiment.
    """

    # hyperparameters
    Hyperparamters.CONTEXT_LENGTH = 256
    Hyperparamters.BATCH_SIZE = 32
    Hyperparamters.VOCAB_SIZE = 40000
    Hyperparamters.EMBEDDING_SIZE = 512
    Hyperparamters.SELF_ATTENTION_HEADS = 8
    Hyperparamters.NUM_LAYERS = 8
    Hyperparamters.FEED_FORWARD_SIZE = 4096

    warmup_epochs: int = 5
    constant_lr_epochs: int = 5
    total_num_epochs: int = 30

    def lr_schedule(epoch):
        current_epoch: int = epoch + 1

        if current_epoch <= warmup_epochs:
            # learning rate should be 1e-5
            return 1e-1
        
        if warmup_epochs < current_epoch <= (warmup_epochs + constant_lr_epochs):
            # learning rate should be 5e-4
            return 5
        
        if epoch > constant_lr_epochs:
            # learning rate should be cosine annealing to 1e-6 from 5e-4
            num_cosine_epochs: int = total_num_epochs - warmup_epochs - constant_lr_epochs
            num_elapsed_epochs: int = current_epoch - warmup_epochs - constant_lr_epochs
            min_lr = 1e-6
            max_lr = 5e-4
            initial_lr = 1e-5

            decay = 0.5 * (1 + math.cos(math.pi * num_elapsed_epochs / num_cosine_epochs))
            return (min_lr + (max_lr - min_lr) * decay) / initial_lr
    
    train.train_model(
        total_num_epochs, lr_scheduler_type='LambdaLR', lr_lambda=lr_schedule,
    )

if __name__ == '__main__':
    main()
