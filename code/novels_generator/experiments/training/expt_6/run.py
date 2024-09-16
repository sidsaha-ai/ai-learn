"""
This script runs the experiment.
"""

from novels_generator.code import train
from novels_generator.code.constants import Hyperparamters

def main():
    """
    The main function to run this experiment.
    """

    num_epochs: int = 30

    # hyperparameters
    Hyperparamters.CONTEXT_LENGTH = 256
    Hyperparamters.BATCH_SIZE = 32
    Hyperparamters.VOCAB_SIZE = 40000
    Hyperparamters.EMBEDDING_SIZE = 512
    Hyperparamters.SELF_ATTENTION_HEADS = 8
    Hyperparamters.NUM_LAYERS = 8
    Hyperparamters.FEED_FORWARD_SIZE = 4096

    def lr_lambda(epoch):
        if epoch < 5:
            return 1e-1
        if epoch < 25:
            return 1
        return 1e-2
    
    train.train_model(
        num_epochs, lr_scheduler_type='LambdaLR', lr_lambda=lr_lambda,
    )

if __name__ == '__main__':
    main()
