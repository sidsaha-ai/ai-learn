"""
This script is to run the experiment.
"""

from novels_generator.code import train
from novels_generator.code.constants import Hyperparamters


def main():
    """
    The main function to start the experiement execution.
    """
    num_epochs: int = 30

    # hyperparameters
    Hyperparamters.CONTEXT_LENGTH = 128
    Hyperparamters.BATCH_SIZE = 32
    Hyperparamters.VOCAB_SIZE = 40000
    Hyperparamters.EMBEDDING_SIZE = 256
    Hyperparamters.SELF_ATTENTION_HEADS = 8
    Hyperparamters.NUM_LAYERS = 4
    Hyperparamters.FEED_FORWARD_SIZE = 1024

    # define the learning rate scheduler
    def lr_lambda(epoch):
        # for 25 epochs, train usually (1e-4) and then muliply learning rate by 1e-2 to make it 1e-6
        return 1 if epoch < 25 else 1e-2

    train.train_model(
        num_epochs,
        lr_scheduler_type='LambdaLR',
        lr_lambda=lr_lambda,
    )


if __name__ == '__main__':
    main()
