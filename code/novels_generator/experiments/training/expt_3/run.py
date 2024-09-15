"""
This script is to run the experiment.
"""
from novels_generator.code import train
from novels_generator.code.constants import Hyperparamters


def main():
    """
    The main function to runs the experiment.
    """
    num_epochs: int = 20

    # hyperparameters
    Hyperparamters.CONTEXT_LENGTH = 128
    Hyperparamters.BATCH_SIZE = 32
    Hyperparamters.VOCAB_SIZE = 40000
    Hyperparamters.EMBEDDING_SIZE = 256
    Hyperparamters.SELF_ATTENTION_HEADS = 8
    Hyperparamters.NUM_LAYERS = 4
    Hyperparamters.FEED_FORWARD_SIZE = 2048

    def lr_lambda(epoch):
        # for 20 epochs, train usally and then at 1e-5
        return 1 if epoch < 20 else 1e-1

    train.train_model(
        num_epochs, lr_scheduler_type='LambdaLR', lr_lambda=lr_lambda,
    )


if __name__ == '__main__':
    main()
