"""
The script to run this experiment.
"""
from novels_generator.code import train
from novels_generator.code.constants import Hyperparamters


def main():
    """
    The main function to start experiment execution.
    """
    num_epochs: int = 20

    # override the hyperparameters
    Hyperparamters.CONTEXT_LENGTH = 128
    Hyperparamters.BATCH_SIZE = 32
    Hyperparamters.VOCAB_SIZE = 40000
    Hyperparamters.EMBEDDING_SIZE = 256
    Hyperparamters.SELF_ATTENTION_HEADS = 4
    Hyperparamters.NUM_LAYERS = 2
    Hyperparamters.FEED_FORWARD_SIZE = 1024

    train.train_model(num_epochs)


if __name__ == '__main__':
    main()
