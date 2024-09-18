"""
This defines the hyperparameters of this experiment.
"""

from novels_generator.code.constants import Hyperparamters


def set_hyperparameters() -> None:
    """
    Sets the hyperparamters for this experiment.
    """
    Hyperparamters.CONTEXT_LENGTH = 128
    Hyperparamters.BATCH_SIZE = 32
    Hyperparamters.VOCAB_SIZE = 40000
    Hyperparamters.EMBEDDING_SIZE = 256
    Hyperparamters.SELF_ATTENTION_HEADS = 4
    Hyperparamters.NUM_LAYERS = 2
    Hyperparamters.FEED_FORWARD_SIZE = 1024
