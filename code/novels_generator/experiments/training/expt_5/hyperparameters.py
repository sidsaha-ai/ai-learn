"""
This defines the hyperparameters for this experiment.
"""

from novels_generator.code.constants import Hyperparamters

def set_hyperparameters() -> None:
    """
    Sets the hyperparameters for this experiment.
    """
    Hyperparamters.CONTEXT_LENGTH = 256
    Hyperparamters.BATCH_SIZE = 32
    Hyperparamters.VOCAB_SIZE = 40000
    Hyperparamters.EMBEDDING_SIZE = 512
    Hyperparamters.SELF_ATTENTION_HEADS = 8
    Hyperparamters.NUM_LAYERS = 4
    Hyperparamters.FEED_FORWARD_SIZE = 2048