"""
This experiment sets the base line, with the following hyperparameters.

CONTEXT_LENGTH = 128      # the block size that must be provided as an input to the model.
BATCH_SIZE = 32           # the mini-batch size
VOCAB_SIZE = 40000        # the vocab size used for tokenizing
EMBEDDING_SIZE = 256      # the size of the input and positional embeddings
SELF_ATTENTION_HEADS = 4  # the number of self-attention heads
NUM_LAYERS = 2            # the number of layers in the transformer model
FEED_FORWARD_SIZE = 1024  # dimensions of the feed-forward layers

NUM_EPOCHS = 20
"""

from novels_generator.code import train

def main():
    """
    The main function to start experiment execution.
    """
    num_epochs: int = 20

    train.train_model(num_epochs)


if __name__ == '__main__':
    main()


"""
RESULTS
=======
"""