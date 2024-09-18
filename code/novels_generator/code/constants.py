"""
Contains constants to be used.
"""


class SpecialTokens:
    """
    Special tokens for tokenizer.
    """
    CHAPTER_NAME_START = '<CHAPTER_NAME_START>'
    CHAPTER_NAME_END = '<CHAPTER_NAME_END>'
    HEADING_START = '<HEADING_START>'
    HEADING_END = '<HEADING_END>'
    PARAGRAPH_START = '<PARA_START>'
    PARAGRAPH_END = '<PARA_END>'
    START = '<START>'
    END = '<END>'
    UNKNOWN = '<UNK>'
    PAD = '<PAD>'


class Hyperparamters:
    """
    Hyperparameters used by the model.
    """
    CONTEXT_LENGTH = 128      # the block size that must be provided as an input to the model.
    BATCH_SIZE = 32           # the mini-batch size
    VOCAB_SIZE = 40000        # the vocab size used for tokenizing
    EMBEDDING_SIZE = 256      # the size of the input and positional embeddings
    SELF_ATTENTION_HEADS = 4  # the number of self-attention heads
    NUM_LAYERS = 2            # the number of layers in the transformer model
    FEED_FORWARD_SIZE = 1024  # dimensions of the feed-forward layers
    DROPOUT = 0.1             # the dropout for the transformer layer
    LAYER_DROP_PROB = 0       # the probability to drop layers during training
