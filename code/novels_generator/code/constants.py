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
    END = '<END>'
    UNKNOWN = '<UNK>'
    PAD = '<PAD>'


class Hyperparamters:
    """
    Hyperparameters used by the model.
    """
    CONTEXT_LENGTH = 512      # the block size that must be provided as an input to the model.
    BATCH_SIZE = 32           # the mini-batch size
    VOCAB_SIZE = 40000        # the vocab size used for tokenizing
    EMBEDDING_SIZE = 512      # the size of the input and positional embeddings
    SELF_ATTENTION_HEADS = 8  # the number of self-attention heads
    NUM_LAYERS = 6            # the number of layers in the transformer model
    FEED_FORWARD_SIZE = 2048  # dimensions of the feed-forward layers
