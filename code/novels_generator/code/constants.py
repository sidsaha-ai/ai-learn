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
    CONTEXT_LENGTH = 512   # the block size that must be provided as an input to the model.
