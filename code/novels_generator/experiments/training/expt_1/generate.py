"""
The file generates the book from the model of this experiment.
"""

import os

import torch
from novels_generator.code.model import BooksTransformerModel
from novels_generator.code.tokenizer import BPETokenizer, BPETokenizerUtils


def load_model() -> BooksTransformerModel:
    """
    Load the model from the path.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'model.pth',
    )

    print(f'Loading model from {path}...')
    model = BooksTransformerModel()
    model.load_state_dict(torch.load(path))

    # set the model to evaluation mode
    model.eval()
    print(f'Model loaded from path {path}.')

    return model


def main():
    """
    The main function where the execution starts.
    """
    model = load_model()
    tokenizer: BPETokenizer = BPETokenizerUtils.init()

    print(model)
    print(tokenizer)


if __name__ == '__main__':
    main()
