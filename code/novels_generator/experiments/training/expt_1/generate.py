"""
The file generates the book from the model of this experiment.
"""

import os

import torch
from novels_generator.code.model import BooksTransformerModel


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
    model = load_model()
    print(model)


if __name__ == '__main__':
    main()
