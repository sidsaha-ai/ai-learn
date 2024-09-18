"""
The file generates the book from the model of this experiment.
"""

import torch
import os
from novels_generator.code.model import BooksTransformerModel

def load_model() -> BooksTransformerModel:
    """
    Load the model from the path.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'model.pth',
    )
    print(f'Model Path: {path}')

    return None

def main():
    model = load_model()


if __name__ == '__main__':
    main()
