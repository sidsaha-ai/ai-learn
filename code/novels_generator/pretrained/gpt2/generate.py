"""
We use this script to generate text from the finetuned model.
"""

import os

from novels_generator.pretrained.gpt2.model import BooksGPTModel
from novels_generator.pretrained.gpt2.tokenizer import BooksTokenizer

def load_model() -> BooksGPTModel:
    """
    This function loads and returns the finetuned model.
    """
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'model.pth',
    )
    print(f'Loading model from path {model_path}...')

    tokenizer = BooksTokenizer()
    model = BooksGPTModel(tokenizer)
    model.load(model_path)
    
    print(f'Finished loading model from {model_path}.')

    return model

def main():
    model = load_model()
    print(model)


if __name__ == '__main__':
    main()
