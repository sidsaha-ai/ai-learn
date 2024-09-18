"""
The file generates the book from the model of this experiment.
"""

import os

import torch
from novels_generator.code.constants import SpecialTokens
from novels_generator.code.model import BooksTransformerModel
from novels_generator.code.tokenizer import BPETokenizer, BPETokenizerUtils
from novels_generator.experiments.training.expt_1 import hyperparameters
from novels_generator.experiments.training.generation_utils import GenUtils
from tqdm import tqdm


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


def write_to_file(text: str) -> None:
    """
    Write the generated text to file.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'generated_text.txt',
    )

    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Wrote {len(text)} characters to file {path}')


def main():
    """
    The main function where the execution starts.
    """
    hyperparameters.set_hyperparameters()

    # load model
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'model.pth',
    )
    model = GenUtils.load_model(model_path)
    model.eval()  # set model to evaluation model

    max_text_length: int = 200000
    text = GenUtils.generate_text(model, max_text_length, hyperparameters.Hyperparamters.CONTEXT_LENGTH)

    print(text)

    # write to file
    filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'model.pth',
    )
    GenUtils.write_text_to_file(text, filepath)


if __name__ == '__main__':
    main()
