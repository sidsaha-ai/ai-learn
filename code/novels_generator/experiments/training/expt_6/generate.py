"""
The file generates the book from the model of this experiment.
"""

import os

from novels_generator.experiments.training.expt_6 import hyperparameters
from novels_generator.experiments.training.generation_utils import GenUtils


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
        os.path.dirname(os.path.abspath(__file__)), 'generated_text.txt',
    )
    GenUtils.write_text_to_file(text, filepath)


if __name__ == '__main__':
    main()
