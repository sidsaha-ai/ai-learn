"""
The file generates the book from the model of this experiment.
"""

import os

import torch
from novels_generator.code.constants import SpecialTokens
from novels_generator.code.model import BooksTransformerModel
from novels_generator.code.tokenizer import BPETokenizer, BPETokenizerUtils
from novels_generator.experiments.training.expt_1 import hyperparameters
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

    model = load_model()
    tokenizer: BPETokenizer = BPETokenizerUtils.init()

    start_token = tokenizer.encode(SpecialTokens.START).ids[0]
    end_token = tokenizer.encode(SpecialTokens.END).ids[0]

    sequence = [start_token]
    max_text_length = 200000  # the maximum length to be generated

    for _ in tqdm(range(max_text_length), leave=False):
        inputs = torch.tensor(sequence[-hyperparameters.Hyperparamters.CONTEXT_LENGTH:]).unsqueeze(0)

        logits = None
        with torch.no_grad():
            logits = model(inputs)

        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, replacement=True).item()
        sequence.append(next_token)

        if next_token == end_token:
            break

    text = tokenizer.tokenizer.decode(sequence, skip_special_tokens=False)
    print(text)
    write_to_file(text)


if __name__ == '__main__':
    main()
