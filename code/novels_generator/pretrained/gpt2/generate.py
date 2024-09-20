"""
We use this script to generate text from the finetuned model.
"""

import os

from novels_generator.pretrained.gpt2.model import BooksGPTModel
from novels_generator.pretrained.gpt2.tokenizer import BooksTokenizer
from novels_generator.code.constants import SpecialTokens

from tqdm import tqdm
import torch


def load_model(tokenizer: BooksTokenizer) -> BooksGPTModel:
    """
    This function loads and returns the finetuned model.
    """
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'model.pth',
    )
    print(f'Loading model from path {model_path}...')

    model = BooksGPTModel(tokenizer)
    model.load(model_path)

    print(f'Finished loading model from {model_path}.')

    return model


def main():
    """
    The main function to start execution.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    tokenizer = BooksTokenizer()

    model = load_model(tokenizer)
    model.eval()  # set to eval mode
    model.to(device)

    max_text_length: int = 500
    context_length: int = 1024
    end_token = tokenizer.encode(SpecialTokens.END)[0]

    story: str = 'Naina had set foot in New York for the first time in her life yesterday. And here she is today.'
    story = f'{SpecialTokens.START}{story}'

    sequence = tokenizer.encode(story)

    for _ in tqdm(range(max_text_length), leave=True):
        inputs = torch.tensor(sequence[-context_length:]).unsqueeze(0)
        inputs = inputs.to(device)

        outputs = None
        with torch.no_grad():
            outputs = model.forward(inputs)

        logits = outputs.logits 
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, replacement=True).item()
        sequence.append(next_token)

        if next_token == end_token:
            break
    
    print(f'{len(sequence)=}')
    generated_text = tokenizer.decode(sequence)
    print(generated_text)

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'generated_text.txt',
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(generated_text)
    print(f'Wrote to file {path}')


if __name__ == '__main__':
    main()
