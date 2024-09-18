"""
Utils for generating the text from the model.
"""

import torch
from novels_generator.code.constants import SpecialTokens
from novels_generator.code.model import BooksTransformerModel
from novels_generator.code.tokenizer import BPETokenizer, BPETokenizerUtils
from tqdm import tqdm


class GenUtils:

    @classmethod
    def load_model(cls, model_path: str) -> BooksTransformerModel:
        """
        Loads the model from the model's path.
        """
        print(f'Loading model from path {model_path}')
        
        model = BooksTransformerModel()
        model.load_state_dict(torch.load(model_path))

        print(f'Model loaded from path {model_path}')
        return model
    
    @classmethod
    def generate_text(cls, model: BooksTransformerModel, max_text_length: int, context_length: int) -> str:
        """
        Generates the text.
        """
        tokenizer: BPETokenizer = BPETokenizerUtils.init()

        start_token = tokenizer.encode(SpecialTokens.START).ids[0]
        end_token = tokenizer.encode(SpecialTokens.END).ids[0]

        sequence = [start_token]

        for _ in tqdm(range(max_text_length), leave=False):
            inputs = torch.tensor(sequence[-context_length:]).unsqueeze(0)

            logits = None
            with torch.no_grad():
                logits = model(inputs)
            
            probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, replacement=True).item()
            sequence.append(next_token)

            if next_token == end_token:
                break
        
        text = tokenizer.decode(sequence)
        return text

    @classmethod
    def write_text_to_file(cls, text: str, path: str) -> None:
        """
        Writes the generated text to a txt file.
        """
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f'Wrote {len(text)} characters to file {path}')
