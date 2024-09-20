"""
Implements the tokenizer for finetuning the GPT2 model.
"""
import torch
from novels_generator.code.constants import SpecialTokens
from transformers import AutoTokenizer


class BooksTokenizer:
    """
    The books tokenizer to use for finetuning the model.
    """

    def __init__(self) -> None:
        model_name = 'openai-community/gpt2'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        special_tokens = {
            'additional_special_tokens': [
                SpecialTokens.CHAPTER_NAME_START,
                SpecialTokens.CHAPTER_NAME_END,
                SpecialTokens.HEADING_START,
                SpecialTokens.HEADING_END,
                SpecialTokens.PARAGRAPH_START,
                SpecialTokens.PARAGRAPH_END,
                SpecialTokens.START,
                SpecialTokens.END,
            ],
            'pad_token': SpecialTokens.PAD,
        }
        self.tokenizer.add_special_tokens(special_tokens)

        self.context_length = 1024

    def encode_into_sequences(self, book_content: str) -> list:
        """
        Takes an entire book and returns context length wise encoded chunks.
        """
        sequences = []

        start_ix = 0
        while start_ix < len(book_content):
            chunk = book_content[start_ix:start_ix + self.context_length]

            tokenized_chunk = self.tokenizer(
                chunk, return_tensors='pt', max_length=self.context_length, truncation=True, padding='max_length', add_special_tokens=True,
            )
            tokenized_chunk = tokenized_chunk.get('input_ids')

            sequences.append(tokenized_chunk)
            start_ix += self.context_length

        return sequences

    def __len__(self) -> int:
        return len(self.tokenizer)
    
    def encode(self, text: str) -> list:
        """
        Encodes the text using the tokenizer.
        """
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode the token and return.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
