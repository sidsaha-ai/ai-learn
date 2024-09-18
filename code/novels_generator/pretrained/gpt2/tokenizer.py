from novels_generator.code.constants import SpecialTokens
from transformers import AutoTokenizer


class BooksTokenizer:

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
    
    def encode_into_sequences(self, book_content: str) -> list:
        context_length: int = 1024
        sequences = []

        start_ix = 0
        while start_ix < len(book_content):
            chunk = book_content[start_ix:start_ix + context_length]

            tokenized_chunk = self.tokenizer(
                chunk, return_tensors='pt', max_length=context_length, truncation=True, padding='max_length', add_special_tokens=True,
            )
            tokenized_chunk = tokenized_chunk.get('input_ids')
            
            sequences.append(tokenized_chunk)
            start_ix += context_length

        return sequences
    
    def __len__(self) -> int:
        return len(self.tokenizer)
