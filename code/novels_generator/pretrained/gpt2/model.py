from transformers import AutoModelForCausalLM

from novels_generator.pretrained.gpt2.tokenizer import BooksTokenizer

class BooksGPTModel:

    def __init__(self, tokenizer: BooksTokenizer) -> None:
        model_name = 'openai-community/gpt2'

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))
