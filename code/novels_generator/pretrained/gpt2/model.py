import torch
from novels_generator.pretrained.gpt2.tokenizer import BooksTokenizer
from torch import Tensor
from transformers import AutoModelForCausalLM


class BooksGPTModel:

    def __init__(self, tokenizer: BooksTokenizer) -> None:
        model_name = 'openai-community/gpt2'

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))

        # move to GPU
        self.device = torch.device('mps') if torch.has_mps else torch.device('cpu')
        
        self.model = self.model.to(self.device)
    
    def forward(self, input_ids) -> Tensor:
        input_ids = input_ids.to(self.device)
        
        outputs = self.model(input_ids, labels=input_ids)

        return outputs.loss

    def parameters(self):
        return self.model.parameters
    