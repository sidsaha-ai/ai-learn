"""
This defines the GPT 2 model that we are using.
"""
import torch
from novels_generator.pretrained.gpt2.tokenizer import BooksTokenizer
from torch import Tensor
from transformers import AutoModelForCausalLM


class BooksGPTModel:
    """
    The GPT-2 model that we will finetune.
    """

    def __init__(self, tokenizer: BooksTokenizer) -> None:
        model_name = 'openai-community/gpt2'

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids) -> Tensor:
        """
        Implements the forward pass.
        """
        outputs = self.model(input_ids, labels=input_ids)
        return outputs

    def parameters(self):
        """
        Returns the parameters of the model.
        """
        return self.model.parameters()

    def save(self, path: str) -> None:
        """
        Saves the model as a file.
        """
        torch.save(self.model.state_dict(), path)

    def train(self) -> None:
        """
        Put the model to training mode.
        """
        self.model.train()

    def eval(self) -> None:
        """
        Put the model to evaluation mode.
        """
        self.model.eval()
    
    def to(self, device) -> None:
        """
        Moves the model to the device.
        """
        self.model = self.model.to(device)
