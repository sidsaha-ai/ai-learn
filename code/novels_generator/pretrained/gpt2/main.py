"""
In this file, let's try finetuning GPT2 for novels generation.
"""

from novels_generator.code.constants import SpecialTokens
from novels_generator.pretrained.gpt2.model import BooksGPTModel
from novels_generator.pretrained.gpt2.tokenizer import BooksTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    tokenizer = BooksTokenizer()
    model = BooksGPTModel(tokenizer)
    

if __name__ == '__main__':
    main()
