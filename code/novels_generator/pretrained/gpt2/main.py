"""
In this file, let's try finetuning GPT2 for novels generation.
"""

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from novels_generator.code.constants import SpecialTokens

from novels_generator.pretrained.gpt2.tokenizer import BooksTokenizer
from novels_generator.pretrained.gpt2.model import BooksGPTModel


def main():
    tokenizer = BooksTokenizer()
    model = BooksGPTModel(tokenizer)
    

if __name__ == '__main__':
    main()
