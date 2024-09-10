"""
This fetches the tiny shakespeare dataset and creates a training datset and a validation dataset.
"""

import requests
import torch
from torch import Tensor
from libs.encoder import Encoder


class Dataset:

    def __init__(self, block_size: int) -> None:
        super().__init__()

        # the URL from where to fetch the dataset
        self.url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        self.block_size = block_size

        self.build()
    
    def fetch_data(self) -> None:
        try:
            r = requests.get(self.url)
            if not r.ok:
                raise Exception(r.content)
            
            self.text = str(r.text)

        except Exception as e:
            print(f'Exception: {e}')
            raise
    
    def make_encoder(self) -> None:
        vocab = sorted(list(set(self.text)))

        self.encoder = Encoder(vocab)
    
    def build_train_val_data(self) -> None:
        train_index = int((90 * len(self.text)) / 100)

        train_data = self.text[0:train_index]
        val_data = self.text[train_index:]

        self.train_data = [self.encoder.encode(el) for el in list(train_data)]
        self.val_data = [self.encoder.encode(el) for el in list(val_data)]
    
    def build(self) -> None:
        self.fetch_data()
        self.make_encoder()
        self.build_train_val_data()
    