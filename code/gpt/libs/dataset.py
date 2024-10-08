"""
This fetches the tiny shakespeare dataset and creates a training datset and a validation dataset.
"""

import requests
import torch
from libs.encoder import Encoder
from torch import Tensor


class Dataset:
    """
    Class for dataset.
    """

    def __init__(self, block_size: int) -> None:
        super().__init__()

        # the URL from where to fetch the dataset
        self.url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        self.block_size = block_size

        # variables
        self.text = None
        self.encoder = None
        self.train_data = None
        self.val_data = None

        self.build()

    def fetch_data(self) -> None:
        """
        Method to fetch the dataset.
        """
        try:
            r = requests.get(self.url, timeout=10)
            if not r.ok:
                raise Exception(r.content)

            self.text = str(r.text)

        except Exception as e:
            print(f'Exception: {e}')
            raise

    def make_encoder(self) -> None:
        """
        Method to create the encoder.
        """
        vocab = sorted(list(set(self.text)))

        self.encoder = Encoder(vocab)

    def build_train_val_data(self) -> None:
        """
        Method to build the training and validation dataset.
        """
        train_index = int((90 * len(self.text)) / 100)

        train_data = self.text[0:train_index]
        val_data = self.text[train_index:]

        self.train_data = [self.encoder.encode(el) for el in list(train_data)]
        self.val_data = [self.encoder.encode(el) for el in list(val_data)]
        self.train_data = torch.tensor(self.train_data)
        self.val_data = torch.tensor(self.val_data)

    def build(self) -> None:
        """
        Method to build the things necessary for the dataset.
        """
        self.fetch_data()
        self.make_encoder()
        self.build_train_val_data()

    def batch(self, batch_size: int, split: str = 'train') -> tuple[Tensor, Tensor]:
        """
        Creates a batch of either training dataset or val dataset.
        """
        data = self.train_data if split == 'train' else self.val_data

        # random indexes to select for the batch
        rand_ix = torch.randint(
            low=0,
            high=len(data) - self.block_size,
            size=(batch_size,),  # select `batch_size` number of batches
        )

        inputs_batch = torch.stack(
            [data[ix:ix + self.block_size] for ix in rand_ix],
        )
        targets_batch = torch.stack(
            [data[ix + 1:ix + 1 + self.block_size] for ix in rand_ix],
        )

        return inputs_batch, targets_batch
