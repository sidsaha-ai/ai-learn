"""
This file trains the neural network.
"""

import argparse
import os

import torch
from novels_generator.code.constants import Hyperparamters
from novels_generator.code.dataset import BooksDataset
from novels_generator.code.epub_reader import EPubReader
from novels_generator.code.model import BooksTransformerModel
from novels_generator.code.tokenizer import BPETokenizer
from torch import nn
from torch.utils.data import DataLoader


def read_train_books() -> list:
    """
    This reads all the training books.
    """
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(path, 'data')
    path = os.path.join(path, 'train')

    book_contents = []
    reader = EPubReader()

    for f in os.listdir(path):
        if not f.endswith('.epub'):
            continue

        filepath = os.path.join(path, f)
        content = reader.read(filepath)
        if not content:
            continue

        book_contents.append(content)

    return book_contents


def make_dataset() -> BooksDataset:
    """
    Function to create a dataset of all the training books.
    """
    # read all the books
    book_contents = read_train_books()

    # train a tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(book_contents)

    # create the dataset
    books_dataset = BooksDataset(tokenizer)

    return books_dataset


def main(num_epochs: int) -> None:
    """
    The main function that trains the model.
    """
    books_dataset: BooksDataset = make_dataset()
    books_dataloader: DataLoader = DataLoader(
        books_dataset, batch_size=Hyperparamters.BATCH_SIZE, shuffle=True,
    )

    model = BooksTransformerModel()                             # create a model
    loss_fn = nn.CrossEntropyLoss()                             # the loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # the optimizer

    device = torch.device('mps') if torch.has_mps else torch.device('cpu')
    # device = torch.device('cpu')

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Num Parameters: {num_params:,}')

    for epoch in range(num_epochs):
        # run all the batches in one epoch
        for batch in books_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # forward pass
            inputs = batch[:, :-1]  # all but the last token
            targets = batch[:, 1:]  # all but the first token

            logits = model(inputs)

            # reshape for the loss function
            logits = logits.view(-1, Hyperparamters.VOCAB_SIZE) if logits.is_contiguous() else logits.reshape(-1, Hyperparamters.VOCAB_SIZE)
            targets = targets.view(-1) if targets.is_contiguous() else targets.reshape(-1)

            loss = loss_fn(logits, targets)

            # backward pass
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs', required=True, type=int,
    )
    args = parser.parse_args()

    main(
        args.num_epochs,
    )
