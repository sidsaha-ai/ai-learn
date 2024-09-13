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


def make_dataset(folder: str) -> BooksDataset:
    """
    Function to create a dataset of all the training books.
    """
    # read all the books
    book_contents = read_train_books()

    # train a tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(book_contents)

    # create the dataset
    books_dataset = BooksDataset(tokenizer, folder)

    return books_dataset


@torch.no_grad()
def validate(dataloader, model, loss_fn) -> float:
    """
    Find the validation loss.
    """
    device = torch.device('mps') if torch.has_mps else torch.device('cpu')

    model.to(device)
    model.eval()  # set the model to eval so that no gradient tracking happens

    loss = 0

    for batch in dataloader:
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)

        logits = logits.view(-1, Hyperparamters.VOCAB_SIZE) if logits.is_contiguous() else logits.reshape(-1, Hyperparamters.VOCAB_SIZE)
        targets = targets.view(-1) if targets.is_contiguous() else targets.reshape(-1)

        loss += loss_fn(logits, targets).item()

    loss = loss / len(dataloader)
    model.train()  # set it back to training mode

    return loss


def main(num_epochs: int) -> None:  # pylint: disable=too-many-locals
    """
    The main function that trains the model.
    """
    # make training dataset
    books_train_dataset: BooksDataset = make_dataset('train')
    books_train_dataloader: DataLoader = DataLoader(
        books_train_dataset, batch_size=Hyperparamters.BATCH_SIZE, shuffle=True,
    )

    # validation dataset
    books_val_dataset: BooksDataset = make_dataset('val')
    books_val_dataloader: DataLoader = DataLoader(
        books_val_dataset, batch_size=Hyperparamters.BATCH_SIZE, shuffle=True,
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
        for batch in books_train_dataloader:
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

        val_loss = validate(books_val_dataloader, model, loss_fn)
        print(f'Epoch: {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs', required=True, type=int,
    )
    args = parser.parse_args()

    main(
        args.num_epochs,
    )
