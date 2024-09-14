"""
This file trains the neural network.
"""

import os

import torch
from matplotlib import pyplot as plt
from novels_generator.code.constants import Hyperparamters
from novels_generator.code.dataset import BooksDataset
from novels_generator.code.epub_reader import EPubReader
from novels_generator.code.model import BooksTransformerModel
from novels_generator.code.tokenizer import BPETokenizer
from torch import nn
from torch.optim import lr_scheduler
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


def plot_losses(train_losses: list, val_losses: list) -> None:
    """
    Function that plots losses.
    """
    x = [ix + 1 for ix, _ in enumerate(train_losses)]

    plt.plot(x, train_losses, label='Train Loss')
    plt.plot(x, val_losses, label='Val Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Val losses')
    plt.legend()
    plt.show()


def train_model(num_epochs: int, lr_scheduler_type = None, lr_lambda = None) -> None:  # pylint: disable=too-many-locals
    """
    The train function that trains the model.
    """
    # train tokenizer
    tokenizer = BPETokenizer()
    book_contents = read_train_books()
    tokenizer.train(book_contents)

    # make training dataset
    books_train_dataset: BooksDataset = BooksDataset(tokenizer, 'train')
    books_train_dataloader: DataLoader = DataLoader(
        books_train_dataset, batch_size=Hyperparamters.BATCH_SIZE, shuffle=True,
    )

    # validation dataset
    books_val_dataset: BooksDataset = BooksDataset(tokenizer, 'val')
    books_val_dataloader: DataLoader = DataLoader(
        books_val_dataset, batch_size=Hyperparamters.BATCH_SIZE, shuffle=True,
    )

    model = BooksTransformerModel()                             # create a model
    loss_fn = nn.CrossEntropyLoss()                             # the loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # the optimizer
    scheduler = None                                            # the learning rate scheduler

    if lr_scheduler_type:
        match lr_scheduler_type:
            case 'LambdaLR':
                scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    device = torch.device('mps') if torch.has_mps else torch.device('cpu')
    # device = torch.device('cpu')

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Num Parameters: {num_params:,}')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss_sum = 0

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
            train_loss_sum += loss.item()

            # backward pass
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss_sum / len(books_train_dataloader)
        val_loss = validate(books_val_dataloader, model, loss_fn)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        loss_diff = val_loss - avg_train_loss
        
        # get current learning rate
        current_lr = [group['lr'] for group in optimizer.param_groups][0]
        print(f'Epoch: {epoch}, LR: {current_lr}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Loss Diff: {loss_diff:.4f}')

        # step the scheduler if available
        if scheduler:
            scheduler.step()

    # plot the losses
    plot_losses(train_losses, val_losses)
