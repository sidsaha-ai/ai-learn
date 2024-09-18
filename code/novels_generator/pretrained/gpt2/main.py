"""
In this file, let's try finetuning GPT2 for novels generation.
"""

import torch
from novels_generator.pretrained.gpt2.dataset import BooksDataset
from novels_generator.pretrained.gpt2.model import BooksGPTModel
from novels_generator.pretrained.gpt2.tokenizer import BooksTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def _train_dataloader(tokenizer) -> DataLoader:
    dataset = BooksDataset(tokenizer, 'train')
    return DataLoader(
        dataset, batch_size=8, shuffle=True,
    )


def _val_dataloader(tokenizer) -> DataLoader:
    dataset = BooksDataset(tokenizer, 'val')
    return DataLoader(
        dataset, batch_size=8, shuffle=True,
    )


def main():
    tokenizer = BooksTokenizer()                                # the tokenizer
    model = BooksGPTModel(tokenizer)                            # the model
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=5e-5)  # the optimizer

    num_epochs = 3
    train_dataloader = _train_dataloader(tokenizer)

    for epoch in range(num_epochs):
        model.model.train()
        total_loss = 0
        
        with tqdm(train_dataloader, unit='batch', leave=False) as dataloader:
            dataloader.set_description(f'Epoch {epoch}')
            for batch in dataloader:
                optimizer.zero_grad()
                loss = model.forward(batch)  # forward pass
                loss.backward()  # backprop
                optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch: {epoch}, Train Loss: {avg_loss:.4f}')


if __name__ == '__main__':
    main()
