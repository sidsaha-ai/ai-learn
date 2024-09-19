"""
In this file, let's try finetuning GPT2 for novels generation.
"""

import os

import torch
from novels_generator.pretrained.gpt2.dataset import BooksDataset
from novels_generator.pretrained.gpt2.model import BooksGPTModel
from novels_generator.pretrained.gpt2.tokenizer import BooksTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Trainer:
    """
    Trainer class to finetune the model.
    """
    
    def __init__(self) -> None:
        self.batch_size = 4
        self.lr = 5e-5
        self.num_epochs = 3

        self.tokenizer = BooksTokenizer()
        self.model = BooksGPTModel(self.tokenizer)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # make training dataloader
        train_dataset = BooksDataset(self.tokenizer, 'train')
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
        )

        # make validation dataloader
        val_dataset = BooksDataset(self.tokenizer, 'val')
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=True,
        )
    
    def _save_model(self) -> None:
        # saves the model
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'model.pth',
        )
        self.model.save(path)
    
    def run(self) -> None:
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            with tqdm(self.train_dataloader, unit='batch', leave=False) as dataloader:
                dataloader.set_description(f'Training Epoch: {epoch}')

                for batch in dataloader:
                    self.optimizer.zero_grad()
                    loss = self.model.forward(batch)  # forward pass
                    loss.backward()                   # backward pass
                    self.optimizer.step()
                    total_loss += loss.item()
            
            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}')
        
        self._save_model()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
