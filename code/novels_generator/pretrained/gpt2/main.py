"""
In this file, let's try finetuning GPT2 for novels generation.
"""

import math
import os

import torch
from accelerate import Accelerator
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
        # hyperparameters
        self.batch_size = 4
        self.base_lr = 1e-4
        self.num_epochs = 10

        self.tokenizer = BooksTokenizer()
        self.model = BooksGPTModel(self.tokenizer)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_schedule)

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

        # accelerate
        self.accelerator = Accelerator()
        self.train_dataloader, self.val_dataloader, self.model, self.optimizer = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader, self.model, self.optimizer,
        )

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')  # pylint: disable=line-too-long  # NOQA
        print(f'{self.device=}')
        self.model.to(self.device)

    def lr_schedule(self, epoch: int) -> float:
        """
        Learning rate schedule function.
        """
        res = 1
        current_epoch = epoch + 1

        if current_epoch <= 2:
            res = 1e-1  # LR should become 1e-5

        if 3 <= current_epoch <= 5:
            res = 5  # LR should become 5e-4

        if current_epoch > 5:
            # use cosine annealing to reduce LR from 5e-4 to 1e-6
            num_cosine_epochs: int = 5
            num_elapsed_epochs: int = current_epoch - 5
            min_lr = 1e-6
            max_lr = 5e-4

            res = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * num_elapsed_epochs / num_cosine_epochs))
            res = res / self.base_lr

        return res

    def _save_model(self) -> None:
        # saves the model
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'model.pth',
        )
        self.model.save(path)

    @torch.no_grad()
    def _val_loss(self, epoch: int) -> float:
        # finds the validation loss after every epoch
        total_loss = 0
        self.model.eval()

        with tqdm(self.val_dataloader, unit='batch', leave=False) as dataloader:
            dataloader.set_description(f'Validation Epoch: {epoch}')

            for batch in dataloader:
                batch = batch.to(self.device)
                outputs = self.model.forward(batch)

                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_dataloader)
        self.model.train()

        return avg_loss

    def run(self) -> None:
        """
        Finetunes the model.
        """
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            with tqdm(self.train_dataloader, unit='batch', leave=False) as dataloader:
                dataloader.set_description(f'Training Epoch: {epoch}')

                for batch in dataloader:
                    batch = batch.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model.forward(batch)  # forward pass
                    loss = outputs.loss

                    self.accelerator.backward(loss)  # backward pass

                    self.optimizer.step()
                    total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_dataloader)
            avg_val_loss = self._val_loss(epoch)
            current_lr = [group['lr'] for group in self.optimizer.param_groups][0]
            print(f'Epoch: {epoch}, LR: {current_lr}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            self.scheduler.step()

        self._save_model()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
