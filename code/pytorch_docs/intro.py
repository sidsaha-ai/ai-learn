"""
Trying out stuff from Pytorch's turtorial chapter on datasets and dataloaders.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from matplotlib import pyplot as plt
from torch import nn, Tensor


def dataloader(train: bool, batch_size: int) -> DataLoader:
    """
    Makes the training or testing dataloader and returns it.
    """
    data = datasets.FashionMNIST(
        root='data', train=train, download=True, transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)),
    )
    
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader


class NeuralNetwork(nn.Module):
    """
    The neural network that we will train.
    """
    
    def __init__(self, inputs_shape: tuple[int, int], outputs_shape: tuple[int, int]) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(inputs_shape[0] * inputs_shape[1], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, outputs_shape[1]),
        )
    
    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.flatten(inputs)
        logits = self.stack(inputs)
        return logits


def get_labels_map() -> dict:
    labels_map = {
        0: 'T-Shirt',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle Boot',
    }
    return labels_map


def try_model(model, inputs_shape: tuple[int, int]) -> None:
    """
    Tries the model with random input.
    """
    inputs = torch.randn(1, inputs_shape[0], inputs_shape[1])

    logits = model(inputs)
    preds = nn.Softmax(dim=1)(logits)

    label = preds.argmax(dim=1).item()
    labels_map = get_labels_map()
    print(f'Label: {labels_map.get(label)}')


def main():
    batch_size: int = 8
    train_dataloader = dataloader(train=True, batch_size=batch_size)
    test_dataloader = dataloader(train=False, batch_size=batch_size)

    labels_map = get_labels_map()

    # let's see an image from the input
    train_features, train_labels = next(iter(train_dataloader))  # fetch one batch from the dataloader
    random_ix = torch.randint(0, batch_size, (1,)).item()
    label = train_labels[random_ix]
    
    img = train_features[random_ix].squeeze(dim=0)
    plt.imshow(img, cmap='gray')
    plt.title(
        labels_map.get(label.argmax(dim=0).item()),
    )
    plt.show()

    model = NeuralNetwork(
        inputs_shape=train_features[1].shape,
        outputs_shape=train_labels.shape,
    )

    # try the neural network on a random input
    try_model(model, train_features[1].shape)

if __name__ == '__main__':
    main()
