"""
Trying out stuff from Pytorch's turtorial chapter on datasets and dataloaders.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from matplotlib import pyplot as plt

def main():
    train_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)),
    )
    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)),
    )

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

    figure = plt.figure(figsize=(6, 6))
    cols, rows = 3, 3

    for ix in range(1, cols * rows + 1):
        sample_ix = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_ix]

        figure.add_subplot(rows, cols, ix)
        plt.title(
            labels_map.get(label.argmax(dim=0).item()),
        )
        plt.axis('off')
        plt.imshow(
            img.squeeze(), cmap='gray',
        )
    
    plt.show()

    # dataloaders
    batch_size = 8
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))  # fetch one batch from the dataloader

    random_ix = torch.randint(0, batch_size, (1,)).item()
    label = train_labels[random_ix]
    
    img = train_features[random_ix].squeeze(dim=0)
    plt.imshow(img, cmap='gray')
    plt.title(
        labels_map.get(label.argmax(dim=0).item()),
    )
    plt.show()


if __name__ == '__main__':
    main()
