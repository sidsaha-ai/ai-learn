import torch
import torch.nn as nn
from torch import Tensor


class SimpleNeuralNet(nn.Module):
    """
    Class that defines the neural network architecture. This class defines the layers and
    how to do the forward pass through the neural network.
    """

    def __init__(self):
        super().__init__()

        # define the layers
        self.layer1 = nn.Linear(3, 6)  # input layer takes 3 inputs and feeds 6 outputs to the next layer
        self.layer2 = nn.Linear(6, 4)  # 6 neurons feeds to 4 neurons in the next layer
        self.layer3 = nn.Linear(4, 2)  # 4 neurons feeds to 2 neurons in the next layer
        self.layer4 = nn.Linear(2, 1)  # output layer takes 2 inputs and outputs 1 output.

        self.activation = nn.Tanh()  # the activation function

    def forward(self, x: Tensor):
        """
        Defines how to do the forward pass through the network.
        """
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)
        return x