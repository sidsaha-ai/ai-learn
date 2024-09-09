"""
This implements the cross entropy loss from scratch.
"""

import torch
from torch import Tensor


class CrossEntropy:
    """
    Custom cross-entropy loss function.
    """

    def __init__(self) -> None:
        super().__init__()

        self.logits = None
        self.targets = None
        self.output = None

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Computes the cross-entropy loss.
        """
        self.logits = logits
        self.targets = targets

        N = logits.shape[0]  # the size of the mini-batch

        softmax = torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)
        target_probs = torch.log(
            softmax.gather(1, targets.unsqueeze(1)).squeeze(),
        )
        mean = torch.sum(target_probs) / N

        self.output = (-1) * mean

        return self.output
