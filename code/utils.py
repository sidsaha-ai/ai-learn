from torch import Tensor

def print_loss(epoch: int, loss: Tensor) -> None:
    print(f'{epoch=}, loss={loss.item():.4f}')
