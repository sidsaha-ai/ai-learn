import torch
from torch import nn, Tensor
from torch.nn import functional as F

class LogRMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1e-5  # to avoid log(0)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # compute the logarithm of the predicted and true values
        # add the eps so that the value is non-zero, because log(0) is undefined
        log_y_pred: Tensor = torch.log(y_pred + self.eps)
        log_y_true: Tensor = torch.log(y_true + self.eps)

        # compute the mean squared error
        mse: Tensor = F.mse_loss(log_y_pred, log_y_true)

        # compute the root mean squared error
        rmse: Tensor = torch.sqrt(mse)
        
        return rmse
