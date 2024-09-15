"""
This script is to run the experiment.
"""

from novels_generator.code import train
from novels_generator.code.constants import Hyperparamters


def main():
    """
    The main function to run this experiment.
    """
    num_epochs: int = 30

    # hyperparameters
    Hyperparamters.CONTEXT_LENGTH = 128
    Hyperparamters.BATCH_SIZE = 32
    Hyperparamters.VOCAB_SIZE = 40000
    Hyperparamters.EMBEDDING_SIZE = 512
    Hyperparamters.SELF_ATTENTION_HEADS = 8
    Hyperparamters.NUM_LAYERS = 4
    Hyperparamters.FEED_FORWARD_SIZE = 2048

    def lr_lambda(epoch):
        # base LR in the model is 1e-4.
        # first 5 epochs, LR should be 1e-5. next 20 epochs, LR should be 1e-4. last 5 epochs, LR should be 1e-6.
        res = 1
        if epoch < 5:
            res = 1e-1
        elif epoch < 20:
            res = 1
        else:
            res = 1e-2
        return res
    
    train.train_model(
        num_epochs, lr_scheduler_type='LambdaLR', lr_lambda=lr_lambda,
    )


if __name__ == '__main__':
    main()
