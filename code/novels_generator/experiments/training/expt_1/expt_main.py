"""
The script to run this experiment.
"""
from novels_generator.code import train


def main():
    """
    The main function to start experiment execution.
    """
    num_epochs: int = 20

    train.train_model(num_epochs)


if __name__ == '__main__':
    main()
