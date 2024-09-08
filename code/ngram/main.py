"""
The main file to start execution.
"""
import argparse

from ngram.model import NGramModel
from ngram.new_model import NewNgramModel


def _read_words(filepath: str) -> list:
    words: list = []

    with open(filepath, encoding='utf-8') as f:
        words = f.read().splitlines()

    words = [w.lower().strip() for w in words]
    return words


def main(train_data_filepath: str, batch_size: int, num_epochs: int) -> None:
    """
    The main method to start execution.
    """
    words: list = _read_words(train_data_filepath)

    model = NGramModel(words, batch_size)
    model.train(num_epochs)

    train_loss: float = model.train_loss()
    print(f'Overall training loss: {train_loss:.4f}')

    dev_loss: float = model.dev_loss()
    print(f'Dev Loss: {dev_loss:.4f}')

    test_loss: float = model.test_loss()
    print(f'Test Loss: {test_loss:.4f}')

    for _ in range(20):
        word: str = model.predict()
        print(word)


def new_main(train_data_filepath: str, batch_size: int, num_epochs: int) -> None:
    """
    Main method that starts execution using the new model.
    """
    words: list = _read_words(train_data_filepath)

    model = NewNgramModel(words, batch_size)
    model.train(num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_filepath', required=True, type=str,
    )
    parser.add_argument(
        '--batch_size', required=True, type=int,
    )
    parser.add_argument(
        '--num_epochs', required=True, type=int,
    )
    parser.add_argument(
        '--type', required=True, choices=['old', 'new'],
    )

    args = parser.parse_args()

    fn = main if args.type == 'old' else new_main
    fn(
        args.train_data_filepath, args.batch_size, args.num_epochs,
    )
