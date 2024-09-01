"""
The main file to start execution.
"""
import argparse

from ngram.model import NGramModel


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

    for _ in range(1):
        word: str = model.predict()
        print(word)


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

    args = parser.parse_args()

    main(
        args.train_data_filepath,
        args.batch_size,
        args.num_epochs,
    )
