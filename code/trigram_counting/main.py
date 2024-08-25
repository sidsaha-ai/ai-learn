"""
The main file which will start the execution.
"""
import argparse
from trigram_counting.model import TrigramCountingModel


def _read_words(filepath: str) -> list:
    words: list = []
    with open(filepath, mode='r', encoding='utf-8') as f:
        words = f.read().splitlines()

    words = [w.lower().strip() for w in words]
    return words


def main(train_data_filepath: str) -> None:
    """
    The main executable function.
    """
    words = _read_words(train_data_filepath)

    model = TrigramCountingModel(words)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_filepath', required=True, type=str,
    )

    args = parser.parse_args()

    main(
        args.train_data_filepath,
    )
