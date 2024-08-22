import pandas as pd
import argparse


def _read_words(filepath: str) -> list:
    # reads the file to load all the words in the train dataset.
    words: list = open(filepath).read().splitlines()
    return words


def main(
    train_data_filepath: str,
) -> None:
    words: list = _read_words(train_data_filepath)
    print(words[0:10])
    print(len(words))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_filepath', required=True, type=str,
    )
    
    args = parser.parse_args()

    main(
        args.train_data_filepath,
    )
