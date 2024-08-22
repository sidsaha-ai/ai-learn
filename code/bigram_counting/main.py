"""
The main file that is the executable.
"""
import argparse

from bigram_counting.model import BigramLM


def _read_words(filepath: str) -> list:
    # reads the file to load all the words in the train dataset.
    words: list = []
    with open(filepath, encoding='utf-8') as f:
        words = f.read().splitlines()

    return [w.lower() for w in words]  # make all words to lower case


def main(
    train_data_filepath: str,
) -> None:
    """
    The main method where execution starts.
    """
    words: list = _read_words(train_data_filepath)

    bigram_lm: BigramLM = BigramLM(input_words=words)

    # train the model
    bigram_lm.train()

    # let's predict 20 words
    for _ in range(20):
        word: str = bigram_lm.predict()
        print(word)

    # now let's print the loss of the model
    loss: float = bigram_lm.loss()
    print(f'Loss = {loss:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_filepath', required=True, type=str,
    )

    args = parser.parse_args()

    main(
        args.train_data_filepath,
    )
