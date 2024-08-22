import argparse

from bigram_nn.model import BigramNN

def _read_words(filepath: str) -> list:
    words: list = []

    with open(filepath, encoding='utf-8') as f:
        words = f.read().splitlines()
    
    return [w.lower() for w in words]

def main(train_data_filepath: str) -> None:
    words: list = _read_words(train_data_filepath)

    bigram_nn = BigramNN(words)
    bigram_nn.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_filepath', required=True, type=str,
    )

    args = parser.parse_args()

    main(
        args.train_data_filepath,
    )
