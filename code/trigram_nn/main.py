import argparse

from trigram_nn.model import TrigramNN

def _read_words(filepath: str) -> list:
    words: list = []

    with open(filepath, encoding='utf-8') as f:
        words = f.read().splitlines()
    
    words = [w.strip().lower() for w in words]
    return words

def main(train_data_filepath: str, num_epochs: int) -> None:
    words: list = _read_words(train_data_filepath)
    print(f'Num words: {len(words)}')

    model = TrigramNN(words)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_filepath', required=True, type=str,
    )
    parser.add_argument(
        '--num_epochs', required=True, type=int,
    )

    args = parser.parse_args()

    main(
        args.train_data_filepath,
        args.num_epochs,
    )
