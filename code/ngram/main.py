"""
The main file to start execution.
"""
import argparse


def main(train_data_filepath: str, num_epochs: int) -> None:
    print(f'{train_data_filepath=}')
    print(f'{num_epochs=}')


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
