"""
Script to run through with the quick tour guide of Transformers on hugging face.
"""
from transformers import pipeline


def main():
    """
    The main function to try out examples.
    """
    classifier = pipeline('sentiment-analysis')

    # e.g. 1
    output = classifier('We are very happy to show you the transformes library')
    print(output)


if __name__ == '__main__':
    main()
