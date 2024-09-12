"""
Experiment Objective
====================

The objective of this experiement is to arrive at a vocab size to use in the BPE Tokenizer. The size of the vocabulary determines how many
unique tokens will be used to represent the text. A smaller vocabulary size will break down more words in sub-words and could help in
better generalization, but can lose context about potential rare and important words. A larger vocabulary size will keep more words
intact, but might increase model complexity.

Experiment Setup
================

Let's try with the following vocab sizes - 

1. (Small) 10K tokens
2. (Small) 15K tokens
3. (Small) 20K tokens
4. (Medium) 30K tokensx
5. (Medium) 40K tokens
6. (Medium) 50K tokens
7. (Large) 60K tokens
8. (Large) 80K tokens
9. (Large) 100K tokens

What to measure
===============

For each vocab size, let's measure the following:

1. Token Count - Pick one book (and potentially a subset of books), encode, and compare the number of tokens in 
each setup to gauge compression efficiency.

2. UNK Token Count - Pick the same book, encode, and compare the number of `<UNK>` tokens in each setup to see how well
rare words are handled.

3. Token Diversity - After each encoding, count how many distinct tokens were used in the book to understand the
diversity and utility of the vocabulary.

5. UNK Token Positioning - Analyze where `<UNK>` tokens occur to determine if any critical or common terms are being represented by `<UNK>`.
"""