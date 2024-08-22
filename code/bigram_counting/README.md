### Task

In this exercise, we want to build a bigram character model that can generate names. **Bigram character model** means that the model is trained in a way that it will take one character and will generate the next character in the sequence. 

### How to do this?

- We have a dataset of names in the `../data/names.txt` as the data on which we will train our model.
- We will do this with counting the pair of letters. We will add a start character and an end character, which is `.`, at the start and end of each word.
- We will create a matrix of pairs of letters and count it over all the words.
- We will convert the count into probabilities row-wise, which essentially means, the matrix will contain the probabilities of the next letter, given the preceding letter.
- We will use the above probability matrix to generate words.