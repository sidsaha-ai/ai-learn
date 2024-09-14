## Objective
After the first experiment, in this experiment, we will try to improve the loss by increasing the model complexity and tuning the learning rate. 

We will do the following things -

- **Increase the model complexity**: Increase the number of tranformer layers from 2 to 4 and increase the number of self attention heads from 4 to 8. (Because we are keeping the number of embeddings same as before to 256, number of attention heads must squarely divide the number of embeddings.)
- **Increase the number of epochs**: From the results of experiment #1, we see that we are not overfitting. As we are also increasing the model complexity, let's train for 30 epochs (as opposed to 20 epochs in experiment #1).
- **Tune the learning rate**: In Experiment #1, we used a learning rate of 1e-4 for 20 epochs. In this experiment, we will use the learning rate if 1e-4 for the first 25 epochs and then use 1e-6 fot the next 5 epochs.

## Hyperparameters
As per the above, we will use the following hyperparameters for this experiment -
- **Context Length**: 128
- **Batch Size**: 32
- **Vocab Size**: 40000
- **Embeddings Size**: 256
- **Number of self-attention heads**: 6
- **Number of transformer block layers**: 4
- **Feed Forward Size**: 1024
- **Number of epochs to train**: 20


We are hoping to get a better training and validation loss as compared to experiment #1.
