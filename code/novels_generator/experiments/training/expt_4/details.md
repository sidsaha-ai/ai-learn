## Objective
In the last experiment, we saw that increasing the feed forward size help a little. Let's now increase the model complexity by increasing the embedding size (to try to more effectively represent the input) and play with the learning rate a bit.

We will do the following things - 

- **Increase Embedding Size**: Let's increase the embedding size from 256 to 512.
- **Learning Rate Warmup**: We generally trained at 1e-4 and then decreased to 1e-6 in later epochs in the previous experiment(s). In this experiment, let's start with a learning rate of 1e-5 for the first 5 epochs, then do the learning rate of 1e-4 for 20 epochs, and then finish the last 5 epochs with a learning rate of 1e-6. 
- **Change number of epochs**: Because we are trying with 3 different learning rates, let's train for 30 epochs to give space to different learning rates.

## Hyperparameters
As per the above, we will use the following hyperparameters for this experiment -
- **Context Length**: 128
- **Batch Size**: 32
- **Vocab Size**: 40000
- **Embeddings Size**: 512
- **Number of self-attention heads**: 8
- **Number of transformer block layers**: 4
- **Feed Forward Size**: 2048
- **Number of epochs to train**: 30

## Results
