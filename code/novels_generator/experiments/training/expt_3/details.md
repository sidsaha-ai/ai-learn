## Experiment
From the previous experiment, we saw that increasing the model complexity by increasing the number of layers and increasing the number of self-attention heads did not help significantly.

In this experiment, let's try increasing the model complexity by increasing the dimensionality of the feed forward layers. This might help in representing higher dimensional data.

We also saw from the previous experiment that towards the later epochs, the validation loss slightly starts to increase and hence maybe indicating overfitting (but barely). We also see very less improvement after epoch 20. So, in this experiment we will train for 20 epochs.

We will do the following - 

- **Increase the feed-forward layer** - Increase the feed-forward layer from 1024 to 2048.

## Hyperparamters
As per the above, we will use the following hyperparameters for this experiment - 
- **Context Length**: 128
- **Batch Size**: 32
- **Vocab Size**: 40000
- **Embeddings Size**: 256
- **Number of self-attention heads**: 6
- **Number of transformer block layers**: 4
- **Feed Forward Size**: 2048
- **Number of epochs to train**: 20

## Results
Below are the training and validation losses.
```
Epoch: 0, LR: 0.0001, Train Loss: 7.6822, Val Loss: 6.4531, Loss Diff: -1.2291
Epoch: 1, LR: 0.0001, Train Loss: 5.8468, Val Loss: 6.0024, Loss Diff: 0.1556
Epoch: 2, LR: 0.0001, Train Loss: 5.4775, Val Loss: 5.7764, Loss Diff: 0.2989
Epoch: 3, LR: 0.0001, Train Loss: 5.2644, Val Loss: 5.6449, Loss Diff: 0.3804
Epoch: 4, LR: 0.0001, Train Loss: 5.1313, Val Loss: 5.5793, Loss Diff: 0.4480
Epoch: 5, LR: 0.0001, Train Loss: 5.0400, Val Loss: 5.5575, Loss Diff: 0.5175
Epoch: 6, LR: 0.0001, Train Loss: 4.9684, Val Loss: 5.4862, Loss Diff: 0.5178
Epoch: 7, LR: 0.0001, Train Loss: 4.9094, Val Loss: 5.4766, Loss Diff: 0.5673
Epoch: 8, LR: 0.0001, Train Loss: 4.8589, Val Loss: 5.4644, Loss Diff: 0.6055
Epoch: 9, LR: 0.0001, Train Loss: 4.8131, Val Loss: 5.4375, Loss Diff: 0.6244
Epoch: 10, LR: 0.0001, Train Loss: 4.7709, Val Loss: 5.4157, Loss Diff: 0.6448
Epoch: 11, LR: 0.0001, Train Loss: 4.7322, Val Loss: 5.4069, Loss Diff: 0.6747
Epoch: 12, LR: 0.0001, Train Loss: 4.6969, Val Loss: 5.4224, Loss Diff: 0.7255
Epoch: 13, LR: 0.0001, Train Loss: 4.6634, Val Loss: 5.4074, Loss Diff: 0.7439
Epoch: 14, LR: 0.0001, Train Loss: 4.6311, Val Loss: 5.4033, Loss Diff: 0.7721
Epoch: 15, LR: 0.0001, Train Loss: 4.5998, Val Loss: 5.4104, Loss Diff: 0.8106
Epoch: 16, LR: 0.0001, Train Loss: 4.5699, Val Loss: 5.3911, Loss Diff: 0.8212
Epoch: 17, LR: 0.0001, Train Loss: 4.5407, Val Loss: 5.4041, Loss Diff: 0.8633
Epoch: 18, LR: 0.0001, Train Loss: 4.5140, Val Loss: 5.4044, Loss Diff: 0.8904
Epoch: 19, LR: 0.0001, Train Loss: 4.4860, Val Loss: 5.4186, Loss Diff: 0.9326
```

## Findings

As compared to previous experiment, we see the following - 

- **Initial losses**: The initial training and validation loss starts off better in this experiment.
- **Validation loss**: We end up with a slightly lower validation loss in this case as compared with previous experiments.
- **Loss Rate** - Both training and validation losses trend towards decreasing, so not indicating any overfitting yet.

This indicates that there is still some juice in making the network more complex. So, we will try making the model a little more complex in the next experiment as well.

Following is the loss graph - 
![Losses](loss.png)
