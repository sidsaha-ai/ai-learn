## Objective
From the previous experiments, we saw very small improvements. In this experiments, let's try multiple things at once to bump up improvements. We will try the following things - 

#### Increase model complexity
Let's increase the model complexity so that the model learns more complex patterns in the data. To do this, let's tune the following hyperparameters - 
- **Increase the number of transformer layers** from `4` to `8` so that we add depth to capture more complex patterns.
- **Increase the feed-forward size** to `4096`, so that the model again learns more deeply.

#### Learning Rate scheduling
Let's also experiment with learning rate scheduling. Let's do the following things - 
- Like previous experiments, we will start warming up the learning rate from 1e-5 for the first 5 epochs, but with **cosine annealing**.
- Then, we will increase the learning rate to 5e-4 (from previously of 1e-4) so that the model can converge a little faster than before.
- Then, we will use **cosine annealing** to decay the learning rate more smoothly rather than abruptly changing it.

#### Dropout
To bring in a little more regularization, let's increase the dropout to `0.3`.

#### Layer Drop
To bring in more regularization, probabilitistically drop some layers from the tranformer layers during training. Let's set this probability to `0.1`.
