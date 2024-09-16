## Objective
From the previous experiments, we saw very small improvements. In this experiments, let's try multiple things at once to bump up improvements. We will try the following things - 

#### Increase model complexity
Let's increase the model complexity so that the model learns more complex patterns in the data. To do this, let's tune the following hyperparameters - 
- **Increase the number of transformer layers** from `4` to `8` so that we add depth to capture more complex patterns.
- **Increase the feed-forward size** to `4096`, so that the model again learns more deeply.