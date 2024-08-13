## Problem
In this problem, we will create a simple neural network using `pytorch` and train it with some inputs. Following defines the task that needs to be done.


### Neural Network Architecture
1. Create a neural network with the input layers with 3 neurons. These neurons will each take a scalar input.
2. There are 3 hidden layers. The first layer will hav3 6 neurons, the second layer has 4 neurons, and the third layer has 2 neurons.
3. The ouput layer has only one neuron, which is the final output of the neural network.

### Neuron
1. Each neuron is a linear function.
2. The activation function at each neuron is the `tanh` function.

### Loss Function
We must use the mean squared error (MSE) as the loss function of the overall neural network.

### Optimizer
We will use the schotastic gradient descent method to optimize the neural network.

### Training
The neural network will be trained with some input (3 scalars as per the network architecture). You can choose your inputs and target outputs as you wish, but for this task, they must be scalars and not tensors.
