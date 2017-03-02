# mnist-tensor
Experimenting with MNIST Database Recognition via Tensorflow

## Requirements

Requires Tensorflow: https://www.tensorflow.org/install/

## Brief Overview of Solutions:

There are a few different solutions available: 

* mnist_simple.py
  * Uses only a single fully-connected layer. Lowest accuracy.
* mnist_naive_layers.py
  * Uses 6 fully-connected layers to improve accuracy over the previous solution.
* mnist_improved_layers.py
  * Uses 6 fully-connected layers with dropout and a learning rate decay to improve accuracy, and reduce overfitting, compared to the previous solution.
* mnist_convolutional.py
  * Uses a convolutional network with 3 convolutional layers followed by a fully-connected layer with dropout. Maintains the highest accuracy of any solution at ~99%.
