# About

We use Neural Network all the time, but have you ever implemented your own? Learning the mathematical details of how it works internally is truly an amazing experience and I suggest you trying too!

This is a vanilla Recurrent Neural Network with backpropagation based on the sklearn structure. 

# How to use

First, you'll need to instantiate the NN class with the desired parameters and then `fit()` the network using inputs and targets. After that, the `predict()` method can be used to take a look at how the neurons react based on the input.

The class topology must be built using the format `[X,Y,Z,...]` where `X`, `Y`, and `Z` are the number of neurons of the respective layer (X neurons for layer 0, Y neurons for layer 1, Z neurons for layer 2 and so on). The network is trained by the `fit()` function and `predict()` returns the output layer.

## Example

```
from vanillaNN import VanillaNeuralNetwork

myNeuralNetwork = VanillaNeuralNetwork([4,4,3],numberOfEpochs=2500)
myNeuralNetwork.fit(input_train,targets_train, testAccuracy= True ,testInputs = input_test, testTargets= targets_test)
myNeuralNetwork.predict(input_test[0])
myNeuralNetwork.accuracy(input_test,targets_test)

```

# Requirements: 

- Python 3
- Numpy
- sklearn
- Matplotlib

