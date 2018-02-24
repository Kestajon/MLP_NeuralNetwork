# MLP_NeuralNetwork

Neural Network implementation in python 3, which supports any number of layers (within reason). This also requires the numpy package for using the neural network, but also the matplotlib package to draw graphs used in testing the neural network.

A neural network using this system is created by creating an instance of the NeuralNetwork class, which contains the __init__ function signature:

def __init__(self, networkTopology, inputNormalization, outputNormalization, inverseNormalization, Lambda = 0)

the networkTopology parameter passes in the whole network topology, passed in in the form of:

[inputs, any number of hidden layers, outputs]

for example:

[2, 3, 2, 1] would create a neural network which takes in two inputs, has two hidden layers, the first with three neurones and the second with two neurones and then a final output layer which has only one neurone, producing a single output.

The inputNormalization, outputNormalization and inverseNormalization are all functions which are passed in in order to define how to convert data to and from the neural networks' representation. As this is dependent on the data, these are defined by the user. The input normalization is applied to any inputs entered into the network, the output normalization is applied to any outputs entered into the network for use in the supervised training, and the inverse normalization converts any inputs into the expected output style. Therefore the inverseNormalization function should always be the inverse of the outputNormalization function.

For example, in the standard loadData function the following functions are used to convert the inputs into a normal distributon around zero. 

    def inputNormalization(X):
        return (X - meanValue[:inputLength]) / maxValue[:inputLength]

    def outputNormalization(y):
        return (y - meanValue[inputLength]) / maxValue[inputLength]

    def inverseNormalization(y):
        return y*maxValue[inputLength] + meanValue[inputLength]
        
Where the inputLength is the length of the data.

The final term in the initialization function is the lamba term, this is the regularization constant, which defines how much the sum of the square of the weights will impact the cost function. 

The other important features of the neural network are input during the training phase, with a call to the stochastic function which implements a stochastic back-propogation algorithm using gradient descent, which has the signature

def stochastic(self, X, y, validX, validY, learningRate=0.2, epochs=15)

Where X is the inputs for the training data-set, y is the outputs for the training data-set, validX is the inputs for the validation data-set, validY is the output for the validation data-set, the learningRate is the learning rate used during traning and the epochs is the number of iteratons that the training will run for.

this function returns four values: trainingCost, validationCost, trainingCostNL, validationCostNL

The trainingCost is the cost value associated with the training data-set, and the validationCost is the cost value associated with the validation data-set. The NL versions of these costs are the costs without the regularization terms, that is the cost associated purely with the data. 

# Tests

This neural network was tested by experimenting with predictions of the S&P 500 using data from April 29, 2011 to December 02, 2016. These tests are outlined in the tests folder. 
