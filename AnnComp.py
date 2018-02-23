import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import csv
import pickle # Used to serialise Neural_Networks for later use

# ANN Neural Network Class, adapted from stephencwelch's code on GitHub
# https://github.com/stephencwelch/Neural-Networks-Demystified
#
# This Neural_Network class has a constant output layer sizes
# but variable hidden layer size, number of hidden layers and input
# layer size. The Neural_Network makes use of a activation function as its'
# activation function, and Sum Squared Error as its' cost function for
# training
#
# @author Jonathan Keslake

class NeuralNetwork(object):

    # The passed in networkTopology describes the neural network,
    # it is a list of integers, with the position of the integer
    # representing the layer, and the integer representing the number
    # of neurons in that layer
    def __init__(self, networkTopology, inputNormalization, outputNormalization
                                        , inverseNormalization, Lambda = 0):
        # Define Network Topology
        self.networkTopology = networkTopology

        self.Lambda = Lambda

        # Set normalization equations
        self.inputNormalization = inputNormalization
        self.outputNormalization = outputNormalization
        self.inverseNormalization = inverseNormalization

        # weightList contains a list of the weights for the system
        self.weightList = []

        for i in range(len(networkTopology)-1):
            self.weightList.append(np.random.normal(0, self.networkTopology[i]**-0.5,
                    (self.networkTopology[i], self.networkTopology[i+1])))

    # Activation function for the neural network
    def activation(self, z):
        # Apply activation activation function to scalar, vector, or matrix
        return 1.7159*np.tanh((2.0/3)*z)

    # Differential of the activation function
    def activationPrime(self, z):
        return 1.14393*(1.0 - np.tanh((2.0/3)*z)**2) # Check This calculation

    # Compute costs from approximations and expected results, using the current state of the system
    def cost(self, X, y):
        X = self.inputNormalization(X)
        y = self.outputNormalization(y)
        weightSquareSum = 0
        for i in nn.weightList:
            weightSquareSum += np.sum(i**2)
        return 0.5*sum((y-self.costPredict(X))**2)/X.shape[0] + (self.Lambda/2)*weightSquareSum

    # Function that calculates the target from normalised data, used within the class, not for
    # use by users of the class
    def costPredict(self, X):
        preWeight = X
        for weight in self.weightList:
            # Inputs to the layer are combinations of the preWeights
            # and the weights
            layerInput= np.dot(preWeight, weight)

            # preWeight for next layer is the layerInput passed
            # through the activation function
            preWeight = self.activation(layerInput)

        # At the end of the loop, the preWeight will contain the Y values
        return preWeight

    # Function to obtain the current layer z, and the previous layers a
    # Specialised for use in the costFunctionPrime function
    def forward(self, X, layerToReach):

        # Set up the 1st layer (input -> 1st Hidden)
        previousA = X
        layerZ = np.dot(previousA, self.weightList[0])

        # Only run if the layerToReach does not equal 0
        # If it runs through the whole network layerZ will equal yHat
        for j in range(layerToReach):
            previousA = self.activation(layerZ)
            layerZ = np.dot(previousA, self.weightList[j+1])

        # Return
        return layerZ, previousA

    # Function to calculate the dJdW in order to update the individual layer weights
    def costFunctionPrime(self, X, y):

        #Generate final layer z (input to final layer) and output of previous
        layerZ, previousA = self.forward(X, len(self.networkTopology)-2)

        #yHat is the activation of the input to the final layer
        yHat = self.activation(layerZ)

        #Final delta
        delta = np.multiply(-(y-yHat), self.activationPrime(layerZ))

        # dJdW list instantiation
        dJdW = [0]*(len(self.networkTopology)-1)

        # Set final dJdW from previous, iterate over next for subsequent
        dJdW[len(self.networkTopology)-2] = np.dot(previousA.T,delta) + self.Lambda*self.weightList[
                                                                        len(self.networkTopology)-2]

        for i in range(len(self.networkTopology)-3, -1, -1):
            layerZ, previousA = self.forward(X, i)
            delta = np.dot(delta, self.weightList[i+1].T) * self.activationPrime(layerZ)
            dJdW[i] = np.dot(previousA.T,delta) + self.Lambda*self.weightList[i]
        return dJdW

    # Function for training the network using a batch back-propagation method
    def backPropogationTraining(self, X, y, validX, validY, learningRate=0.2, epochs=30):
        X = self.inputNormalization(X)
        y = self.outputNormalization(y)
        validX = self.inputNormalization(validX)
        validY = self.outputNormalization(validY)

        trainingCost = []
        validationCost = []
        trainingCostNL = []
        validationCostNL = []

        for i in range(epochs):

            trainingCost.append(self.t_cost(X, y))
            validationCost.append(self.t_cost(validX, validY))
            trainingCostNL.append(self.nl_cost(X, y))
            validationCostNL.append(self.nl_cost(validX, validY))
            dJdW = self.costFunctionPrime(X,y)
            for j in range(len(dJdW)):
                self.weightList[j] = self.weightList[j] - learningRate*dJdW[j]
        return trainingCost, validationCost, trainingCostNL, validationCostNL

    # Function used to obtain the networks predictions for input X
    def predict(self, X):
        preWeight = self.inputNormalization(X)
        for weight in self.weightList:
            # Inputs to the layer are combinations of the preWeights
            # and the weights
            layerInput= np.dot(preWeight, weight)

            # preWeight for next layer is the layerInput passed
            # through the activation function
            preWeight = self.activation(layerInput)

        # At the end of the loop, the preWeight will contain the Y values
        return self.inverseNormalization(preWeight)

    # Cost function associated with the system, with the lambda generalisation term
    def t_cost(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        weightSquareSum = 0
        for i in nn.weightList:
            weightSquareSum += np.sum(i**2)
        return 0.5*sum((y-self.costPredict(X))**2)/X.shape[0] + (self.Lambda/2)*weightSquareSum

    # Cost function associated with the system, without the lambda generalisation term
    def nl_cost(self, X, y):
        return 0.5*sum((y-self.costPredict(X))**2)

    # Function for training the network using a stochastic back-propagation method
    def stochastic(self, X, y, validX, validY, learningRate=0.2, epochs=15):
        X = self.inputNormalization(X)
        y = self.outputNormalization(y)
        validX = self.inputNormalization(validX)
        validY = self.outputNormalization(validY)

        trainingCost = []
        validationCost = []
        trainingCostNL = []
        validationCostNL = []

        # Add cost list so costs can be tracked over iterations
        # Possibly also over index in indexList once for tracking purposes?

        innerLayer = self.networkTopology[0]
        outputLayer = self.networkTopology[len(self.networkTopology)-1]
        indexList = np.arange(len(X))

        if epochs >= 50:
            spaces = 50
            hashTime = int(epochs/spaces)
        else:
            spaces = epochs
            hashTime = 1
        print("Training Neural Network")
        print("|" + "-"*spaces + "|")
        sys.stdout.write(" ")

        counter = 0
        for i in range(epochs):
            trainingCost.append(self.t_cost(X, y))
            validationCost.append(self.t_cost(validX, validY))
            trainingCostNL.append(self.nl_cost(X, y))
            validationCostNL.append(self.nl_cost(validX, validY))

            np.random.shuffle(indexList)
            for index in indexList:

                np.reshape(X[index], (1, innerLayer))
                dJdW = self.costFunctionPrime(np.reshape(X[index], (1, innerLayer))
                                             ,np.reshape(y[index], (1, outputLayer)))
                for j in range(len(dJdW)):
                    self.weightList[j] -= learningRate*dJdW[j]

            # Handling Training Progress Bar
            counter += 1
            if counter >= hashTime:
                sys.stdout.write ("#"*int((counter/hashTime)))
                counter = 0

        print("")

        return trainingCost, validationCost, trainingCostNL, validationCostNL

# Load files from text as an np array
def loadFromFile(filename):
    return np.genfromtxt(filename, delimiter=',')


# Function which instantiates an instance of the NeuralNetwork class shown above using the parameters
# passed in as keyword arguments. Loads the data from the "datao.csv" file, which is a manipulated
# version of the stock-market data. This function also takes care of the normalisation functions
# associated with the data and inputs them into the network, allowing raw data to be passed in to the
# system.
def loadData(inputLength = 1, outputPos = 5, trainingOffestFromBottom = 3, validationOffsetFromBottom = 3,
             inputLambda = 0, hiddenLayers = [15]):

    # Loading Training Data
    # Assume the expected output is always the last element
    X = loadFromFile("datao.csv")

    tempX = np.array(X[0][:inputLength], ndmin=2)
    tempy = np.array([X[0][outputPos]], ndmin=2)
    close = np.array([X[0][outputPos + 1]], ndmin=2)

    meanValue = np.zeros(inputLength + 1)
    meanValue[:inputLength] = X[0][:inputLength]
    meanValue[inputLength] = X[0][outputPos]

    maxValue = np.copy(meanValue)

    counter = 1
    for i in range(1,len(X) - trainingOffestFromBottom):
        tempX = np.append(tempX, np.array(X[i][:inputLength], ndmin=2), axis=0)
        tempy = np.append(tempy, np.array(X[i][outputPos], ndmin=2), axis=0)
        close = np.append(close, np.array(X[i][outputPos + 1], ndmin=2), axis=0)

        # Retrieve the new row from the csv file
        newRow = np.zeros(inputLength + 1)
        newRow[:inputLength] = X[i][:inputLength]
        newRow[inputLength] = X[i][outputPos]

        # Add newRow to the average being counted
        meanValue += newRow

        for j in range(inputLength + 1):
            if maxValue[j] < newRow[j]:
                maxValue[j] = newRow[j]
            if maxValue[j] < -1 * newRow[j]:
                maxValue[j] = -1 * newRow[j]

        counter += 1

    # Calculate mean from average
    meanValue /= counter

    # Define the normalization functions to be used within the network
    def inputNormalization(X):
        return (X - meanValue[:inputLength]) / maxValue[:inputLength]

    def outputNormalization(y):
        return (y - meanValue[inputLength]) / maxValue[inputLength]

    def inverseNormalization(y):
        return y*maxValue[inputLength] + meanValue[inputLength]

    nn = NeuralNetwork([inputLength] + hiddenLayers + [1], inputNormalization, outputNormalization
                       , inverseNormalization, Lambda=inputLambda)

    # Loading Validation Data
    X = loadFromFile("validation.csv")

    validX = np.array(X[0][:inputLength], ndmin=2)
    validy = np.array([X[0][outputPos]], ndmin=2)
    validClose = np.array([X[0][outputPos + 1]], ndmin=2)

    for i in range(1,len(X) - validationOffsetFromBottom):
        validX = np.append(validX, np.array(X[i][:inputLength], ndmin=2), axis=0)
        validy = np.append(validy, np.array(X[i][outputPos], ndmin=2), axis=0)
        validClose = np.append(validClose, np.array(X[i][outputPos + 1], ndmin=2), axis=0)

    return nn, tempX, tempy, validX, validy, close, validClose


# Statements used to carry out the learning rate tests within the report
'''
# Learning Rate Tests
plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('SSE Cost')
plt.grid(1)
learningRateTests = [0.01, 0.001, 0.0001]

for element in learningRateTests:
    nn, X, y, validX, validy, close, validClose = loadData(inputLength=3, inputLambda=0.001, hiddenLayers=[6, 7, 4, 5, 2])
    trainingCost, validationCost, trainingCostNL, validationCostNL = nn.stochastic(X, y, validX, validy, epochs=2000, learningRate=element)
    plt.plot(validationCostNL, label='Topology = ' + str(element))
plt.legend()
plt.show()
'''
# Statements used to carry out the hidden layer tests within the report
'''
# Hidden Layer Tests
plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('SSE Cost')
plt.grid(1)
#plt.savefig('costPlot.png')
#hiddenLayerTests = [[0], [0.0005], [0.00075], [0.001], [0.00125]]
hiddenLayerTests = [[15], [6, 7, 4, 2], [10, 5, 3, 4], [8, 7, 4, 2], [6, 7, 4, 5, 2], [10, 5, 3, 4, 2], [10, 5, 3, 4, 2, 2,], [10, 5, 3, 4, 1]]

for element in hiddenLayerTests:
    nn, X, y, validX, validy, close, validClose = loadData(inputLength=3, inputLambda=0.001, hiddenLayers=element)
    trainingCost, validationCost, trainingCostNL, validationCostNL = nn.stochastic(X, y, validX, validy, epochs=600, learningRate=0.001)
    legendString = '3, '
    for i in element:
        legendString += str(i) + ', '
    legendString += ' 1'
    plt.plot(validationCostNL, label='Topology = ' + legendString)
plt.legend()
plt.show()
'''

# Statements used to carry out the lambda tests within the report
'''
# Lambda Tests 96 88
plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('SSE Cost')
plt.grid(1)
#plt.savefig('costPlot.png')
lambdaTests = [0, 0.0005, 0.00075, 0.001, 0.00125]
for element in lambdaTests:
    nn, X, y, validX, validy, close, validClose = loadData(inputLength=3, inputLambda=element)
    trainingCost, validationCost, trainingCostNL, validationCostNL = nn.stochastic(X, y, validX, validy, epochs=400, learningRate=0.001)
    plt.plot(validationCostNL, label='Lambda = ' + str(element))
plt.legend()
plt.show()
'''

# Statements used to carry out the input tests within the report
'''
# Input Tests
plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('SSE Cost')
plt.grid(1)
#plt.savefig('costPlot.png')
for i in range(1,6):
    nn, X, y, validX, validy, close, validClose = loadData(inputLength=i)
    trainingCost, validationCost, trainingCostNL, validationCostNL = nn.stochastic(X, y, validX, validy, epochs=250, learningRate=0.001)
    plt.plot(validationCostNL, label='Number of Inputs = ' + str(i))
plt.legend()
plt.show()
'''

# Statements used to obtain the three graphs shown at the end of each test:

# Network instantiation using loadData function
nn, X, y, validX, validy, close, validClose = loadData(inputLength=3, inputLambda=0.001, hiddenLayers=[6, 7, 4, 5, 2])

# Call stochastic training method
trainingCost, validationCost, trainingCostNL, validationCostNL = nn.stochastic(X, y, validX, validy, epochs=200, learningRate=0.001)

# Validation Set Untreated
plt.plot(nn.predict(validX), label='Prediction')
plt.plot(validy, label = 'Actual')
plt.xlabel('Dataset Number')
plt.ylabel('(High/Close) - 1')
plt.grid(1)
plt.legend()
plt.savefig('valSetUn.png')
plt.show()

# Cost Plotting
plt.figure(1)
plt.plot(trainingCostNL, label='Training Data')
plt.plot(validationCostNL, label='Validation Data')
plt.xlabel('Epoch')
plt.ylabel('SSE Cost')
plt.grid(1)
plt.legend()
plt.savefig('costPlot.png')
plt.show()

# Prediction vs. Expected
totalX = np.concatenate((X, validX))
totaly = np.concatenate((y, validy))
totalClose = np.concatenate((close, validClose))
plotVals = (nn.predict(totalX) - totaly)*totalClose

some = np.absolute(plotVals)
mean = np.mean(some)

# Calculate standard deviation
stdv = 0
for x in some:
    stdv += (x - mean)*(x - mean)
stdv = np.sqrt(stdv/some.size)

print(stdv)
print(mean)

plt.figure(1)
plt.plot(plotVals)
plt.xlabel('Dataset Number')
plt.ylabel('Prediction - Total')
plt.grid(1)
plt.savefig('predic_-_expec.png')
plt.show()

