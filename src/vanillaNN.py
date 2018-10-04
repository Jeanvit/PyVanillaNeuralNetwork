'''
Author: Jean Vitor de Paulo
Coded on Sublime text, 02/09/2018
'''


import numpy as np
import math
import logging
import sys
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------------------------
class VanillaNeuralNetwork(object):
	'''
	| This is the class that creates the neural network.
	|	Atrributes : 
	|	Topology: the format of the network. Each position of this list represents the number of neurons of the respective layer 
	|	numberOfLayers: The size of topology
	|	learningRate: The learning rate of the network
	|	weights: The weights of the network. Instantiated randomly in the constructor
	|	layers: The neural network itself. Each position of a layer is an abstraction of a neuron
	|	deltaError: The delta error of all iterations
	|	currentError: The current error of the network
	|	sizeOfWeights: The size of the weights matrix
	|	allSquaredErrors: stores the Root Mean Squared Error (RMS) of each Epoch
	|	epochAccuracy: stores the accuracy of each epoch
	|	epochRMS: The RMS of each epoch.
	'''
	def __init__(self, topology,  numberOfEpochs, learningRate=.1):
		self.topology = topology 				#
		self.numberOfLayers = len(topology)
		self.learningRate = learningRate
		self.weights = []
		self.layers = []
		self.deltaError = []
		self.currentError = 0
		self.sizeOfWeights = self.numberOfLayers - 1
		self.numberOfEpochs = numberOfEpochs
		self.allSquaredErrors = []
		self.epochAccuracy = []
		self.epochRMS = []

		self.__buildNeuralNetwork();
		
#-----------------------------------------------------------------------------------------------------------------------
	def __buildNeuralNetwork(self):
		#This function is responsible for the creation of the layers and also the weights(as random values)
		
		print ("Creating Neural Network with the following topology: ", self.topology)
		self.layers.append(np.ones(self.topology[0]+1))		# generate the neurons, with an additional one, the bias
		for i in range(1,self.numberOfLayers):
			self.layers.append(np.ones(self.topology[i]))

		
		for i in range(self.sizeOfWeights): 
			tempWeights =  2*np.random.random((self.layers[i].size, self.layers[i+1].size)) - 1
			self.weights.append(tempWeights)
		print ("Network created!")


#-----------------------------------------------------------------------------------------------------------------------
	def __feedForward(self, input):
		#Function responsible for propagating the input values from the input layers all the way to the output

		try:
			self.layers[0][0:-1] = input  #the first layer now is equal the input data
		except Exception as err:
			logging.exception('The input data and the the first layer does not match sizes ')
			sys.exit(1)
		for i in range(1, self.numberOfLayers): #now propagate the data throughout all layers

			dotProduct = np.dot(self.layers[i-1],self.weights[i-1])
			self.layers[i] = self.__sigmoid(dotProduct) # calculating the sigmoid of the dot product of each layer (weighted sum)
		return self.layers[-1]					

#-----------------------------------------------------------------------------------------------------------------------
	def __calculateOutputError(self, targets):
		#Calculate the output layer error in relation to the targets

		try:
			self.currentError = targets - self.layers[-1]
			self.deltaError.append(self.currentError * self.__derivative(self.layers[-1]) ) #delta error of the output layer
			self.allSquaredErrors.append(math.sqrt((self.currentError*self.currentError).sum()/(len(self.layers[-1]))))
			#print (self.allSquaredErrors[-1])
		except Exception as err:
			logging.exception('The targets and the output layer of the Neural Network does not match sizes')

#-----------------------------------------------------------------------------------------------------------------------
	def __calculateHiddenLayersError(self):
		#Calculate the hidden layers errors
		n = len(self.deltaError) -1
		for i in range(self.numberOfLayers-2, 0, -1): # -2 Indicates the first Hidden layer. This Loop goes through the Layers 
			layerDelta = np.dot(self.deltaError[-1], self.weights[i].T )* self.__derivative(self.layers[i])
			n = n -1
			self.deltaError.append(layerDelta) 
		

#-----------------------------------------------------------------------------------------------------------------------
	def __backPropagation(self, targets):
		#This functions calls the methods responsible for doing the backPropagation procedures
		self.__calculateOutputError(targets)
		self.__calculateHiddenLayersError()
		self.__updateNetworkWeights()

#-----------------------------------------------------------------------------------------------------------------------
	def __updateNetworkWeights(self):
		#Update all the weights of the network
		for i in range (self.sizeOfWeights):
			updatedWeights = np.dot(np.atleast_2d(self.layers[i]).T, np.atleast_2d(self.deltaError[((i*-1)-1)]))* self.learningRate
			self.weights[i] =  self.weights[i]+updatedWeights 

#-----------------------------------------------------------------------------------------------------------------------
	def __softmax(self,x):
		#Sofmax activation function
		return np.exp(x) / float(sum(np.exp(x)))		

#-----------------------------------------------------------------------------------------------------------------------
	def __sigmoid(self, x):
		#Sigmoid activation function                                        
		return 1 / (1 + np.exp(-x))

#-----------------------------------------------------------------------------------------------------------------------
	def __derivative(self, x):
		#Derivate of the sigmoid function
		return x * (1 - x)

#-----------------------------------------------------------------------------------------------------------------------
	def __crossEntropy(target):
		#Cross entropy calculation for the output layer
		predictions = np.clip(layers[-1], epsilon, 1. - epsilon)
		N = len(targets)
		crossEntropy = -np.sum(np.sum(targets*np.log(predictions+1e-9)))/N
		return crossEntropy

#-----------------------------------------------------------------------------------------------------------------------
	def fit(self, inputs, targets, showChart=True, testAccuracy = False, testInputs = None, testTargets= None):
		#This function is responsible for training the network accordingly to the number of chosen epochs

		print ("Training Network...")
		for i in range(0, self.numberOfEpochs):
			for j in range(0,len(inputs)):
				self.__feedForward(inputs[j])
				self.__backPropagation(targets[j])
			self.epochRMS.append(self.allSquaredErrors[-1])
			if (testAccuracy):
				self.epochAccuracy.append(self.accuracy(testInputs,testTargets))
		print ("Done!")
		if (showChart):
			plt.plot(self.epochRMS)
			plt.title("Network RMS x Epochs")
			plt.xlabel('Epochs')
			plt.ylabel('RMS')
			plt.show()
		if (testAccuracy):
			plt.plot(self.epochAccuracy)
			plt.title("Accuracy x Epochs")
			plt.xlabel('Epochs')
			plt.ylabel('Accuracy')
			plt.show()

#-----------------------------------------------------------------------------------------------------------------------
	def predict(self, input, classificationOutput=False, softmax=True): #classificationOutput == Winner takes all
		
		#This function outputs the response of a trained neural network by using a given input
		
		outputLayer = self.__feedForward(input)
		if (classificationOutput ==False):
			if (softmax==True):
				return self.__softmax(outputLayer)
			else:
				return outputLayer
		else: #this part changes the biggest probability to 1, and the other elements to 0
			maxValue = max(outputLayer)
			(outputLayer == maxValue)
			outputLayer[outputLayer < maxValue ] = 0
			outputLayer[outputLayer > 0 ] = 1
			return outputLayer

#-----------------------------------------------------------------------------------------------------------------------
	def accuracy(self, inputs, targets):
		#Compute the accuracy of the network

		correctClassifications = 0
		numberOfTargets = len(targets)
		for i in range(len(inputs)):
			outputLayer = self.predict(inputs[i],classificationOutput=True, softmax=False)
			if (np.array_equal(outputLayer,targets[i])):
				correctClassifications += 1
		return correctClassifications/numberOfTargets

#-----------------------------------------------------------------------------------------------------------------------

	
	

