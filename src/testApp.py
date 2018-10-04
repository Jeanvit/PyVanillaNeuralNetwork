'''This is the test code'''
''' Jean Vitor de Paulo'''
'''Requirements: Numpy, sklearn(for the built int Iris dataset and train/test split) and matplotlib'''

from vanillaNN import VanillaNeuralNetwork
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
	irisDataset = load_iris()
	input = irisDataset.data
	tempTargets = irisDataset.target
	targets = []
	for i in tempTargets:
		if (i==0):
			targets.append([1,0,0])
		elif(i==1):
			targets.append([0,1,0])
		elif(i==2):
			targets.append([0,0,1])

	input_train, input_test, targets_train, targets_test = train_test_split(input, targets, test_size=0.30, random_state=42)

	myNeuralNetwork = VanillaNeuralNetwork([4,4,3],2500)
	#myNeuralNetwork.fit(input_train,targets_train)
	myNeuralNetwork.fit(input_train,targets_train, testAccuracy= True ,testInputs = input_test, testTargets= targets_test)
	print("Predict: ",myNeuralNetwork.predict(input_test[0]))
	print ("Accuracy: ", myNeuralNetwork.accuracy(input_test,targets_test))