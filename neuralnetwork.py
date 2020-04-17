################################################################################################################################
#
#       By:@YQDevelops
#Objective:To Create A Simple Neural Network That Can Recognise Handwritten Digits
#    Edits:29/3/2020-- Tried to do the cost function.
#           1/4/2020-- Added way to find out accuracy, attempted the cost function
#           3/4/2020-- Added the cost function, ran it but took to long to compute. Computer was overheating.
#           9/4/2020-- Changed to Hadamard Product instead of matmul
#
################################################################################################################################
import numpy as np

class ArtificialBrain:

    def __init__(self, layerSizes):
        weightShapes = [(a,b) for a,b in zip(layerSizes[1:], layerSizes[:-1])] #zip(from 2nd item until last),(from 1st item until 2nd last)
        self.weights = [np.random.standard_normal(s)/layerSizes[0]**0.5 for s in weightShapes] #random.standard_normal() is randoms in the standard random deviation
        self.biases = [np.zeros((s,1)) for s in layerSizes[1:]]

    def predict(self, input):
        for weights, biases in zip(self.weights, self.biases):

            input = self.activation(np.matmul(weights, input) + biases)

        return input

    def printAccuracy(self, images, labels):
        predictions = self.predict(images)
        numCorrect = sum([np.argmax(a) == np.argmax(b) for a, b in zip(predictions, labels)])

        print(f"{numCorrect}/{len(images)} accuracy: {(numCorrect/len(images)) * 100}%")
        print("HelloWorld!")

    def getCost(self,input,trueLabels):
        numberOfRuns = 0
        cost = 0
        for i in range(len(input)):

            cost += self.cost(input,trueLabels,numberOfRuns)
            numberOfRuns += 1
            print(f"cost:{cost}")
            print(f"times:{numberOfRuns}")
            if numberOfRuns >= len(input):
                break

        cost = cost/(numberOfRuns*2)
        return cost

    """Quadratic Cost Function"""
    def cost(self,input, trueLabels, runNumber):
        outputSqrt = trueLabels[runNumber] - (self.predict(input)[runNumber])
        return outputSqrt * outputSqrt
        #y(i) = (0,0,0,0,1,0,0,0,0,0)Transposed
        #x = 28 x 28 = 784 dimensional vector for image
        #C(weights, biases) = sum of (y(i) - prediction(x)) divided 2*number of pixels/ training inputs

    """Activation Function"""
    @staticmethod
    def activation(x):
        return 1/(1 + np.exp(-x))

  #  print(weightShapes) #Output == [(3,2),(5,3),(2,5)]
