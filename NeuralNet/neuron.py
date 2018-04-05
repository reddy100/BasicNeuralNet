import math
import numpy as np
from connection import Connection

class Neuron:
    eta = 0.001  #learning rate
    alpha = 0.01 #momentum

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)

    def addError(self, err):
        self.error = self.error + err

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x * 1.0))

    def dSigmoid(self, x):
        return x * (1.0 - x)

    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def setGradient(self, gradient):
        self.gradient = gradient

    def getGradient(self):
        return self.gradient

    def getOutput(self):
        return self.output

    def getError(self):
	return self.error

    def _sumOutput(self, dendrons):
        sumOutput = 0
	for dendron in dendrons:
            sumOutput = sumOutput + dendron.connectedNeuron.getOutput() * dendron.weight
	return sumOutput

    def feedForward(self):
        if len(self.dendrons) == 0:
            return
        sumOutput = self._sumOutput(self.dendrons)
        self.output = self.sigmoid(sumOutput)

    def _findGradient(self):
	return self.error * self.dSigmoid(self.output)

    def _updateddWeight(self, dendron):
	return Neuron.eta * (dendron.connectedNeuron.getOutput() * self.gradient) + self.alpha * dendron.getdWeight();

    def _updatedWeight(self, dendron):
	return dendron.getWeight() + dendron.getdWeight()

    def _updatedError(self, dendron):
	return dendron.getWeight() * self.gradient

    def backPropagate(self):
        self.gradient = self._findGradient();
        for dendron in self.dendrons:
            dendron.dWeight = self._updateddWeight(dendron)
            dendron.setWeight(self._updatedWeight(dendron));
            dendron.connectedNeuron.addError(self._updatedError);
        self.error = 0;

