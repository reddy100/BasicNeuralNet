import numpy as np

class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0

    def setWeight(self, weight):
	self.weight = weight

    def setdWeight(self, dWeight):
	self.dWeight = dWeight

    def getWeight(self):
	return self.weight

    def getdWeight(self):
	return self.dWeight
