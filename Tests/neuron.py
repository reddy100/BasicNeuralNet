import sys, os, unittest
sys.path.append(os.path.abspath('/home/abishek/Code/BasicNeuralNet'))
from NeuralNet.neuron import Neuron
from NeuralNet.connection import Connection

class NeuronTest(unittest.TestCase):

	def setUp(self):
		connectedNeuron1 = Neuron([])
		connectedNeuron2 = Neuron([])
		connectedNeuron1.setError(1)
		connectedNeuron2.setError(1)
		connectedNeuron1.setOutput(1)
		connectedNeuron2.setOutput(1)
		connectedNeuron1.setGradient(1)
		connectedNeuron2.setGradient(1)
		#connection1 = Connection(connectedNeuron1)
		#connection2 = Connection(connectedNeuron2)
		#connection1.setWeight(1)
		#connection2.setWeight(1)
		self.neuron1 = Neuron([connectedNeuron1, connectedNeuron2])
		self.neuron1.setGradient(1)
		self.neuron1.setError(1)
		self.neuron1.setOutput(2)
		for con in self.neuron1.dendrons:
			con.setWeight(1)
			con.setdWeight(2)
		self.neuron2 = Neuron([])
		

    	def test_addError(self):
		self.neuron1.addError(1)
		self.assertEqual(self.neuron1.error,2.0)

	def test_sigmoid(self):
		self.assertEqual(round(self.neuron1.sigmoid(1),3),0.731)

	def test_dSignmoid(self):
		self.assertEqual(self.neuron1.dSigmoid(2),-2)

	def test_setError(self):
		self.neuron1.setError(2)
		self.assertEqual(self.neuron1.error,2)

	def test_setOutput(self):
		self.neuron1.setOutput(2)
		self.assertEqual(self.neuron1.output,2)

	def test_sumOutput(self):
		dendrons = self.neuron1.dendrons
		self.assertEqual(self.neuron1._sumOutput(dendrons), 2)

	def test_feedForward1(self):
		self.assertIsNone(self.neuron2.feedForward())

	def test_feedForward2(self):
		self.assertEqual(self.neuron1.getOutput(),2)
		self.neuron1.feedForward()
		self.assertEqual(self.neuron1.getOutput(),self.neuron1.sigmoid(2))

	def test_findGradient(self):
		self.assertEqual(self.neuron1._findGradient(), -2)

	def test_updateddWeight(self):
		dendron = self.neuron1.dendrons[0]
		self.assertEqual(self.neuron1._updateddWeight(dendron), 0.021)

	def test_updatedWeight(self):
		dendron = self.neuron1.dendrons[0]
		self.assertEqual(self.neuron1._updatedWeight(dendron), 3)

	def test_updatedError(self):
		dendron = self.neuron1.dendrons[0]
		self.assertEqual(self.neuron1._updatedError(dendron), 1)


def main():
    unittest.main()

if __name__ == '__main__':
    main()



