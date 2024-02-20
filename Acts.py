import numpy as np

neurons = np.array([1, 2, 3, 4])
from mpmath import mp

class Activations:
    def __init__(self, input_array, e=False):
        self.input_array = input_array
        self.input_size = self.input_array.size
        
        self.biases = self.Grwb(size=self.input_size)
        self.weights = self.Grwb(size=self.input_size)
        
        self.e = 2.71828
        
    
    def Grwb(self, size):
        """
        <Generate Random Weights or biases> for the given input size.

        Parameters:
        - input_size (int): Number of input features.

        Returns:
        - weights (numpy.ndarray): Randomly generated weights.
        """
        # Generate random weights using a normal distribution
        return np.random.normal(size=(size,))

    def Iter_neuron(self):
        try:    
            for element, bias, weights in zip(self.input_array, self.biases, self.weights):
                yield (element, bias, weights)
        
        except StopIteration:
            pass
    
    def Sigmoid(self, x):
        return 1/(1+(float(self.e)**(-x)))
        
            
        #return 1/(1+(float(self.e)**(-x)))
        
        #return 1 / 1 + (float(self.e) ** -x)
    
        
