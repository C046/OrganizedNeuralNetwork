import numpy as np

neurons = np.array([1, 2, 3, 4])
from mpmath import mp
import matplotlib.pyplot as plt

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
        return 1/(1+np.power((float(self.e)),(-x)))
    
    def slope(self, input_var, step_size):
        return (self.Sigmoid((input_var+step_size))-self.Sigmoid(input_var))/step_size
            
    
        
    def plot_sigmoid_derivative(self, inputs, derivative_values):
        x_values = inputs  # Adjust the range accordingly
        #@derivative_values = self.Sigmoid_derivative(x_values)

        plt.plot(x_values, derivative_values, label='Sigmoid Derivative')
        plt.title('Derivative of Sigmoid Function')
        plt.xlabel('Input')
        plt.ylabel('Derivative Value')
        plt.legend()
        plt.grid(True)
        plt.show()
