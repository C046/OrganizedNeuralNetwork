import numpy as np

neurons = np.array([1, 2, 3, 4])

class Activations:
    def __init__(self, input_array):
        self.input_array = input_array
        self.input_size = self.input_array.size
        
        self.biases = self.Grw(size=self.input_size)
        self.weights = self.Grw(size=self.input_size)
        
    def Grw(self, size):
        """
        Generate random weights for the given input size.

        Parameters:
        - input_size (int): Number of input features.

        Returns:
        - weights (numpy.ndarray): Randomly generated weights.
        """
        # Generate random weights using a normal distribution
        return np.random.normal(size=(size,))

    def iter_neuron(self):
        try:    
            for element, bias, weights in zip(self.input_array, self.biases, self.weights):
                yield (element, bias, weights)
        
        except StopIteration:
            pass
        
        
# Example usage:
activations_instance = Activations(neurons)
for neuron in activations_instance.iter_neuron():
    print(neuron)