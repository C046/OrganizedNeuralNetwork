import numpy as np

import plotly.express as px
neurons = np.array([1, 2, 3, 4])
import mpmath as mp
import plotly.io as pio


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
    
    def Sigmoid(self, x, threshold=np.random.uniform(0.40,0.50), epsilon=1e-15):
        # Sigmoid function
        sigmoid_result = 1 / (1 + np.exp(-x))

        # Apply 1,0 conditional with threshold
        sigmoid_result = np.where(sigmoid_result >= threshold, 1.0, 0.0)

        return (sigmoid_result, threshold)
    
    
    
    def Sigmoid_Derivative(self, sigmoid_output):
        return sigmoid_output * (1 - sigmoid_output)
    
            
    def update_weights(self, gradients, learning_rate):
        # Perform the weight updates here based on your optimization algorithm
        # Example: Simple gradient descent update
        mean_gradient = np.mean(gradients, axis=0)
        self.weights -= learning_rate * mean_gradient.reshape(self.weights.shape)


    def plot_sigmoid_derivative(self, inputs, derivative_values):
        # Adjust the range accordingly
        x_values = inputs
    
        # Create a scatter plot
        fig = px.line(x=x_values, y=derivative_values, labels={'x': 'Input', 'y': 'Derivative Value'},
                         title='Derivative of Sigmoid Function', template='plotly')
                        
        # Show the plot
        fig.show()

        # Save the HTML representation of the plot to a file
        with open("plot.html", "w", encoding="utf-8") as file:
            plot_html = str(pio.to_html(fig, full_html=False))
            file.write(plot_html)
        