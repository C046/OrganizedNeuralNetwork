# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:05:40 2024

@author: hadaw
"""
import numpy as np
from OrganizedNeuralNetwork.Acts import Activations

class InputLayer(Activations):
    def __init__(self, input_array, batch_size=5):
        super().__init__(input_array)
        self.batch_size = batch_size
        self.batch_inputs = self.batch_inputs
    
    def batch_inputs(self):
        # Check if the batch size is not evenly divisible
        if self.input_size % self.batch_size != 0:
            raise ValueError("Batch size must be evenly divisible by the input size.")
        
        # Iterate over batches and yield them
        for i in range(0, self.input_size, self.batch_size):
            batch_elements = self.input_array[i:i + self.batch_size]
            yield batch_elements

# Example usage:
neurons = np.array([1, 2, 3, 4, 5, 6, 7, 8])
input_layer_instance = InputLayer(neurons, batch_size=2)

# Use the batch_inputs method to get batches
batch_iterator = input_layer_instance.batch_inputs()

# Example of using the generator for batches
try:
    while True:
        batch_elements = next(batch_iterator)
        print("Batch Elements:", batch_elements)
        print("---")
except StopIteration:
    pass
except ValueError as e:
    print(e)