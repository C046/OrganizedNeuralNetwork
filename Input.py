# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:05:40 2024

@author: hadaw
"""
import numpy as np
from OrganizedNeuralNetwork.Acts import Activations

class InputLayer(Activations):
    def __init__(self, input_array, batch_size=5):
        self.input_array = np.array(input_array)
        self.input_size = input_array.size
        
        self.batch_size = batch_size
        self.batch_input_method = self.batch_inputs  # Change the attribute name
    
    def batch_inputs(self):
        # Check if the batch size is not evenly divisible
        if self.input_size % self.batch_size != 0:
            raise ValueError("Batch size must be evenly divisible by the input size.")
        
        # Iterate over batches and yield them
        for i in range(0, self.input_size, self.batch_size):
            batch_elements = self.input_array[i:i + self.batch_size]
            yield batch_elements

