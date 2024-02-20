# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:35:28 2024

@author: hadaw
"""
# Import the numpy module
import numpy as np
# Custom imports for networks layers
from OrganizedNeuralNetwork.Input import InputLayer
from OrganizedNeuralNetwork.Hidden import HiddenLayer
from OrganizedNeuralNetwork.Output import OutputLayer

# Custom imports for networks activators and normalization
from OrganizedNeuralNetwork.Acts import Activations
from OrganizedNeuralNetwork.Norm import Normalization


# Input array 
neurons = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Input layer pipeline
Input_Layer = InputLayer(neurons, batch_size=2)
batch_iterator = Input_Layer.batch_input_method()
# Example of using the generator for batches
try:
    while True:
        batched_inputs = next(batch_iterator)
        print("Batch Elements:", batched_inputs)
        print("---")
except StopIteration:
    pass
except ValueError as e:
    print(e)
    
# # Hidden layer pipeline
Hidden_Layer = HiddenLayer()
# # Output layer pipeline
Output_Layer = OutputLayer()

# Activator functions
Activators = Activations(batched_inputs)
for neuron in Activators.Iter_neuron():
    print(neuron)
# # Normal functions
# Normals = Normalization()

