# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:35:28 2024

@author: hadaw
"""
import numpy as np
from OrganizedNeuralNetwork.Input import InputLayer
from OrganizedNeuralNetwork.Hidden import HiddenLayer
from OrganizedNeuralNetwork.Output import OutputLayer
from OrganizedNeuralNetwork.Acts import Activations

# Input array
neurons = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Input layer pipeline
input_layer = InputLayer(neurons, batch_size=2)

# Hidden layer pipeline
hidden_layer = HiddenLayer()

# Output layer pipeline
output_layer = OutputLayer()

# Example of using a single loop for both batched inputs and activations
for batched_inputs in input_layer.batch_inputs():
    print("Batch Elements:", batched_inputs)
    
    # Activator functions
    activators = Activations(batched_inputs)
    for neuron in activators.Iter_neuron():
        neuron,bias,weights = neuron
        print("Neuron: ", neuron,"Bias: ", bias,"Weights: ", weights)

    
# # Hidden layer pipeline
Hidden_Layer = HiddenLayer()
# # Output layer pipeline
Output_Layer = OutputLayer()


# # Normal functions
# Normals = Normalization()

