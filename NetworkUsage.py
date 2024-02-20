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
neurons = np.array([i for i in range(0,100)])

# Input layer pipeline
input_layer = InputLayer(neurons, batch_size=2)

# Hidden layer pipeline
hidden_layer = HiddenLayer()

# Output layer pipeline
output_layer = OutputLayer()



num_hidden_layers = 10
for i in range(num_hidden_layers):    
    output = []
    # Example of using a single loop for both batched inputs and activations
    for batched_inputs in input_layer.batch_inputs():
        print("\nBatch Elements:", batched_inputs)
    
        # Activator functions
        activators = Activations(batched_inputs)
    
        for neuron in activators.Iter_neuron():
            
            neuron,bias,weights = neuron
            Neuron_weighted_sum = np.dot(weights, neuron)+ bias
            print("Neuron:", neuron,"\nBias:", bias,"Weights:", weights, f"\nNeuron Weighted Sum: {Neuron_weighted_sum}")
        
        
            output.append(activators.Sigmoid(Neuron_weighted_sum))
        
        neurons = output
        print(f"\n\nSigmoid: {output}") 
    
# # Hidden layer pipeline
Hidden_Layer = HiddenLayer()
# # Outpudt layer pipeline
Output_Layer = OutputLayer()


# # Normal functions
# Normals = Normalization()

