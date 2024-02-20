# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:21:41 2024

@author: hadaw
"""

import numpy as np
import pandas as pd
from mpmath import mp
from OrganizedNeuralNetwork.Input import InputLayer
from OrganizedNeuralNetwork.Acts import Activations
from OrganizedNeuralNetwork.Norm import Normalization


class NeuralNetwork:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.tumor_descriptions = self.data.drop('diagnosis', axis=1)
        self.tumor_descriptions = self.tumor_descriptions.applymap(mp.mpf)
        self.y_true = (self.data['diagnosis'].values == 'M').astype(int)

    def train(self, num_hidden_layers=10):
        for column_name, neurons in self.tumor_descriptions.items():
            input_layer = InputLayer(neurons, batch_size=569)

            for i in range(num_hidden_layers):    
                output = []
                for batched_inputs in input_layer.batch_inputs():
                    activators = Activations(batched_inputs)
                    for neuron in activators.Iter_neuron():
                        neuron, bias, weights = neuron
                        Neuron_weighted_sum = np.dot(weights, neuron) + bias
                        output.append(activators.Sigmoid(Neuron_weighted_sum))

                    neurons = np.array(output)
                    Y = neurons
                    
                    # but since i cant divide by zero its unable to go forward, hmmmm
                    partial_derivative = (Y*(1-Y))/(Y*(1-self.y_true))
                 
                    
                    #print(gradient)
                    
                    norm = Normalization(neurons).binary_cross_entropy(self.y_true)
                    print(f"\nNormal: {norm}")
                    
                    

if __name__ == "__main__":
    neural_network = NeuralNetwork("OrganizedNeuralNetwork/breast-cancer.csv")
    neural_network.train(10)
    