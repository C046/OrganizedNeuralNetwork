# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:55:02 2024

@author: hadaw
"""
from OrganizedNeuralNetwork.Acts import Activations
import numpy as np


class Normalization(Activations):
    def __init__(self, input_array):
        super().__init__(input_array)
        #print(input_array)
        self.neurons = input_array
    
    def binary_cross_entropy(self, true_labels):
        epsilon = 1e-15  # Small constant to avoid log(0)
        sigmoid_output = np.clip(self.neurons, epsilon, 1 - epsilon)  # Clip predicted values to avoid log(0) or log(1)
        true_labels = np.clip(true_labels, -epsilon, 1+epsilon)
        
        sigmoid_output = np.array(sigmoid_output, dtype=np.float64)
        
        return -np.mean((true_labels*np.log(sigmoid_output))+((1-true_labels)*np.log((1-sigmoid_output))))
        #return -np.mean(true_labels * np.log(sigmoid_output) + (1 - true_labels) * np.log(1 - sigmoid_output))


    def mean_squared_error(self, predicted, actual):
        return 0.5 * np.mean((predicted - actual)**2)
    
    def mse_derivative(self, predicted, actual):
        return predicted - actual