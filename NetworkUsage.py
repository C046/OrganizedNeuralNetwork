# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:21:41 2024

@author: hadaw
"""

import numpy as np
import pandas as pd
from OrganizedNeuralNetwork.Input import InputLayer
from OrganizedNeuralNetwork.Acts import Activations
from OrganizedNeuralNetwork.Norm import Normalization

thresh=0.5
class NeuralNetwork:
    def __init__(self, data_path):
        # Load dataset and initialize attributes
        self.data = pd.read_csv(data_path)
        self.tumor_descriptions = self.data.drop('diagnosis', axis=1)
        self.tumor_descriptions = self.tumor_descriptions  # Double assignment, might be unintentional
        self.y_true = (self.data['diagnosis'].values == 'M').astype(int)

    def train(self, num_hidden_layers=1, learning_rate=0.01, epochs=100):
        # Loop over training epochs
        for epoch in range(epochs):
            gradients = []  # To store gradients for each layer
            # Loop over columns (features)
            for column_name, neurons in self.tumor_descriptions.items():
                input_layer = InputLayer(neurons, batch_size=569)

                # Loop over hidden layers
                for i in range(num_hidden_layers):
                    output = []
                    derivative = []
                    norm_der = []
                    # Loop over batches
                    for batched_inputs in input_layer.batch_inputs():
                        activators = Activations(batched_inputs)
                        # Loop over neurons in the layer
                        for neuron in activators.Iter_neuron():
                            neuron, bias, weights = neuron
                            Neuron_weighted_sum = np.dot(weights, neuron) + bias
                            sigmoid_output = activators.Sigmoid(Neuron_weighted_sum)

                            predicted_labels = np.int32(sigmoid_output >= thresh)
                            output.append(sigmoid_output)

                            derivatives = activators.Sigmoid_Derivative(sigmoid_output)
                            derivative.append(derivatives)

                        accuracy = sum(predicted_labels == self.y_true) / len(self.y_true)

                        neurons = np.array(output)

                        # Calculate binary cross-entropy loss and its derivative
                        norm = Normalization(neurons).binary_cross_entropy(self.y_true)
                        norm_derivative = neurons - self.y_true
                        norm_der.append(norm_derivative)

                    # Store normalized derivatives and gradients for this layer
                    norm_der = np.array(norm_der)
                    gradient = norm_der * derivative * neurons
                    gradients.append(gradient)

            # Update weights after processing all batches and layers
            activators.update_weights(gradients, learning_rate)

            # Optionally, print or store other information after each epoch
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {np.mean(norm)}")

if __name__ == "__main__":
    neural_network = NeuralNetwork("OrganizedNeuralNetwork/breast-cancer.csv")
    neural_network.train(num_hidden_layers=1, learning_rate=0.01, epochs=100)




            
                    
                    

    