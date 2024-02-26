import numpy as np
import pandas as pd
from OrganizedNeuralNetwork.Input import InputLayer
from OrganizedNeuralNetwork.Acts import Activations
from OrganizedNeuralNetwork.Norm import Normalization
from OrganizedNeuralNetwork.Plotts import Plotter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mpmath as mpl

thresh = 0.5
plots = Plotter()

class NeuralNetwork:
    def __init__(self, data_path):
        # Load dataset and initialize attributes
        self.data = pd.read_csv(data_path)
        self.features = self.data.drop('diagnosis', axis=1)
        self.labels = (self.data['diagnosis'].values == 'M').astype(int)
        self.predicted_labels = []
        self.weights = []
        self.biases = []
        self.epsilon = 1e-15

    def backpropagation(self, output_gradient, input_layer, num_layers=1, learning_rate=0.01, epochs=100):
        print(type(output_gradient))
        print(type(self.weights))
        print(type(self.biases))
        print(type(learning_rate))

        if not isinstance(input_layer.input_array, np.ndarray):
            inputs = np.asarray(input_layer.input_array)
        else:
            inputs = input_layer.input_array

        loss_history = []

        for epoch in range(epochs):
            gradients = []

            for hidden_layer in reversed(range(num_layers)):
                
                weights_gradient = np.dot(inputs.T, output_gradient)
                bias_gradient = np.sum(output_gradient, axis=0)
                self.weights[hidden_layer] = np.array(self.weights[hidden_layer])
                self.biases[hidden_layer] = np.array(self.biases[hidden_layer])

                self.weights[hidden_layer] -= learning_rate * weights_gradient
                self.biases[hidden_layer] -= learning_rate * bias_gradient

            # Calculate gradients for weights and biases in the input layer
            weights_gradient_input = np.array(np.dot(inputs.T, output_gradient), dtype=np.float64)
            bias_gradient_input = np.asarray(np.sum(output_gradient, axis=0, dtype=np.float64))

            # Update weights and biases in the input layer
            self.biases = np.asarray(self.biases, dtype=np.float64)
            self.weights -= learning_rate * weights_gradient_input
            self.biases -= learning_rate * bias_gradient_input

            # Calculate and store the loss for monitoring
            loss = Normalization(inputs).binary_cross_entropy(self.labels)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")
            loss_history.append(loss)

            return self.weights,self.biases, loss_history

    def train(self, num_hidden_layers=1, learning_rate=0.01, epochs=100, batch_size=569):
        loss = []

        # Loop over training epochs
        for epoch in range(epochs):
            norm_der = []
            sigmoid_der = []

            # Loop over features (columns)
            for feature_name, neurons in self.features.items():
                input_layer = InputLayer(neurons, batch_size=batch_size)

                # Loop over hidden layers
                for _ in range(num_hidden_layers):
                    output = []

                    # Loop over batches
                    for batched_inputs in input_layer.batch_inputs():
                        activators = Activations(batched_inputs)

                        # Loop over neurons in the layer and unzip the weights
                        for neuron, bias, weights in activators.Iter_neuron():
                            neuron, bias, weights = mpl.mpf(neuron), mpl.mpf(bias), mpl.mpf(weights)

                            # Calculate the weighted sum on the neuron
                            Neuron_weighted_sum = np.dot(weights, neuron) + bias

                            # Calculate the sigmoid output on the weighted sum
                            sigmoid_output = activators.Sigmoid(Neuron_weighted_sum)

                            # Sigmoid Derivative
                            sigmoid_derivative = activators.Sigmoid_Derivative(sigmoid_output)

                            # Append the sigmoid output to a list/array.
                            output.append(sigmoid_output)

                            # Append the sigmoid derivative to a list/array
                            sigmoid_der.append(sigmoid_derivative)

                            # The sigmoid output is the prediction the model is making
                            predicted_labels = np.int64(sigmoid_output >= thresh)

                            # Append predictions to a list/array
                            self.predicted_labels.append(predicted_labels)

                            # Append weights and biases
                            self.weights = list(self.weights)
                            self.biases = list(self.biases)
                            self.weights.append(weights)
                            self.biases.append(np.array(bias))

                        # Calculate the accuracy of the model
                        accuracy = sum(predicted_labels == self.labels) / len(self.labels)

                        # Set the neurons var to the output to run it through the model again.
                        neurons = np.array(output)

                        # Calculate binary cross-entropy loss and its derivative
                        norm = Normalization(neurons).binary_cross_entropy(self.labels)
                        loss.append(norm)

                        norm_derivative = (neurons - self.labels) / (neurons * (1 - neurons) + self.epsilon)
                        norm_der.append(norm_derivative)
                
                plots.plot_loss_curve(loss)
                
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {np.mean(norm)}")
            norm_der = np.array(norm_der).flatten()
            sigmoid_der = np.array(sigmoid_der)
            output_gradient = np.dot(sigmoid_der, norm_der) * np.array(output, dtype=np.int64)
            self.weights = np.array(self.weights, dtype=float)
            self.weights,self.biases, loss_history = self.backpropagation(output_gradient, input_layer, epochs=epochs)
            plots.plot_loss_curve(loss_history)
            
        

        

        
        return self.weights, self.predicted_labels, np.array(output, dtype=np.int64), sigmoid_der, norm_der


if __name__ == "__main__":
    neural_network = NeuralNetwork("OrganizedNeuralNetwork/breast-cancer.csv")
    output = neural_network.train(num_hidden_layers=1, learning_rate=0.00000000000001, epochs=10)