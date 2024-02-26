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
        
        self.neuron_data = {
            "Neuron": [],
            "Weights": [],
            "Bias": [],
            "Weighted_Sum":[],
        }
        
        self.activation_data = {
            "Sigmoid_Output":[],
            "Sigmoid_Derivative":[],
            "Predicted_Labels": [],
        }
        
        self.normal_data = {
            "Accuracy":[],
            "Normal":[],
            "Normal_Derivative":[]
        }
        


        self.epsilon = 1e-15

    # def backpropagation(self, output_gradient, input_layer, num_layers=1, learning_rate=0.01, epochs=100):
    #     print(type(output_gradient))
    #     print(type(self.weights))
    #     print(type(self.biases))
    #     print(type(learning_rate))

    #     if not isinstance(input_layer.input_array, np.ndarray):
    #         inputs = np.asarray(input_layer.input_array)
    #     else:
    #         inputs = input_layer.input_array

    #     loss_history = []

    #     for epoch in range(epochs):
    #         gradients = []

    #         for hidden_layer in reversed(range(num_layers)):
                
    #             weights_gradient = float(np.dot(inputs.T, output_gradient))
    #             bias_gradient = float(np.sum(output_gradient, axis=0))
    #             self.weights[hidden_layer] = np.array(self.weights[hidden_layer], dtype=np.float64)
    #             self.biases[hidden_layer] = np.array(self.biases[hidden_layer], dtype=np.float64)
                
    #             self.weights[hidden_layer] -= learning_rate * weights_gradient
    #             self.biases[hidden_layer] -= learning_rate * bias_gradient

    #         # Calculate gradients for weights and biases in the input layer
    #         weights_gradient_input = np.array(np.dot(inputs.T, output_gradient), dtype=np.float64)
    #         bias_gradient_input = np.asarray(np.sum(output_gradient, axis=0, dtype=np.float64))

    #         # Update weights and biases in the input layer
            
    #         self.biases = np.asarray(self.biases, dtype=np.float64)
    #         self.weights -= learning_rate * weights_gradient_input
    #         self.biases -= learning_rate * bias_gradient_input

    #         # Calculate and store the loss for monitoring
    #         loss = Normalization(inputs).binary_cross_entropy(self.labels, sigmoid_output)
    #         loss_history.append(loss)
    #     print(f"loss:{loss}")
        

    def train(self, num_hidden_layers=1, learning_rate=0.01, epochs=100, batch_size=569):
   

        # Loop over training epochs
        for epoch in range(epochs):
            # Loop over features (columns)
            for feature_name, neurons in self.features.items():
                input_layer = InputLayer(neurons, batch_size=batch_size)

                # Loop over hidden layers
                for _ in range(num_hidden_layers):
         

                    # Loop over batches
                    for batched_inputs in input_layer.batch_inputs():
                        activators = Activations(batched_inputs)

                        # Loop over neurons in the layer and unzip the weights
                        for neuron, bias, weights in activators.Iter_neuron():
                            neuron=float(neuron)
                            # Turn neuron, bias, weights intp mpf objects for floating point percision.
                            neuron, bias, weights = mpl.mpi(neuron), mpl.mpf(bias), mpl.mpf(weights)
        
                            # Check whether the lists are already populated
                            # else append the data to the data structure
                            if len(self.neuron_data["Neuron"]) <= 0:
                                self.neuron_data["Neuron"].append(neuron)
                            else:
                                pass
                            
                            if len(self.neuron_data["Weights"]) <= 0:
                                self.neuron_data["Weights"].append(weights)
                            else:
                                pass
                            
                            if len(self.neuron_data["Bias"]) <= 0:
                                self.neuron_data["Bias"].append(bias)
                            else:
                                pass
                            
                            neuron_weighted_sum = np.dot(weights, neuron)+ bias
                            if len(self.neuron_data["Weighted_Sum"]) <= 0:
                                self.neuron_data["Weighted_Sum"].append(neuron_weighted_sum)
                            else:
                                pass
                            sigmoid_out = activators.Sigmoid(neuron_weighted_sum)
                            
                            self.activation_data["Sigmoid_Output"].append(sigmoid_out)
                            self.activation_data["Sigmoid_Derivative"].append(activators.Sigmoid_Derivative(sigmoid_out))
                            self.activation_data["Predicted_Labels"].append(np.int64(sigmoid_out >= thresh))
                            
          
                            
          
                        
                        
                        # Calculate the accuracy of the model
                        # self.activation_data["Predicted_Labels"] = np.array(self.activation_data["Predicted_Labels"])
                        # # print(self.activation_data["Predicted_Labels"])
                        # self.normal_data["Accuracy"].append(sum(self.activation_data["Predicted_Labels"]) == self.labels)/len(self.labels)
                        
                        # self.normal_data["Normal"].append(Normalization(neurons).binary_cross_entropy(self.labels, sigmoid_out))
                        # self.normal_data["Normal_Derivative"] = (neurons - self.labels) / (neurons * (1 - neurons) + self.epsilon)
                        
                        
                        # # Clear the predicted_labels list
                        # self.activation_data["Predicted_Labels"].clear()
                        # Print the accuracy
                       
   
                        # Set the neurons var to the output to run through the model again.
                        neurons = np.array(self.activation_data["Sigmoid_Output"])
                        
                        
                        
                        plots.plot_loss_curve(self.normal_data["Accuracy"])
                   
                        
                        # self.normal_data["Accuracy"].append(sum(self.activation_data["Predicted_Labels"] == self.labels) / len(self.labels))
                        # self.normal_data["Normal"].append(Normalization(neurons).binary_cross_entropy(self.labels))
                        # self.normal_data["Normal_Derivative"].append((neurons - self.labels) / (neurons * (1 - neurons) + self.epsilon))

                        
        
                
            # print(f"Epoch {epoch + 1}/{epochs}, Loss: {np.mean(normal_data['Accuracy'])}")
            # normal_data["Normral_Derivative"] = np.array(normal_data["Normal_Derivative"], dtype=np.float64).flatten()
            # sigmoid_data["Sigmoid_Derivative"] = np.array((sigmoid_data["Sigmoid_Derivative"]),dtype=np.float64).flatten()
            
            # output_gradient = np.dot(sigmoid_data["Sigmoid_Derivative"], normal_data["Normal_Derivative"])*neurons
            
            
            #self.backpropagation(output_gradient, input_layer, epochs=epochs)
            
        
            
            
        

        

        
        return self.weights, self.predicted_labels, np.array(output, dtype=np.int64), 


if __name__ == "__main__":
    neural_network = NeuralNetwork("OrganizedNeuralNetwork/breast-cancer.csv")
    output = neural_network.train(num_hidden_layers=1, learning_rate=10, epochs=10)