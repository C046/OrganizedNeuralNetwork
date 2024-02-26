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
    def __init__(self, data_path, batch_size):
        
        # Load dataset and initialize attributes
        self.data = pd.read_csv(data_path)
        self.features = self.data.drop('diagnosis', axis=1)
        self.labels = (self.data['diagnosis'].values == 'M').astype(int)
        self.batch_size = batch_size
        self.neurons,self.weights,self.biases,self.sums,self.sig_out,self.sig_der,self.pred_labels,self.accu,self.norm,self.norm_der=([],[],[],[],[],[],[],[],[],[])
        
        self.neuron_data = {
            "Neuron": self.neurons,
            "Weights": self.weights,
            "Bias": self.biases,
            "Neuron_Weighted_Sum":self.sums,
        }
        
        self.activation_data = {
            "Sigmoid_Output":self.sig_out,
            "Sigmoid_Derivative":self.sig_der,
            "Predicted_Labels": self.pred_labels,
        }
        
        self.normal_data = {
            "Accuracy":self.accu,
            "Normal":self.norm,
            "Normal_Derivative":self.norm_der
        }
        


        self.epsilon = 1e-15
    def propagation(self):
        pass
    
    def train(self, hidden_layers=1, epochs=1, learning_rate=0.001):
        # Iterate through epochs
        for epoch in range(epochs):
            # Configure the batching process
            for feature_name, neurons in self.features.items():
                input_layer = InputLayer(neurons, batch_size=self.batch_size)
                
                # Iterate over hidden layers
                for hidden_layers in range(hidden_layers):
                    # Process inputs into batches
                    for batched_inputs in input_layer.batch_inputs():
                        # Init iterator
                        activators = Activations(batched_inputs)
                        
                        # Iterate neuron, bias, and weights
                        for neuron, bias, weights in activators.Iter_neuron():
                            neuron, bias, weights = neuron, bias, weights
                            # Calculate the weighted sum
                            Neuron_Weighted_Sum = np.dot(neuron, weights) + bias
                            
  
                            self.neurons.append(neuron)
                            self.weights.append(weights)
                            self.biases.append(bias)
                            self.sums.append(Neuron_Weighted_Sum)
               
                            
                            sigmoid_out, threshold = activators.Sigmoid(Neuron_Weighted_Sum,threshold=np.random.uniform(0.40,0.50))
                            self.sig_out.append(sigmoid_out)
                            sigmoid_der = activators.Sigmoid_Derivative(sigmoid_out)
                            self.sig_der.append(sigmoid_der)
                           
                            print(f"Sigmoid output: {sigmoid_out}", flush=True)
                            
                        
                            # Append the prediction set
                            self.pred_labels.append(sigmoid_out)
                
                            
            self.norm.append(Normalization(np.array(self.neurons)).binary_cross_entropy(self.labels, self.sig_out))
            self.norm_der.append((neurons - self.labels) / (neurons * (1 - neurons) + self.epsilon))
            self.neurons = self.sig_out
            self.pred_labels.clear()
            
        
                                 
    #                     # Calculate the accuracy of the model
    #                     # self.activation_data["Predicted_Labels"] = np.array(self.activation_data["Predicted_Labels"])
    #                     # # print(self.activation_data["Predicted_Labels"])
    #                     # self.normal_data["Accuracy"].append(sum(self.activation_data["Predicted_Labels"]) == self.labels)/len(self.labels)
                        
    #                     # self.normal_data["Normal"].append(Normalization(neurons).binary_cross_entropy(self.labels, sigmoid_out))
    #                     # self.normal_data["Normal_Derivative"] = (neurons - self.labels) / (neurons * (1 - neurons) + self.epsilon)
                        
                        
    #                     # # Clear the predicted_labels list
    #                     # self.activation_data["Predicted_Labels"].clear()
    #                     # Print the accuracy
                       
   
    #                     # Set the neurons var to the output to run through the model again.
    #                     neurons = np.array(self.activation_data["Sigmoid_Output"])
                                           
                                
                            
        return (self.neuron_data, self.activation_data, self.normal_data)
            
            
            

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
        

    # def train(self, num_hidden_layers=1, learning_rate=0.01, epochs=100, batch_size=569):
   

    #     # Loop over training epochs
    #     for epoch in range(epochs):
    #         # Loop over features (columns)
    #         for feature_name, neurons in self.features.items():
    #             input_layer = InputLayer(neurons, batch_size=batch_size)

    #             # Loop over hidden layers
    #             for _ in range(num_hidden_layers):
         

    #                 # Loop over batches
    #                 for batched_inputs in input_layer.batch_inputs():
    #                     activators = Activations(batched_inputs)

    #                     # Loop over neurons in the layer and unzip the weights
    #                     for neuron, bias, weights in activators.Iter_neuron():
    #                         neuron=float(neuron)
    #                         # Turn neuron, bias, weights intp mpf objects for floating point percision.
    #                         neuron, bias, weights = mpl.mpi(neuron), mpl.mpf(bias), mpl.mpf(weights)
        
    #                         # Check whether the lists are already populated
    #                         # else append the data to the data structure
    #                         if len(self.neuron_data["Neuron"]) <= 0:
    #                             self.neuron_data["Neuron"].append(neuron)
    #                         else:
    #                             pass
                            
    #                         if len(self.neuron_data["Weights"]) <= 0:
    #                             self.neuron_data["Weights"].append(weights)
    #                         else:
    #                             pass
                            
    #                         if len(self.neuron_data["Bias"]) <= 0:
    #                             self.neuron_data["Bias"].append(bias)
    #                         else:
    #                             pass
                            
    #                         neuron_weighted_sum = np.dot(weights, neuron)+ bias
    #                         if len(self.neuron_data["Weighted_Sum"]) <= 0:
    #                             self.neuron_data["Weighted_Sum"].append(neuron_weighted_sum)
    #                         else:
    #                             pass
    #                         sigmoid_out = activators.Sigmoid(neuron_weighted_sum)
                            
    #                         self.activation_data["Sigmoid_Output"].append(sigmoid_out)
    #                         self.activation_data["Sigmoid_Derivative"].append(activators.Sigmoid_Derivative(sigmoid_out))
    #                         self.activation_data["Predicted_Labels"].append(np.int64(sigmoid_out >= thresh))
                            
          
                            
          
                        
                        
    #                     # Calculate the accuracy of the model
    #                     # self.activation_data["Predicted_Labels"] = np.array(self.activation_data["Predicted_Labels"])
    #                     # # print(self.activation_data["Predicted_Labels"])
    #                     # self.normal_data["Accuracy"].append(sum(self.activation_data["Predicted_Labels"]) == self.labels)/len(self.labels)
                        
    #                     # self.normal_data["Normal"].append(Normalization(neurons).binary_cross_entropy(self.labels, sigmoid_out))
    #                     # self.normal_data["Normal_Derivative"] = (neurons - self.labels) / (neurons * (1 - neurons) + self.epsilon)
                        
                        
    #                     # # Clear the predicted_labels list
    #                     # self.activation_data["Predicted_Labels"].clear()
    #                     # Print the accuracy
                       
   
    #                     # Set the neurons var to the output to run through the model again.
    #                     neurons = np.array(self.activation_data["Sigmoid_Output"])
                        
                        
                        
    #                     plots.plot_loss_curve(self.normal_data["Accuracy"])
                   
                        
    #                     # self.normal_data["Accuracy"].append(sum(self.activation_data["Predicted_Labels"] == self.labels) / len(self.labels))
    #                     # self.normal_data["Normal"].append(Normalization(neurons).binary_cross_entropy(self.labels))
    #                     # self.normal_data["Normal_Derivative"].append((neurons - self.labels) / (neurons * (1 - neurons) + self.epsilon))

                        
        
                
    #         # print(f"Epoch {epoch + 1}/{epochs}, Loss: {np.mean(normal_data['Accuracy'])}")
    #         # normal_data["Normral_Derivative"] = np.array(normal_data["Normal_Derivative"], dtype=np.float64).flatten()
    #         # sigmoid_data["Sigmoid_Derivative"] = np.array((sigmoid_data["Sigmoid_Derivative"]),dtype=np.float64).flatten()
            
    #         # output_gradient = np.dot(sigmoid_data["Sigmoid_Derivative"], normal_data["Normal_Derivative"])*neurons
            
            
    #         #self.backpropagation(output_gradient, input_layer, epochs=epochs)
            
        
            
            
        

        

        
        # return self.weights, self.predicted_labels, np.array(output, dtype=np.int64), 


if __name__ == "__main__":
    neural_network = NeuralNetwork("OrganizedNeuralNetwork/breast-cancer.csv", 569)
    output = neural_network.train()