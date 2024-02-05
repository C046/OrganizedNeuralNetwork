# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:05:40 2024

@author: hadaw
"""
import numpy as np


weights = np.matrix([
    [3, 1, 5, -4],
    [-1.5, -3, 7.1, 5.2],
    [2, 1, -6, 2.9]
])
inputs = np.array([0.5, 2.8, 0, -0.1])
bias = np.array([-2, -2, -2])

# Calculate the predicted output
pred = np.dot(weights, inputs) + bias
class InputLayer:
    def __init__(self, n_qubits):
        # The __init__ method is a special method that initializes the object when it's created.
        # self.n_qubits = n_qubits
        # self.circuit = QuantumCircuit(n_qubits)
        # self.circuit.h(range(n_qubits))
        
        # self.circuit.measure_all()
        
        # self.backend = Aer.get_backend('qasm_simulator')
        # self.result = execute(self.circuit, self.backend).result()
        # self.counts = self.result.get_counts(self.circuit)
        # # Create a 1D array of counts
        # counts_values = list(self.counts.values())
        
        # # Convert it to a NumPy matrix
        # self.total_counts = np.matrix(counts_values)
        # self.total_counts = self.total_counts.reshape((-1,n_qubits))
 
        
        #self.measured_configuration = sum(list(self.counts.keys()))/len(self.counts.keys())
        #self.learned_weights = np.array([int(bit) for bit in self.measured_configuration])

        pass
    