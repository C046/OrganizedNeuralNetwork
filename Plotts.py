# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:04:43 2024

@author: hadaw
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from OrganizedNeuralNetwork.Acts import Activations


class Plotter:
    def __init__(self, name="Function"):
        
        self.name=name
        
    def plot_3d(self, inputs, func):
        X,Y = np.meshgrid(inputs, func)
        Z = self.Sigmoid((X+Y))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis')

        # Set labels
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Sigmoid(X + Y)')

        # Show the plot
        plt.title("3D Surface Plot of Your Sigmoid Function")
        plt.show()
        
    def plot_loss_curve(self, loss_history):
        """
        Plots the training loss curve over epochs.
        
        Parameters:
            - loss_history (list): List of loss values for each epoch.
        """
        plt.plot(loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.show()