# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:04:43 2024

@author: hadaw
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from OrganizedNeuralNetwork.Acts import Activations
class Plotter(Activations):
    def __init__(self, inputs, func, name="Function"):
        super(Plotter, self).__init__(inputs)
        self.inputs = inputs
        self.func = func
        self.name=name
        
    def plot_3d(self):
        X,Y = np.meshgrid(self.inputs, self.func)
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
        
    def plot(self):
        plt.plot(self.inputs, self.func, label=self.name)
        plt.title("Activation Function")
        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        