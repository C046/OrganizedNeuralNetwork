# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 07:45:39 2024

@author: hadaw
"""

import os

class Books:
    def __init__(self, Name):
        self.ext = ".txt"
        if self.ext in Name:
            # Remove the suffix because we add this later
            Name = Name.removesuffix(self.ext)
            
        self.name = Name
        self.content = self._openBook()
        
        if self.content == None:
            print(f"Unable to retrieve the book: '{self.name}'")
        else:
            print(f"Successfully retrieved the book: '{self.name}'")
            
    def _openBook(self):
        path =( os.path.abspath("OrganizedNeuralNetwork\\Books") + "\\" + self.name + self.ext).replace("\\", "/")
        try:
            with open(path, "r", encoding="utf-8") as file:
                book_content = file.read()
                
        except Exception as e:
            print(f"Error opening the book '{self.name}': {e}")
            book_content = None
        
        # delete everything un-needed
        del self, path
        
        return book_content
    
