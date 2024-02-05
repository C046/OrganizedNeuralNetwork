# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 08:29:02 2024

@author: hadaw
"""


import re
import string
import pandas as pd
import requests
from OrganizedNeuralNetwork.Books.open import *

def load_data(Name):
    return Books(f"{Name}").content

data = load_data("Frankenstein.txt")

class Token:
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Include all punctuation symbols in the character set
        self.token_pattern = re.compile(r' ')
    def contains_punctuation(self, token):
        # Check if any character in the token is a punctuation symbol
        return any(char in string.punctuation for char in token)
    
    def get_definitions(self, Wordlist, dictionaryType="collegiate", api_key="7157541e-2e7a-4a18-bd9c-f3a1724fc6fc"):
        Wordlist = list(set(Wordlist))
        def get_definition(Word, dictionaryType=dictionaryType, api_key=api_key):
            base_url = "https://www.dictionaryapi.com/api/v3/references"
            endpoint = f"{base_url}/{dictionaryType}/json/{Word}"
        
            try:
                resp = requests.get(endpoint, params={"key":api_key})
                content = resp.content
                print(resp)
            
                if isinstance(resp.json(), list):
                    return resp.json()[0]["shortdef"]
        
            except requests.exceptions.HTTPError as HTTPError:
                print(f"HTTP Error: {HTTPError}")
            
            except requests.exceptions.RequestException as err:
                print(f"Request Error: {err}")
        
            except Exception as e:
                print(f"An unexpected error has occured: {e}")
        
            return ["Error retrieving definition"]
        definition = [{word:get_definition("\n"+word, dictionaryType=dictionaryType, api_key=api_key)} for word in Wordlist]
        
        return definition
    
    def tokenize(self, Message):
        ##########################################################
        """Ye, right in this section right here"""
        tokens = self.token_pattern.split(Message) # over here
        # Remove empty strings from the list       # over here
        tokens = [token for token in tokens if token] # over here
        # Isolate these tokens more                   # over here
        """This is probably where you need to work""" # over here
        #########################################################


        # Initialize scorecards
        scorecard = {"Statements": [], "Questions": []}

        for word in tokens:
            # Check if the token contains any punctuation
            if self.contains_punctuation(word):
                # Create a scorecard system for statements
                if word[-1] == ".":
                    word = word.removesuffix(".")
                    scorecard["Statements"].append(True)
                else:
                    scorecard["Statements"].append(False)
                
                # Create a scorecard for questions
                if word[-1] == "?":
                    word = word.removesuffix("?")
                    scorecard["Questions"].append(True)
                else:
                    scorecard["Questions"].append(False)

        # Convert scorecards to pandas Series
        statement_series = pd.Series(scorecard["Statements"], name="Statements")
        questions_series = pd.Series(scorecard["Questions"], name="Questions")

        return (pd.concat([statement_series, questions_series], axis=1), tokens)

# Example usage
T = Token()
ScoreCard, MessageTokens = T.tokenize(data)
e = T.get_definitions(MessageTokens)