# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 08:29:02 2024

@author: hadaw
"""


import re
import string
import pandas as pd
import requests

class Token:
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Include all punctuation symbols in the character set
        self.token_pattern = re.compile(r' ')
    def contains_punctuation(self, token):
        # Check if any character in the token is a punctuation symbol
        return any(char in string.punctuation for char in token)
    
    def get_definitions(self, Wordlist, url="", api_key=""): 
        def get_(Word, url=url, api_key=""):
            
            url = url+Word
       
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    definitions = response.json()
                    return definitions[0][0]
                else:
                    print(f"Error: {response.status_code}")
    
            except Exception as E:
                print(f"An error has occured fucker: {E}")
            
           
        return [get_(word, url=url, api_key=api_key) for word in Wordlist]
            
    def tokenize(self, Message):
        tokens = self.token_pattern.split(Message)
        # Remove empty strings from the list
        tokens = [token for token in tokens if token]
        # Initialize scorecards
        scorecard = {"Statements": [], "Questions": []}

        for i in tokens:
            # Check if the token contains any punctuation
            if self.contains_punctuation(i):
                # Create a scorecard system for statements
                if i[-1] == ".":
                    scorecard["Statements"].append(True)
                else:
                    scorecard["Statements"].append(False)
                
                # Create a scorecard for questions
                if i[-1] == "?":
                    scorecard["Questions"].append(True)
                else:
                    scorecard["Questions"].append(False)

        # Convert scorecards to pandas Series
        statement_series = pd.Series(scorecard["Statements"], name="Statements")
        questions_series = pd.Series(scorecard["Questions"], name="Questions")

        return (pd.concat([statement_series, questions_series], axis=1), tokens)



    
# Example usage
T = Token()
ScoreCard, MessageTokens = T.tokenize("This is a message? This is a statement.")
e = T.get_definitions(MessageTokens,url=f"https://api.dictionaryapi.dev/api/v2/entries/en/")