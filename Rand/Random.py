import numpy as np

class Rand:
    def __init__(self, size, shuffle_count=1, special_chars=False, capitals=False, lower=False, numbers=False, bytes=False):
        self.size = size
        self.special_chars = special_chars
        self.capitals = capitals
        self.lower = lower
        self.numbers = numbers
        self.bytes = bytes
        
        self.string = self.random_string(self.size, shuffle_count)
        

    def random_string(self, size, shuffle_count):
        string = ""
        cache = ""

        # Define the characters for the character set
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        capitals = alphabet.capitalize()
        numbers = "0123456789"
        special_chars = "`~!@#$%^&*()_+-=[]{}|;':\",.<>/?"
        
        # Define a character set object
        char_set = ""
        
        # Include capital-case alphabet if specified
        if self.capitals:
            char_set+=capitals
        
        # Include lower-case alphabet if specified
        if self.lower:
            char_set+=alphabet
            
        # Include numbers if specified
        if self.numbers:
            char_set+=numbers
            
        # Include special characters if specified
        if self.special_chars:
            char_set+=special_chars


        # Generate random characters within the specified character set
        for _ in range(size):
            try:  
                random_char = np.random.choice(list(char_set))
                
            except ValueError as valErr:
                print(f"\nError:\n    Please specify what you want to be included in the random selection.\n\n    out:\n        {valErr}\n")
                return None
            
            string += random_char
            cache += random_char

        # Perform multiple shuffles
        for _ in range(shuffle_count):
            cache = ''.join(np.random.permutation(list(cache)))
        
        if self.bytes == True:
            return bytes(string.encode("utf-8"))
        else:
            return string 
        
        
# # Example usage
# rand_instance = Rand(1000, shuffle_count=5, special_chars=True,capitals=True,lower=True,numbers=True,bytes=True)
# print("Final String:", rand_instance.string)
