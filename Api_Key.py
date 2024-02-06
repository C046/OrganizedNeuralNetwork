from OrganizedNeuralNetwork.MessageEncrypt import *

   
def encryptKey(key):
    iv, ciphertext, tag, derived_key, associated_data = pecg.encrypt_message(api_key, pecg.public_key)
    with open("encryptedMessage.txt", "wb") as file:
        file.write(ciphertext)
    
    file.close()
    
    with open("decryptionKey.txt", "wb") as file:
        file.write(derived_key)
        
    file.close()
    
    with open("decryptionTag.txt", "wb") as file:
        file.write(tag)
    
    file.close()
    
    with open("associatedData.txt", "wb") as file:
        file.write(associated_data)
    
    file.close()
    
    with open("iv.txt", "wb") as file:
        file.write(iv)
    
    file.close()
    del iv, ciphertext, tag, derived_key, associated_data, file
    

def load_api_key(key_file, tag_file, ciphertext_file, associated_data_file, iv_file):
    with open(key_file, "rb") as file:
        decryption_key = file.read()

    with open(tag_file, "rb") as file:
        decryption_tag = file.read()

    with open(ciphertext_file, "rb") as file:
        ciphertext = file.read()

    with open(associated_data_file, "rb") as file:
        associated_data = file.read()

    with open(iv_file, "rb") as file:
        iv = file.read()

    return (ciphertext, decryption_key, decryption_tag, associated_data, iv)

if __name__ == "__main__":
    pass

