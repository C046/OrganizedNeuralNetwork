try:    
    # Attempt to import cupy
    import cupy as np
    
except ImportError:
    # Attempt to import numpy 
    import numpy as np
    
# Import os module
import os

# Import cryptography modules
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes

# Define Custom imports below
from OrganizedNeuralNetwork.Rand.Random import Rand


class PECG:
    """
    Elliptic Curve Cryptography using a Password-Encrypted Key Pair.

    Attributes:
        size (int): Size used for generating a random password.
        password (str): Random password used for encrypting the private key.
        private_key (cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePrivateKey): Private key.
        public_key (cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePublicKey): Public key.
        serialized_private_key (bytes): Serialized and encrypted private key.
        serialized_public_key (bytes): Serialized public key.
        deserialized_private_key (cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePrivateKey): De-serialized private key.
        deserialized_public_key (cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePublicKey): De-serialized public key.
    """
    def __init__(self, size):
        """
        Initialize PECG with a random password and generate key pairs.

        Parameters:
            size (int): Size used for generating a random password.
        """
        super().__init__()
        # Generate a random password for encrypting the private key
        self.password = Rand(size, shuffle_count=np.random.randint(size // 2, size), special_chars=True, capitals=True, lower=True, numbers=True, bytes=True).string
        # Generate key pairs
        self.private_key, self.public_key = self.generate_key_pair()
        # Serialize private and public keys
        self.serialized_private_key = self.serialize_private_key(self.private_key, size)
        self.serialized_public_key = self.serialize_public_key(self.public_key)
        # De-serialize private and public keys
        self.deserialized_private_key = self.deserialize_private_key(self.serialized_private_key)
        self.deserialized_public_key = self.deserialize_public_key(self.serialized_public_key)

    def generate_key_pair(self):
        """
        Generate Elliptic Curve key pair using the SECP521R1 curve.

        Returns:
            Tuple: (Private key, Public key)
        """
        private_key = ec.generate_private_key(ec.SECP521R1(), default_backend())
        public_key = private_key.public_key()
        return private_key, public_key

    def serialize_private_key(self, key, size):
        """
        Serialize and encrypt the private key.

        Parameters:
            key (cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePrivateKey): Private key.
            size (int): Size used for generating a random password.

        Returns:
            bytes: Serialized and encrypted private key.
        """
        return key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.BestAvailableEncryption(self.password)
        )

    def serialize_public_key(self, key):
        """
        Serialize the public key.

        Parameters:
            key (cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePublicKey): Public key.
            size (int): Size used for generating a random password.

        Returns:
            bytes: Serialized public key.
        """
        return key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH
        )

    def deserialize_private_key(self, key):
        """
        De-serialize and decrypt the private key.

        Parameters:
            key (bytes): Serialized and encrypted private key.

        Returns:
            cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePrivateKey: De-serialized private key.
        """
        return serialization.load_ssh_private_key(key, self.password, backend=default_backend())

    def deserialize_public_key(self, key):
        """
        De-serialize the public key.

        Parameters:
            key (bytes): Serialized public key.

        Returns:
            cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePublicKey: De-serialized public key.
        """
        return serialization.load_ssh_public_key(key, backend=default_backend())
    
    
    def encrypt_message(self, message, public_key, associated_data):
        self.iv = os.urandom(12)
        shared_key = self.private_key.exchange(ec.ECDH(), peer_public_key=public_key)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=32,
            salt=os.urandom(12),
            iterations=100000,
            backend=default_backend()
        )
        derived_key = kdf.derive(shared_key)
        
        cipher = Cipher(
            algorithms.AES256(derived_key),
            modes.GCM(self.iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(message) + encryptor.finalize()
  
        
        return (ciphertext, encryptor.tag)
    
    def decrypt_message(self, ciphertext, public_key, authTag, associated_data):
        shared_key = self.private_key.exchange(ec.ECDH(), peer_public_key=public_key)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=32,
            salt=self.iv,
            iterations=100000,
            backend=default_backend()
        )
        derived_key = kdf.derive(shared_key)
        
        cipher = Cipher(
            algorithms.AES256(derived_key),
            modes.GCM(self.iv, authTag),
            backend=default_backend()
        )

        decryptor = cipher.decryptor()
        decryptor.authenticate_additional_data(associated_data)
    
        # This is the last part you have to fix then its finished.
        decrypted_bytes = decryptor.update(ciphertext) + decryptor.finalize()
        decrypted_message = decrypted_bytes.decode('utf-8', errors="replace")
        return decrypted_message

        
        
    




pecg = PECG(1000)
private_key = pecg.private_key
public_key = pecg.public_key

ciphertext, authTag = pecg.encrypt_message(b"This is a message", public_key, b"This is the associated data.")

message = pecg.decrypt_message(ciphertext, public_key, authTag, b'This is the associated data.')



