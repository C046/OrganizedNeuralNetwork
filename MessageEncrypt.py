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

        self.iterations=10000
        self.iv = os.urandom(12)
       
    def generate_key_pair(self):
        """
        Generate Elliptic Curve key pair using the SECP521R1 curve.

        Returns:
            Tuple: (Private key, Public key)
        """
        private_key = ec.generate_private_key(ec.SECP521R1(), default_backend())
        public_key = private_key.public_key()
        return private_key, public_key

    def serialize_private_key(self, key):
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
    
    def deserialize_PEM_public_key(self, key):
        """
        Deserialize a PEM-formatted public key.

        Parameters:
            key (bytes): Serialized PEM-formatted public key.

        Returns:
            cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePublicKey: De-serialized public key.
        """
        return serialization.load_pem_public_key(key, backend=default_backend())

    def derive_key(self, shared_key):
        """
        Derive a key using PBKDF2HMAC.

        Parameters:
            shared_key (bytes): Shared key.

        Returns:
            bytes: Derived key.
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=32,
            salt=self.iv,
            iterations=self.iterations,
            backend=default_backend()
        )
        return kdf.derive(shared_key)[:32]

    def encryptor(self, key):
        """
        Get an AES256 encryptor using GCM mode.

        Parameters:
            key (bytes): Key for encryption.

        Returns:
            cryptography.hazmat.backends.ciphers._cipher.CipherContext: AES256 encryptor.
        """
        return Cipher(
            algorithms.AES256(key),
            modes.GCM(self.iv),
            backend=default_backend()
        ).encryptor()

    def decryptor(self, key, tag):
        """
        Get an AES256 decryptor using GCM mode.

        Parameters:
            key (bytes): Key for decryption.
            tag (bytes): Tag for decryption.

        Returns:
            cryptography.hazmat.backends.ciphers._cipher.CipherContext: AES256 decryptor.
        """
        return Cipher(
            algorithms.AES256(key),
            modes.GCM(self.iv, tag),
            backend=default_backend()
        ).decryptor()

    def shared_key(self, private_key, public_key):
        """
        Compute shared key using ECDH.

        Parameters:
            private_key (cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePrivateKey): Private key.
            public_key (cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePublicKey): Public key.

        Returns:
            bytes: Shared key.
        """
        return private_key.exchange(ec.ECDH(), peer_public_key=public_key)

    def exchange_algorithm(self, peer_public_key):
        """
        Perform key exchange algorithm.

        Parameters:
            peer_public_key (cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePublicKey): Peer's public key.

        Returns:
            Tuple: Shared key, derived key.
        """
        peer_private_key, peer_public_key = self.generate_key_pair()
        shared_key = self.shared_key(self.private_key, peer_public_key)
        derived_key = self.derive_key(shared_key)
        return shared_key, derived_key

    def encrypt_message(self, message, key):
        """
        Encrypt a message using key exchange.

        Parameters:
            message (bytes): Message to be encrypted.
            key (cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePublicKey): Public key.

        Returns:
            Tuple: IV, ciphertext, tag, derived key, associated data.
        """
        key = key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        associated_data, _ = self.generate_key_pair()
        associated_data = self.serialize_private_key(associated_data)
        shared_key, _ = self.exchange_algorithm(key)
        derived_key = self.derive_key(shared_key)
        encryptor = self.encryptor(derived_key)
        encryptor.authenticate_additional_data(associated_data)
        ciphertext = encryptor.update(message) + encryptor.finalize()
        return self.iv, ciphertext.decode("latin"), encryptor.tag, derived_key, associated_data

    def decrypt_message(self, ciphertext, key, tag, associated_data):
        """
        Decrypt a message using key exchange.

        Parameters:
            ciphertext (str): Encrypted message.
            key (bytes): Key for decryption.
            tag (bytes): Tag for decryption.
            associated_data (bytes): Associated data.

        Returns:
            bytes: Decrypted message.
        """
        ciphertext = bytes(ciphertext.encode("latin"))
        decryptor = self.decryptor(key, tag)
        decryptor.authenticate_additional_data(associated_data)
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    
# pecg = PECG(1000)
# message = b"Hello world!"

# iv, ciphertext, tag, derived_key, associated_data = pecg.encrypt_message(message, pecg.public_key)
# ciphertext = ciphertext

# decryption = pecg.decrypt_message(ciphertext, derived_key, tag, associated_data)


