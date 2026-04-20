from cryptography.fernet import Fernet
from fastapi import Header, HTTPException, Depends

# In a production app, you would store this key in an Environment Variable
# We will generate a fixed key for this project demo.
# To generate a new key: Fernet.generate_key()
MASTER_KEY = b'uWTMz4Wv7UvV0-m6S7b3Y_5NfXwS2-C6_i-8S123456='

class DataEncryptor:
    def __init__(self):
        self.cipher = Fernet(MASTER_KEY)

    def encrypt(self, data: str) -> str:
        """Encrypts a string into a secure hash."""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypts a hash back into raw text."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

# Simple Role-Based Access Simulation
# We define hardcoded tokens for Admin and User roles
TOKENS = {
    "admin_token_123": "admin",
    "user_token_456": "user"
}

def verify_token(x_token: str = Header(...)):
    """
    Dependency to verify the presence and validity of an X-Token header.
    """
    if x_token not in TOKENS:
        raise HTTPException(status_code=401, detail="Invalid or Missing Security Token (X-Token)")
    return TOKENS[x_token]

# Export single instance for encryption
encryptor = DataEncryptor()
