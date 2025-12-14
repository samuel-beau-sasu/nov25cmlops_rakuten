# scripts/hash_password.py
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

password = "password"
print(pwd_context.hash(password))
