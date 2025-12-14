from datetime import datetime, timedelta, timezone
import json
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from mlops_rakuten.config.constants import (
    AUTH_DIR,
)

USERS_PATH = AUTH_DIR / "users.json"

SECRET_KEY = "RAKUTEN_SECRET_KEY"  # À mettre dans .env var en prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def load_users() -> Dict[str, Dict[str, Any]]:
    if not USERS_PATH.exists():
        raise FileNotFoundError(f"users.json introuvable: {USERS_PATH}")
    users_list = json.loads(USERS_PATH.read_text(encoding="utf-8"))
    return {u["username"]: u for u in users_list}


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    users = load_users()
    u = users.get(username)
    if not u or not u.get("is_active", True):
        return None
    if not verify_password(password, u["password_hash"]):
        return None
    return u


def create_access_token(
    username: str, role: str, minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES
) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        if not username or not role:
            raise HTTPException(status_code=401, detail="Token invalide")
        return {"username": username, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide ou expiré")


def require_user(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if user["role"] not in {"user", "admin"}:
        raise HTTPException(status_code=403, detail="Accès interdit")
    return user


def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin requis")
    return user
