from __future__ import annotations

import hashlib
import secrets
from typing import Dict, Optional

from fastapi import HTTPException

from .store import read_users, write_users


def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def signup(username: str, password: str) -> Dict[str, str]:
    users = read_users()
    if username in users:
        raise HTTPException(status_code=400, detail="Username already exists.")
    salt = secrets.token_hex(8)
    users[username] = {"salt": salt, "password_hash": hash_password(password, salt)}
    write_users(users)
    return {"username": username}


def login(username: str, password: str, tokens: Dict[str, str]) -> str:
    users = read_users()
    user = users.get(username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    if hash_password(password, user["salt"]) != user["password_hash"]:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = secrets.token_hex(16)
    tokens[token] = username
    return token


def logout(token: str, tokens: Dict[str, str]) -> None:
    tokens.pop(token, None)


def require_user(token: Optional[str], tokens: Dict[str, str]) -> str:
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token.")
    username = tokens.get(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid auth token.")
    return username
