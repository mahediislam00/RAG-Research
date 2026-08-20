"""Authentication: password hashing, session-bound JWTs, and the FastAPI
dependency that turns a bearer token into a trusted user identity.

Design rules that the rest of the app relies on:

  1. ``user_id`` is NEVER accepted from the client. It is derived from a signed
     token, here, and passed down to the store. A request body cannot name a
     user, so it cannot address another user's data.
  2. Tokens are bound to a row in ``sessions``. Signature validity alone is not
     enough — the session must still exist, be unrevoked, and be unexpired. That
     makes logout and "revoke everything" instant rather than "wait for expiry".
  3. Passwords are stored as scrypt hashes with a per-password salt. Verification
     is constant-time.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
import uuid

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

from . import config, db

# --- password hashing ---------------------------------------------------
# scrypt is in the stdlib (no bcrypt/argon2 wheel to install) and is memory-hard.
_SCRYPT = dict(n=2**14, r=8, p=1, dklen=32)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return "scrypt${}${}".format(
        base64.b64encode(salt).decode(), base64.b64encode(dk).decode()
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_b64, dk_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
    except Exception:
        return False
    candidate = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return hmac.compare_digest(candidate, expected)


def check_password_strength(password: str) -> None:
    if len(password) < 10:
        raise HTTPException(400, "Password must be at least 10 characters.")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise HTTPException(400, "Password must contain letters and at least one digit.")


# --- JWT (HS256, no third-party dependency) -----------------------------
def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64u_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(msg: bytes) -> bytes:
    return hmac.new(config.JWT_SECRET.encode(), msg, hashlib.sha256).digest()


def issue_token(user_id: str, session_id: str) -> tuple[str, int]:
    now = int(time.time())
    exp = now + config.JWT_TTL_SECONDS
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": user_id, "sid": session_id, "iat": now, "exp": exp,
               "iss": config.JWT_ISSUER}
    segments = [
        _b64u(json.dumps(header, separators=(",", ":")).encode()),
        _b64u(json.dumps(payload, separators=(",", ":")).encode()),
    ]
    signing_input = ".".join(segments).encode()
    segments.append(_b64u(_sign(signing_input)))
    return ".".join(segments), exp


def decode_token(token: str) -> dict:
    try:
        h_b64, p_b64, sig_b64 = token.split(".")
    except ValueError:
        raise _unauthorized("Malformed token.")
    expected = _sign(f"{h_b64}.{p_b64}".encode())
    if not hmac.compare_digest(_b64u_decode(sig_b64), expected):
        raise _unauthorized("Bad token signature.")
    try:
        payload = json.loads(_b64u_decode(p_b64))
    except Exception:
        raise _unauthorized("Malformed token payload.")
    if payload.get("iss") != config.JWT_ISSUER:
        raise _unauthorized("Bad token issuer.")
    if float(payload.get("exp", 0)) < time.time():
        raise _unauthorized("Token expired. Sign in again.")
    if not payload.get("sub") or not payload.get("sid"):
        raise _unauthorized("Incomplete token.")
    return payload


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail,
                         headers={"WWW-Authenticate": "Bearer"})


# --- login throttling ---------------------------------------------------
# Small in-process throttle so a stolen email can't be brute-forced from one
# host. Behind more than one worker, move this to Redis.
_attempts: dict[str, list[float]] = {}


def _throttle(key: str) -> None:
    now = time.time()
    window = [t for t in _attempts.get(key, []) if now - t < config.LOGIN_WINDOW_SECONDS]
    if len(window) >= config.LOGIN_MAX_ATTEMPTS:
        raise HTTPException(429, "Too many sign-in attempts. Wait a minute and retry.")
    window.append(now)
    _attempts[key] = window


def _clear_throttle(key: str) -> None:
    _attempts.pop(key, None)


# --- models -------------------------------------------------------------
class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class Principal(BaseModel):
    """The authenticated caller. This is the only source of user_id in the app."""
    user_id: str
    email: str
    session_id: str


# --- flows --------------------------------------------------------------
def register(creds: Credentials) -> tuple[Principal, str, int]:
    check_password_strength(creds.password)
    if db.get_user_by_email(creds.email):
        # Same wording as a weak password would give? No — the email is the
        # account name; a duplicate has to be reported to be usable. Signup
        # enumeration is accepted here; login does not leak.
        raise HTTPException(409, "An account with that email already exists.")
    user_id = f"user_{uuid.uuid4().hex[:16]}"
    db.create_user(user_id, str(creds.email), hash_password(creds.password))
    return login(creds)


def login(creds: Credentials) -> tuple[Principal, str, int]:
    key = str(creds.email).lower()
    _throttle(key)
    user = db.get_user_by_email(str(creds.email))
    # Verify against a dummy hash when the user is unknown so the response time
    # doesn't reveal whether the account exists.
    stored = user["password_hash"] if user else hash_password("not-a-real-password")
    ok = verify_password(creds.password, stored)
    if not user or not ok or not user["is_active"]:
        raise _unauthorized("Invalid email or password.")
    _clear_throttle(key)

    session_id = f"sess_{uuid.uuid4().hex}"
    db.create_session(session_id, user["user_id"], config.JWT_TTL_SECONDS)
    token, exp = issue_token(user["user_id"], session_id)
    principal = Principal(user_id=user["user_id"], email=user["email"],
                          session_id=session_id)
    return principal, token, exp


# --- dependency ---------------------------------------------------------
_bearer = HTTPBearer(auto_error=False)


def current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    """FastAPI dependency. Every data route depends on this; there is no code
    path that touches Qdrant without a Principal."""
    token = creds.credentials if creds else None
    if not token:
        # Browsers cannot set headers on EventSource, and the upload/chat
        # endpoints stream. This app streams via fetch(), so the header works —
        # but we also accept ?token= for tooling. Never log the query string.
        token = request.query_params.get("token")
    if not token:
        raise _unauthorized("Sign in to continue.")

    payload = decode_token(token)
    user_id, session_id = payload["sub"], payload["sid"]

    if not db.session_is_valid(session_id, user_id):
        raise _unauthorized("Session ended. Sign in again.")
    user = db.get_user(user_id)
    if not user or not user["is_active"]:
        raise _unauthorized("Account is not active.")
    return Principal(user_id=user_id, email=user["email"], session_id=session_id)
