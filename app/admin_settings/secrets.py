"""Sealing of secret settings (provider keys and the like) with AES-256-GCM.

The same scheme and stored shape as the web app's src/lib/secrets.ts ({v: 1, iv, tag, data}, base64; the key is the
SHA-256 of the configured text), with its own key: NEWS_AGENT_SETTINGS_KEY (32 characters or more). No two services
share a key (the web app, the seo-agent, this one), so a value sealed by one cannot be read by another. Unlike the web app there is no development
fallback key: without the key, secret fields are read-only ("Requires manual configuration").

Rotating the key makes stored secrets unreadable; they must be entered again.

Forked from moks-seo-agent/app/admin_settings/secrets.py@6d6464b on 2026-11-21.
"""

from __future__ import annotations

import base64
import hashlib
import os
from collections.abc import Mapping
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_ENV = "NEWS_AGENT_SETTINGS_KEY"
MIN_KEY = 32


class NoKey(Exception):
    """NEWS_AGENT_SETTINGS_KEY is unset or too short."""


class Unreadable(Exception):
    """A stored secret was sealed with another key, or is damaged."""


def _key(env: Mapping[str, str] | None = None) -> bytes:
    raw = (os.environ if env is None else env).get(KEY_ENV, "")
    if len(raw) < MIN_KEY:
        raise NoKey(f"{KEY_ENV} ({MIN_KEY}+ characters) is not set")
    return hashlib.sha256(raw.encode()).digest()


def available(env: Mapping[str, str] | None = None) -> bool:
    try:
        _key(env)
    except NoKey:
        return False
    return True


def is_sealed(value: Any) -> bool:
    return isinstance(value, dict) and value.get("v") == 1 and all(isinstance(value.get(k), str) for k in ("iv", "tag", "data"))


def seal(plaintext: str, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    iv = os.urandom(12)
    out = AESGCM(_key(env)).encrypt(iv, plaintext.encode(), None)  # ciphertext followed by the 16-byte tag
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return {"v": 1, "iv": b64(iv), "tag": b64(out[-16:]), "data": b64(out[:-16])}


def unseal(sealed: Any, env: Mapping[str, str] | None = None) -> str:
    if not is_sealed(sealed):
        raise Unreadable("not a sealed value")
    try:
        iv, tag, data = (base64.b64decode(sealed[k], validate=True) for k in ("iv", "tag", "data"))
        return AESGCM(_key(env)).decrypt(iv, data + tag, None).decode()
    except (InvalidTag, ValueError) as e:
        raise Unreadable("sealed with another key, or damaged") from e
