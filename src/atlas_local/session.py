from __future__ import annotations

import base64
import json


_SCOPED_THREAD_PREFIX = "atlas-thread-v2:"


def scoped_thread_id(user_id: str, thread_id: str) -> str:
    """Return a versioned, collision-free checkpoint key for a profile thread."""

    payload = json.dumps(
        [str(user_id), str(thread_id)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"{_SCOPED_THREAD_PREFIX}{encoded}"


def legacy_scoped_thread_id(user_id: str, thread_id: str) -> str:
    """Return the pre-v2 key so existing checkpoints can be read and removed."""

    return f"{user_id}::{thread_id}"


def parse_scoped_thread_id(value: str) -> tuple[str, str] | None:
    """Decode a v2 key, returning ``None`` for legacy or malformed values."""

    if not value.startswith(_SCOPED_THREAD_PREFIX):
        return None
    encoded = value[len(_SCOPED_THREAD_PREFIX) :]
    if not encoded:
        return None
    try:
        padding = "=" * (-len(encoded) % 4)
        payload = base64.urlsafe_b64decode((encoded + padding).encode("ascii"))
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError):
        return None
    if not isinstance(decoded, list) or len(decoded) != 2:
        return None
    user_id, thread_id = decoded
    if not isinstance(user_id, str) or not isinstance(thread_id, str):
        return None
    return user_id, thread_id
