"""Process-local keyring backend for deterministic Linux CI tests.

Production never selects this backend. GitHub Actions opts into it only while
running the backend test suite because hosted Linux runners do not provide a
desktop Secret Service session.
"""

from __future__ import annotations

import threading

from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError


class MemoryKeyring(KeyringBackend):
    """Minimal process-local keyring with no filesystem persistence."""

    priority = 1

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._passwords: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        with self._lock:
            return self._passwords.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        with self._lock:
            self._passwords[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        with self._lock:
            try:
                del self._passwords[(service, username)]
            except KeyError as exc:
                raise PasswordDeleteError("Password not found") from exc
