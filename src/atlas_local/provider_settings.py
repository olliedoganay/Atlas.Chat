from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .security import (
    application_secret_protection_available,
    protect_bytes,
    unprotect_bytes,
)


PROVIDER_SETTINGS_FILENAME = "provider-settings.json"
PROVIDER_SETTINGS_FORMAT = "atlas-provider-settings-v1"


def provider_settings_path(data_dir: Path) -> Path:
    return data_dir / PROVIDER_SETTINGS_FILENAME


def load_provider_settings(data_dir: Path) -> dict[str, str]:
    path = provider_settings_path(data_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict) or payload.get("format") != PROVIDER_SETTINGS_FORMAT:
        return {}

    settings = {
        "provider": str(payload.get("provider", "") or "").strip(),
        "base_url": str(payload.get("base_url", "") or "").strip(),
    }
    encoded_key = str(payload.get("protected_api_key", "") or "").strip()
    if encoded_key:
        try:
            protected_key = base64.b64decode(encoded_key.encode("ascii"), validate=True)
            settings["api_key"] = unprotect_bytes(
                protected_key,
                entropy=b"atlas-provider-api-key-v1",
            ).decode("utf-8")
        except (OSError, ValueError, UnicodeError, RuntimeError):
            # A provider can still be selected without exposing or corrupting a
            # key that the current OS account cannot unlock.
            settings["api_key_unavailable"] = "true"
    return settings


def save_provider_settings(
    data_dir: Path,
    *,
    provider: str,
    base_url: str,
    api_key: str | None,
    preserve_existing_key: bool = False,
) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = provider_settings_path(data_dir)
    existing = _load_raw_settings(path)
    payload: dict[str, Any] = {
        "format": PROVIDER_SETTINGS_FORMAT,
        "provider": provider.strip(),
        "base_url": base_url.strip(),
    }

    resolved_key = (api_key or "").strip()
    if resolved_key:
        if not application_secret_protection_available():
            raise RuntimeError(
                "Atlas cannot save a provider API key because secure OS key storage is unavailable."
            )
        protected_key = protect_bytes(
            resolved_key.encode("utf-8"),
            entropy=b"atlas-provider-api-key-v1",
            description="Atlas provider API key",
        )
        payload["protected_api_key"] = base64.b64encode(protected_key).decode("ascii")
    elif preserve_existing_key and existing.get("protected_api_key"):
        payload["protected_api_key"] = existing["protected_api_key"]

    _write_json_atomic(path, payload)
    return {
        "provider": payload["provider"],
        "base_url": payload["base_url"],
        "has_api_key": bool(payload.get("protected_api_key")),
        "restart_required": True,
    }


def clear_provider_api_key(data_dir: Path) -> None:
    path = provider_settings_path(data_dir)
    payload = _load_raw_settings(path)
    if not payload:
        return
    payload.pop("protected_api_key", None)
    _write_json_atomic(path, payload)


def _load_raw_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            temp_path.chmod(0o600)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
