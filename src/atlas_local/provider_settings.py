from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .local_provider import normalize_local_provider_base_url
from .security import (
    application_secret_protection_available,
    protect_bytes,
    unprotect_bytes,
)


PROVIDER_SETTINGS_FILENAME = "provider-settings.json"
PROVIDER_SETTINGS_FORMAT = "atlas-provider-settings-v1"
MAX_PROVIDER_SETTINGS_BYTES = 64 * 1024
MAX_PROTECTED_API_KEY_LENGTH = 16 * 1024
MAX_API_KEY_LENGTH = 4096


def provider_settings_path(data_dir: Path) -> Path:
    return data_dir / PROVIDER_SETTINGS_FILENAME


def load_provider_settings(data_dir: Path) -> dict[str, str]:
    path = provider_settings_path(data_dir)
    payload = _load_raw_settings(path)
    if not isinstance(payload, dict) or payload.get("format") != PROVIDER_SETTINGS_FORMAT:
        return {}

    raw_base_url = str(payload.get("base_url", "") or "").strip()[:2048]
    base_url_invalid = False
    try:
        resolved_base_url = (
            normalize_local_provider_base_url(raw_base_url)
            if raw_base_url
            else ""
        )
    except ValueError:
        # Older Atlas versions accepted remote or malformed provider URLs.
        # Keep the provider/key recoverable while forcing the runtime onto its
        # safe local default until the operator saves a replacement.
        resolved_base_url = ""
        base_url_invalid = True
    settings = {
        "provider": str(payload.get("provider", "") or "").strip()[:64],
        "base_url": resolved_base_url,
    }
    if base_url_invalid:
        settings["base_url_invalid"] = "true"
    encoded_key = str(payload.get("protected_api_key", "") or "").strip()
    if encoded_key:
        try:
            if len(encoded_key) > MAX_PROTECTED_API_KEY_LENGTH:
                raise ValueError("protected provider key is too large")
            protected_key = base64.b64decode(encoded_key.encode("ascii"), validate=True)
            api_key = unprotect_bytes(
                protected_key,
                entropy=b"atlas-provider-api-key-v1",
                require_protection=True,
            ).decode("utf-8")
            if len(api_key) > MAX_API_KEY_LENGTH:
                raise ValueError("provider key is too large")
            settings["api_key"] = api_key
        except Exception:
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
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        try:
            data_dir.chmod(0o700)
        except OSError:
            pass
    path = provider_settings_path(data_dir)
    existing = _load_raw_settings(path)
    resolved_base_url = normalize_local_provider_base_url(base_url)
    existing_provider = str(existing.get("provider", "") or "").strip()
    try:
        existing_base_url = normalize_local_provider_base_url(
            str(existing.get("base_url", "") or "")
        )
    except ValueError:
        existing_base_url = ""
    payload: dict[str, Any] = {
        "format": PROVIDER_SETTINGS_FORMAT,
        "provider": provider.strip(),
        "base_url": resolved_base_url,
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
            require_protection=True,
        )
        payload["protected_api_key"] = base64.b64encode(protected_key).decode("ascii")
    elif (
        preserve_existing_key
        and existing.get("format") == PROVIDER_SETTINGS_FORMAT
        and provider.strip() == existing_provider
        and resolved_base_url == existing_base_url
        and isinstance(existing.get("protected_api_key"), str)
        and 0 < len(existing["protected_api_key"]) <= MAX_PROTECTED_API_KEY_LENGTH
    ):
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
        if path.stat().st_size > MAX_PROVIDER_SETTINGS_BYTES:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if os.name != "nt":
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
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
        if os.name != "nt":
            path.chmod(0o600)
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    pass
                finally:
                    os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)
