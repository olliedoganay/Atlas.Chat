from __future__ import annotations

import base64
import ctypes
import hashlib
import hmac
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import closing
from ctypes import wintypes
from pathlib import Path
from typing import Any

try:
    from sqlcipher3 import dbapi2 as sqlcipher_dbapi
except ImportError:  # pragma: no cover - dependency is required in Windows builds
    sqlcipher_dbapi = None

try:
    import keyring
except ImportError:  # pragma: no cover - dependency is required in non-Windows source builds
    keyring = None

try:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:  # pragma: no cover - dependency is required in non-Windows source builds
    AESGCM = None
    InvalidTag = ValueError  # type: ignore[misc, assignment]


CRYPTPROTECT_UI_FORBIDDEN = 0x01
_STORAGE_KEY_FORMAT = "atlas-dpapi-storage-key-v1"
_SQLITE_HEADER = b"SQLite format 3\x00"
_NON_WINDOWS_FORMAT = b"atlas-aesgcm-v1\0"
_CALLER_KEY_FORMAT = b"atlas-key-aesgcm-v1\0"
_KEYRING_SERVICE = "Atlas"
_KEYRING_ACCOUNT = "atlas-storage-master-key-v1"
_MIGRATION_BACKUP_DIR = "migration-backups"
_STORAGE_KEY_LENGTH = 32
_STORAGE_KEY_LOCK = threading.Lock()
_STORAGE_KEY_CREATION_TIMEOUT_SECONDS = 10.0
_STORAGE_KEY_CREATION_POLL_SECONDS = 0.05


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def protect_bytes(
    data: bytes,
    *,
    entropy: bytes | None = None,
    description: str = "Atlas",
    require_protection: bool = False,
) -> bytes:
    if os.name != "nt":
        if not _non_windows_secret_storage_supported():
            if require_protection:
                raise RuntimeError(
                    "Atlas could not access secure OS key storage on this machine."
                )
            return data
        master_key = _get_or_create_non_windows_master_key()
        encryption_key = _derive_non_windows_encryption_key(master_key, entropy=entropy)
        nonce = os.urandom(12)
        encrypted = AESGCM(encryption_key).encrypt(nonce, data, None)
        return _NON_WINDOWS_FORMAT + nonce + encrypted

    input_blob, _input_buffer = _blob_from_bytes(data)
    entropy_blob, _entropy_buffer = (
        _blob_from_bytes(entropy) if entropy else (None, None)
    )
    output_blob = _DataBlob()

    result = ctypes.windll.crypt32.CryptProtectData(  # type: ignore[attr-defined]
        ctypes.byref(input_blob),
        ctypes.c_wchar_p(description),
        ctypes.byref(entropy_blob) if entropy_blob else None,
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output_blob),
    )
    if not result:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(output_blob.pbData)  # type: ignore[attr-defined]


def unprotect_bytes(
    data: bytes,
    *,
    entropy: bytes | None = None,
    require_protection: bool = False,
) -> bytes:
    if os.name != "nt":
        if not data.startswith(_NON_WINDOWS_FORMAT):
            if require_protection:
                raise RuntimeError("Protected data used an unprotected legacy format.")
            return data
        if not _non_windows_secret_storage_supported():
            raise RuntimeError("Atlas could not access OS keyring storage on this machine.")
        master_key = _get_or_create_non_windows_master_key()
        encryption_key = _derive_non_windows_encryption_key(master_key, entropy=entropy)
        nonce = data[len(_NON_WINDOWS_FORMAT) : len(_NON_WINDOWS_FORMAT) + 12]
        ciphertext = data[len(_NON_WINDOWS_FORMAT) + 12 :]
        try:
            return AESGCM(encryption_key).decrypt(nonce, ciphertext, None)
        except InvalidTag as exc:
            raise RuntimeError("Protected data authentication failed.") from exc

    input_blob, _input_buffer = _blob_from_bytes(data)
    entropy_blob, _entropy_buffer = (
        _blob_from_bytes(entropy) if entropy else (None, None)
    )
    output_blob = _DataBlob()

    result = ctypes.windll.crypt32.CryptUnprotectData(  # type: ignore[attr-defined]
        ctypes.byref(input_blob),
        None,
        ctypes.byref(entropy_blob) if entropy_blob else None,
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output_blob),
    )
    if not result:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(output_blob.pbData)  # type: ignore[attr-defined]


def protect_bytes_with_key(data: bytes, *, key: bytes, aad: bytes = b"") -> bytes:
    """Encrypt bytes with a caller-owned 256-bit key without consulting OS storage."""
    if AESGCM is None:
        raise RuntimeError("AES-GCM support is not available.")
    if len(key) != _STORAGE_KEY_LENGTH:
        raise ValueError("AES-GCM keys must be exactly 32 bytes.")
    nonce = os.urandom(12)
    encrypted = AESGCM(key).encrypt(nonce, data, aad)
    return _CALLER_KEY_FORMAT + nonce + encrypted


def unprotect_bytes_with_key(data: bytes, *, key: bytes, aad: bytes = b"") -> bytes:
    """Decrypt bytes produced by protect_bytes_with_key with the caller-owned key."""
    if AESGCM is None:
        raise RuntimeError("AES-GCM support is not available.")
    if len(key) != _STORAGE_KEY_LENGTH:
        raise ValueError("AES-GCM keys must be exactly 32 bytes.")
    if not data.startswith(_CALLER_KEY_FORMAT):
        raise ValueError("Unsupported caller-key ciphertext format.")
    encoded = data[len(_CALLER_KEY_FORMAT) :]
    if len(encoded) < 12 + 16:
        raise ValueError("Caller-key ciphertext is truncated.")
    nonce = encoded[:12]
    ciphertext = encoded[12:]
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, aad)
    except InvalidTag as exc:
        raise ValueError("Caller-key ciphertext authentication failed.") from exc


def sqlcipher_enabled() -> bool:
    return sqlcipher_dbapi is not None


def local_secret_storage_label() -> str:
    if os.name == "nt":
        return "windows-dpapi"
    if _non_windows_secret_storage_supported():
        return "os-keyring"
    return "not-available"


def application_secret_protection_available() -> bool:
    return local_secret_storage_label() != "not-available"


def get_or_create_storage_key(data_dir: Path) -> bytes:
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    _restrict_permissions(data_dir, directory=True)
    key_path = data_dir / "storage.key.json"
    with _STORAGE_KEY_LOCK:
        if key_path.exists():
            return _load_storage_key(key_path)

        creation_lock_path = data_dir / "storage.key.create.lock"
        creation_lock_fd = _acquire_storage_key_creation_lock(
            key_path=key_path,
            lock_path=creation_lock_path,
        )
        if creation_lock_fd is None:
            return _load_storage_key(key_path)
        try:
            if key_path.exists():
                return _load_storage_key(key_path)
            key = os.urandom(_STORAGE_KEY_LENGTH)
            payload = {
                "format": _STORAGE_KEY_FORMAT,
                "wrapped_key": base64.b64encode(
                    protect_bytes(
                        key,
                        description="Atlas storage key",
                        require_protection=True,
                    )
                ).decode("ascii"),
            }
            _write_private_json_atomic(key_path, payload)
            return key
        finally:
            os.close(creation_lock_fd)
            creation_lock_path.unlink(missing_ok=True)


def _acquire_storage_key_creation_lock(
    *,
    key_path: Path,
    lock_path: Path,
) -> int | None:
    deadline = time.monotonic() + _STORAGE_KEY_CREATION_TIMEOUT_SECONDS
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    while True:
        if key_path.exists():
            return None
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Atlas timed out waiting for another process to finish creating "
                    f"the storage key at {key_path}. The existing creation lock was left "
                    "in place to avoid replacing key material unsafely."
                ) from exc
            time.sleep(_STORAGE_KEY_CREATION_POLL_SECONDS)
            continue
        try:
            os.write(
                descriptor,
                f"pid={os.getpid()}\n".encode("ascii"),
            )
            os.fsync(descriptor)
            _restrict_permissions(lock_path, directory=False)
        except Exception:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)
            raise
        return descriptor


def _load_storage_key(key_path: Path) -> bytes:
    try:
        payload = json.loads(key_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format") != _STORAGE_KEY_FORMAT:
            raise ValueError("unsupported storage-key format")
        encoded = str(payload.get("wrapped_key", "") or "").strip()
        if not encoded:
            raise ValueError("missing wrapped storage key")
        wrapped = base64.b64decode(encoded.encode("ascii"), validate=True)
        try:
            key = unprotect_bytes(wrapped, require_protection=True)
        except RuntimeError:
            if not _legacy_plaintext_migration_enabled():
                raise
            key = unprotect_bytes(wrapped)
            if len(key) != _STORAGE_KEY_LENGTH:
                raise ValueError("invalid legacy storage-key length")
            migrated = {
                "format": _STORAGE_KEY_FORMAT,
                "wrapped_key": base64.b64encode(
                    protect_bytes(
                        key,
                        description="Atlas storage key",
                        require_protection=True,
                    )
                ).decode("ascii"),
            }
            _write_private_json_atomic(key_path, migrated)
        if len(key) != _STORAGE_KEY_LENGTH:
            raise ValueError("invalid storage-key length")
    except (OSError, ValueError, TypeError, UnicodeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Atlas storage key at {key_path} is invalid or unavailable. "
            "Atlas will not replace it because doing so could make existing encrypted data unreadable."
        ) from exc
    _restrict_permissions(key_path, directory=False)
    return key


def _legacy_plaintext_migration_enabled() -> bool:
    return os.environ.get(
        "ATLAS_ALLOW_LEGACY_PLAINTEXT_MIGRATION",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _non_windows_secret_storage_supported() -> bool:
    if keyring is None or AESGCM is None:
        return False
    try:
        backend = keyring.get_keyring()
    except Exception:
        return False
    return bool(getattr(backend, "priority", 0) > 0)


def _get_or_create_non_windows_master_key() -> bytes:
    if not _non_windows_secret_storage_supported():
        raise RuntimeError("Atlas could not access OS keyring storage on this machine.")
    try:
        stored = keyring.get_password(_KEYRING_SERVICE, _KEYRING_ACCOUNT)
        if stored:
            master_key = base64.b64decode(stored.encode("ascii"), validate=True)
            if len(master_key) != _STORAGE_KEY_LENGTH:
                raise ValueError("invalid Atlas keyring master-key length")
            return master_key
        master_key = os.urandom(_STORAGE_KEY_LENGTH)
        encoded_master_key = base64.b64encode(master_key).decode("ascii")
        keyring.set_password(
            _KEYRING_SERVICE,
            _KEYRING_ACCOUNT,
            encoded_master_key,
        )
        if keyring.get_password(_KEYRING_SERVICE, _KEYRING_ACCOUNT) != encoded_master_key:
            raise RuntimeError("Atlas could not verify the key saved to OS keyring storage.")
        return master_key
    except Exception as exc:
        raise RuntimeError("Atlas could not access OS keyring storage on this machine.") from exc


def _derive_non_windows_encryption_key(master_key: bytes, *, entropy: bytes | None) -> bytes:
    if not entropy:
        return hashlib.sha256(master_key + b":atlas").digest()
    return hmac.new(master_key, entropy, hashlib.sha256).digest()


def prepare_encrypted_sqlite(path: Path, *, data_dir: Path, reset_legacy: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _restrict_storage_parent(path.parent, data_dir=data_dir)
    if not path.exists():
        return
    _restrict_permissions(path, directory=False)
    if not sqlcipher_enabled():
        return
    if reset_legacy and _looks_like_plaintext_sqlite(path):
        _migrate_plaintext_sqlite(path, data_dir=data_dir)


def prepare_encrypted_qdrant_storage(path: Path, *, data_dir: Path, reset_legacy: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    _restrict_permissions(path, directory=True)
    if not reset_legacy:
        return
    if not sqlcipher_enabled():
        return
    plaintext_paths = [
        storage_path
        for storage_path in path.rglob("storage.sqlite")
        if _looks_like_plaintext_sqlite(storage_path)
    ]
    if not plaintext_paths:
        return
    _migrate_plaintext_qdrant_storage(path, data_dir=data_dir)


def purge_legacy_migration_backups(
    *,
    data_dir: Path,
    storage_paths: tuple[Path, ...] = (),
) -> None:
    """Remove plaintext recovery artifacts left by legacy storage migrations."""
    backup_roots = {data_dir / _MIGRATION_BACKUP_DIR}
    backup_roots.update(
        storage_path.parent / ".atlas-migration-backups"
        for storage_path in storage_paths
    )
    for backup_root in backup_roots:
        if backup_root.is_symlink() or backup_root.is_file():
            _unlink_with_retry(backup_root)
        elif backup_root.exists():
            _rmtree_with_retry(backup_root)
        _fsync_directory(backup_root.parent)


def open_application_sqlite(
    database: str | Path,
    *,
    data_dir: Path,
    check_same_thread: bool = False,
) -> Any:
    target = Path(str(database)) if str(database) != ":memory:" else None
    if target is not None:
        prepare_encrypted_sqlite(target, data_dir=data_dir)
    if not sqlcipher_enabled():
        connection = sqlite3.connect(str(database), check_same_thread=check_same_thread)
        if target is not None:
            _restrict_permissions(target, directory=False)
        return connection

    connection = sqlcipher_dbapi.connect(str(database), check_same_thread=check_same_thread)
    _apply_sqlcipher_key(connection, get_or_create_storage_key(data_dir))
    if target is not None:
        _restrict_permissions(target, directory=False)
    return connection


def build_encrypted_sqlite_module(*, data_dir: Path) -> Any:
    if not sqlcipher_enabled():
        return sqlite3

    class _SqlcipherModuleProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(sqlcipher_dbapi, name)

        def connect(self, database: str | Path, *args: Any, **kwargs: Any) -> Any:
            target = None if str(database) == ":memory:" else Path(str(database))
            if target is not None:
                prepare_encrypted_sqlite(target, data_dir=data_dir)
            connection = sqlcipher_dbapi.connect(str(database), *args, **kwargs)
            _apply_sqlcipher_key(connection, get_or_create_storage_key(data_dir))
            if target is not None:
                _restrict_permissions(target, directory=False)
            return connection

    return _SqlcipherModuleProxy()


def _apply_sqlcipher_key(connection: Any, key: bytes) -> None:
    if sqlcipher_dbapi is None:
        return
    connection.execute(f"PRAGMA key = \"x'{key.hex()}'\"")
    connection.execute("SELECT count(*) FROM sqlite_master")


def _looks_like_plaintext_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(_SQLITE_HEADER)) == _SQLITE_HEADER
    except OSError:
        return False


def _migrate_plaintext_sqlite(path: Path, *, data_dir: Path) -> None:
    if sqlcipher_dbapi is None or not _looks_like_plaintext_sqlite(path):
        return

    backup_path = _migration_backup_path(
        data_dir=data_dir,
        source=path,
        suffix=".plaintext.sqlite",
    )
    staged_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.encrypted.tmp")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    _restrict_permissions(backup_path.parent, directory=True)
    try:
        _snapshot_plaintext_sqlite(path, backup_path)
        _encrypt_sqlite_snapshot(
            backup_path,
            staged_path,
            key=get_or_create_storage_key(data_dir),
        )
        _verify_encrypted_sqlite(staged_path, key=get_or_create_storage_key(data_dir))
        _restrict_permissions(staged_path, directory=False)
        moved_sidecars: list[tuple[Path, Path]] = []
        try:
            for suffix in ("-wal", "-shm", "-journal"):
                sidecar_path = path.with_name(f"{path.name}{suffix}")
                if not sidecar_path.exists():
                    continue
                backup_sidecar_path = backup_path.with_name(f"{backup_path.name}{suffix}")
                os.replace(sidecar_path, backup_sidecar_path)
                moved_sidecars.append((sidecar_path, backup_sidecar_path))
            os.replace(staged_path, path)
        except Exception:
            for sidecar_path, backup_sidecar_path in reversed(moved_sidecars):
                if backup_sidecar_path.exists() and not sidecar_path.exists():
                    os.replace(backup_sidecar_path, sidecar_path)
            raise
        _fsync_directory(path.parent)
    except Exception as exc:
        staged_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Atlas could not safely migrate plaintext SQLite storage at {path}. "
            f"The original database was left in place and a recovery copy is available at {backup_path}."
        ) from exc

    try:
        _remove_sqlite_migration_backup(backup_path)
    except Exception as exc:
        raise RuntimeError(
            f"Atlas migrated SQLite storage at {path}, but could not remove its "
            f"plaintext recovery copy at {backup_path}."
        ) from exc


def _migrate_plaintext_qdrant_storage(path: Path, *, data_dir: Path) -> None:
    backup_path = _migration_backup_path(
        data_dir=data_dir,
        source=path,
        suffix=".plaintext.qdrant",
    )
    staged_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.encrypted.tmp")
    rollback_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.rollback")
    migration_committed = False
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    _restrict_permissions(backup_path.parent, directory=True)
    try:
        shutil.copytree(path, backup_path)
        for source_storage_path in path.rglob("storage.sqlite"):
            if not _looks_like_plaintext_sqlite(source_storage_path):
                continue
            backup_storage_path = backup_path / source_storage_path.relative_to(path)
            snapshot_path = backup_storage_path.with_name(
                f".{backup_storage_path.name}.{uuid.uuid4().hex}.snapshot.tmp"
            )
            _snapshot_plaintext_sqlite(source_storage_path, snapshot_path)
            os.replace(snapshot_path, backup_storage_path)
            for suffix in ("-wal", "-shm"):
                backup_storage_path.with_name(f"{backup_storage_path.name}{suffix}").unlink(
                    missing_ok=True
                )
        _restrict_tree_permissions(backup_path)
        shutil.copytree(backup_path, staged_path)
        for storage_path in staged_path.rglob("storage.sqlite"):
            if _looks_like_plaintext_sqlite(storage_path):
                _migrate_plaintext_sqlite(storage_path, data_dir=data_dir)
        remaining_plaintext = [
            storage_path
            for storage_path in staged_path.rglob("storage.sqlite")
            if _looks_like_plaintext_sqlite(storage_path)
        ]
        if remaining_plaintext:
            raise RuntimeError(
                "One or more staged Qdrant databases remained plaintext after migration."
            )

        os.replace(path, rollback_path)
        try:
            os.replace(staged_path, path)
        except Exception:
            os.replace(rollback_path, path)
            raise
        migration_committed = True
        _rmtree_with_retry(rollback_path)
        _rmtree_with_retry(backup_path)
        _remove_empty_directory(backup_path.parent)
        _fsync_directory(path.parent)
        return
    except Exception as exc:
        if staged_path.exists():
            _rmtree_with_retry(staged_path)
        if migration_committed:
            raise RuntimeError(
                f"Atlas migrated Qdrant storage at {path}, but could not remove all "
                f"plaintext recovery artifacts. Check {backup_path} and {rollback_path}."
            ) from exc
        if rollback_path.exists() and not path.exists():
            os.replace(rollback_path, path)
        elif rollback_path.exists():
            _rmtree_with_retry(rollback_path)
        raise RuntimeError(
            f"Atlas could not safely migrate plaintext Qdrant storage at {path}. "
            f"The original store was restored and a recovery copy is available at {backup_path}."
        ) from exc


def _snapshot_plaintext_sqlite(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    source_uri = f"file:{source.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(source_uri, uri=True)) as source_conn:
        with closing(sqlite3.connect(destination)) as destination_conn:
            source_conn.backup(destination_conn)
            destination_conn.commit()
    _restrict_permissions(destination, directory=False)
    _fsync_file(destination)


def _encrypt_sqlite_snapshot(source: Path, destination: Path, *, key: bytes) -> None:
    if sqlcipher_dbapi is None:
        raise RuntimeError("SQLCipher is not available.")
    destination.unlink(missing_ok=True)
    connection = sqlcipher_dbapi.connect(str(source))
    try:
        connection.execute(
            f"ATTACH DATABASE {_sqlite_literal(str(destination))} AS encrypted "
            f"KEY \"x'{key.hex()}'\""
        )
        connection.execute("SELECT sqlcipher_export('encrypted')")
        connection.execute("DETACH DATABASE encrypted")
        connection.commit()
    finally:
        connection.close()
    _fsync_file(destination)


def _verify_encrypted_sqlite(path: Path, *, key: bytes) -> None:
    if _looks_like_plaintext_sqlite(path):
        raise RuntimeError("The migrated SQLite database is still plaintext.")
    if sqlcipher_dbapi is None:
        raise RuntimeError("SQLCipher is not available.")
    connection = sqlcipher_dbapi.connect(str(path))
    try:
        _apply_sqlcipher_key(connection, key)
        row = connection.execute("PRAGMA integrity_check").fetchone()
        if not row or str(row[0]).lower() != "ok":
            raise RuntimeError("The migrated SQLite database did not pass its integrity check.")
    finally:
        connection.close()


def _migration_backup_path(*, data_dir: Path, source: Path, suffix: str) -> Path:
    backup_root = data_dir / _MIGRATION_BACKUP_DIR
    try:
        if backup_root.resolve().is_relative_to(source.resolve()):
            backup_root = source.parent / ".atlas-migration-backups"
    except (OSError, ValueError):
        pass
    source_fingerprint = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return backup_root / f"{source.name}.{source_fingerprint}.{timestamp}.{uuid.uuid4().hex}{suffix}"


def _remove_sqlite_migration_backup(backup_path: Path) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        _unlink_with_retry(backup_path.with_name(f"{backup_path.name}{suffix}"))
    _unlink_with_retry(backup_path)
    _remove_empty_directory(backup_path.parent)


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except (FileNotFoundError, OSError):
        return
    _fsync_directory(path.parent)


def _sqlite_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _fsync_file(path: Path) -> None:
    # Windows requires a writable descriptor for FlushFileBuffers, which is
    # what Python's os.fsync delegates to there. Opening an existing migration
    # artifact read/write does not alter its contents and keeps the durability
    # guarantee consistent across platforms.
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _restrict_tree_permissions(path: Path) -> None:
    _restrict_permissions(path, directory=True)
    for item in path.rglob("*"):
        _restrict_permissions(item, directory=item.is_dir())


def _restrict_permissions(path: Path, *, directory: bool) -> None:
    if os.name == "nt":
        return
    try:
        path.chmod(0o700 if directory else 0o600)
    except OSError:
        pass


def _restrict_storage_parent(path: Path, *, data_dir: Path) -> None:
    try:
        resolved_path = path.resolve()
        resolved_data_dir = data_dir.resolve()
        if resolved_path == resolved_data_dir or resolved_path.is_relative_to(resolved_data_dir):
            _restrict_permissions(path, directory=True)
    except (OSError, ValueError):
        return


def _write_private_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _restrict_permissions(path.parent, directory=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _restrict_permissions(path, directory=False)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _unlink_with_retry(path: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(6):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


def _rmtree_with_retry(path: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(6):
        try:
            shutil.rmtree(path, ignore_errors=False)
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


def _blob_from_bytes(data: bytes | None) -> tuple[_DataBlob, ctypes.Array[ctypes.c_char]]:
    raw = data or b""
    buffer = ctypes.create_string_buffer(raw, len(raw))
    blob = _DataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    return blob, buffer
