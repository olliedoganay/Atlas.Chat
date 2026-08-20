import base64
import json
import multiprocessing
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from atlas_local.security import (
    _derive_non_windows_encryption_key,
    application_secret_protection_available,
    get_or_create_storage_key,
    local_secret_storage_label,
    open_application_sqlite,
    prepare_encrypted_qdrant_storage,
    prepare_encrypted_sqlite,
    protect_bytes,
    protect_bytes_with_key,
    purge_legacy_migration_backups,
    sqlcipher_enabled,
    unprotect_bytes,
    unprotect_bytes_with_key,
)


class SecurityStorageTests(unittest.TestCase):
    def test_non_windows_encryption_key_derivation_compatibility_vectors(self) -> None:
        master_key = bytes(range(32))

        self.assertEqual(
            _derive_non_windows_encryption_key(master_key, entropy=None).hex(),
            "60e2e7ea7273baa4275c0cea8f2df69b069f6cf273b3e66d22f8831ee413ad85",
        )
        self.assertEqual(
            _derive_non_windows_encryption_key(master_key, entropy=b"profile-key").hex(),
            "60ca3251ca2d56613dbfd9a6f76a170f9140fb013d220ea09859be7ac80768ca",
        )

    def test_storage_key_is_stable_per_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            first = get_or_create_storage_key(data_dir)
            second = get_or_create_storage_key(data_dir)
            self.assertEqual(first, second)
            if os.name != "nt":
                self.assertEqual(
                    (data_dir / "storage.key.json").stat().st_mode & 0o077,
                    0,
                )

    @unittest.skipIf(os.name == "nt", "fork-based process race test")
    def test_storage_key_first_creation_is_safe_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            context = multiprocessing.get_context("fork")
            start = context.Event()
            results = context.Queue()

            def create_key() -> None:
                start.wait()
                try:
                    results.put(("ok", get_or_create_storage_key(data_dir)))
                except Exception as exc:  # pragma: no cover - child diagnostic
                    results.put(("error", str(exc)))

            with (
                patch(
                    "atlas_local.security.protect_bytes",
                    side_effect=lambda data, **_kwargs: data,
                ),
                patch(
                    "atlas_local.security.unprotect_bytes",
                    side_effect=lambda data, **_kwargs: data,
                ),
            ):
                processes = [
                    context.Process(target=create_key)
                    for _ in range(2)
                ]
                for process in processes:
                    process.start()
                start.set()
                process_results = [
                    results.get(timeout=5)
                    for _ in processes
                ]
                for process in processes:
                    process.join(timeout=5)

            self.assertEqual(
                [status for status, _value in process_results],
                ["ok", "ok"],
            )
            self.assertEqual(process_results[0][1], process_results[1][1])
            self.assertTrue(all(process.exitcode == 0 for process in processes))

    def test_invalid_existing_storage_key_is_not_silently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            key_path = data_dir / "storage.key.json"
            key_path.write_text(
                '{"format":"unsupported","wrapped_key":"AAAA"}',
                encoding="utf-8",
            )
            original = key_path.read_bytes()

            with self.assertRaisesRegex(RuntimeError, "will not replace it"):
                get_or_create_storage_key(data_dir)

            self.assertEqual(key_path.read_bytes(), original)

    @unittest.skipIf(os.name == "nt", "legacy raw storage keys are non-Windows")
    def test_plaintext_storage_key_requires_explicit_one_time_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            key_path = data_dir / "storage.key.json"
            legacy_key = b"k" * 32
            key_path.write_text(
                json.dumps(
                    {
                        "format": "atlas-dpapi-storage-key-v1",
                        "wrapped_key": base64.b64encode(legacy_key).decode("ascii"),
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "atlas_local.security._non_windows_secret_storage_supported",
                    return_value=True,
                ),
                patch(
                    "atlas_local.security._get_or_create_non_windows_master_key",
                    return_value=b"\x22" * 32,
                ),
                patch.dict(
                    os.environ,
                    {"ATLAS_ALLOW_LEGACY_PLAINTEXT_MIGRATION": ""},
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid or unavailable"):
                    get_or_create_storage_key(data_dir)

            with (
                patch(
                    "atlas_local.security._non_windows_secret_storage_supported",
                    return_value=True,
                ),
                patch(
                    "atlas_local.security._get_or_create_non_windows_master_key",
                    return_value=b"\x22" * 32,
                ),
                patch.dict(
                    os.environ,
                    {"ATLAS_ALLOW_LEGACY_PLAINTEXT_MIGRATION": "1"},
                ),
            ):
                self.assertEqual(get_or_create_storage_key(data_dir), legacy_key)
                migrated = json.loads(key_path.read_text(encoding="utf-8"))
                wrapped = base64.b64decode(
                    migrated["wrapped_key"],
                    validate=True,
                )
                self.assertNotEqual(wrapped, legacy_key)
                self.assertEqual(
                    unprotect_bytes(wrapped, require_protection=True),
                    legacy_key,
                )

    @unittest.skipUnless(
        application_secret_protection_available(),
        "tamper authentication requires an available OS secret backend",
    )
    def test_tampered_storage_key_has_a_deterministic_fail_closed_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            get_or_create_storage_key(data_dir)
            key_path = data_dir / "storage.key.json"
            payload = json.loads(key_path.read_text(encoding="utf-8"))
            wrapped = bytearray(
                base64.b64decode(payload["wrapped_key"], validate=True)
            )
            wrapped[-1] ^= 1
            payload["wrapped_key"] = base64.b64encode(wrapped).decode("ascii")
            key_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "invalid or unavailable"):
                get_or_create_storage_key(data_dir)

    def test_caller_key_ciphertext_authentication_failure_is_normalized(self) -> None:
        key = b"k" * 32
        encrypted = bytearray(
            protect_bytes_with_key(b"atlas", key=key, aad=b"scope")
        )
        encrypted[-1] ^= 1

        with self.assertRaisesRegex(
            ValueError, "ciphertext authentication failed"
        ):
            unprotect_bytes_with_key(bytes(encrypted), key=key, aad=b"scope")

    def test_windows_dpapi_unprotect_failure_is_normalized(self) -> None:
        windows_error = OSError(87, "The parameter is incorrect.")
        with (
            patch("atlas_local.security.os.name", "nt"),
            patch("atlas_local.security.ctypes.windll", create=True) as windll,
            patch(
                "atlas_local.security.ctypes.WinError",
                return_value=windows_error,
                create=True,
            ),
        ):
            windll.crypt32.CryptUnprotectData.return_value = 0
            with self.assertRaisesRegex(
                RuntimeError,
                "Protected data authentication failed",
            ) as raised:
                unprotect_bytes(b"forged-windows-dpapi-payload")

        self.assertIs(raised.exception.__cause__, windows_error)

    def test_application_sqlite_writes_encrypted_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            db_path = data_dir / "secure.sqlite"
            with closing(open_application_sqlite(db_path, data_dir=data_dir)) as conn:
                conn.execute("CREATE TABLE sample (value TEXT)")
                conn.execute("INSERT INTO sample (value) VALUES ('atlas')")
                conn.commit()

            with closing(open_application_sqlite(db_path, data_dir=data_dir)) as conn:
                row = conn.execute("SELECT value FROM sample").fetchone()

            self.assertEqual(row[0], "atlas")
            header = db_path.read_bytes()[:16]
            if sqlcipher_enabled():
                self.assertNotEqual(header, b"SQLite format 3\x00")

    def test_prepare_encrypted_sqlite_migrates_data_without_plaintext_recovery_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            db_path = data_dir / "legacy.sqlite"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("CREATE TABLE legacy (value TEXT)")
                conn.execute("INSERT INTO legacy (value) VALUES ('preserved')")
                conn.commit()

            prepare_encrypted_sqlite(db_path, data_dir=data_dir)

            if sqlcipher_enabled():
                self.assertTrue(db_path.exists())
                self.assertNotEqual(db_path.read_bytes()[:16], b"SQLite format 3\x00")
                with closing(open_application_sqlite(db_path, data_dir=data_dir)) as conn:
                    self.assertEqual(
                        conn.execute("SELECT value FROM legacy").fetchone()[0],
                        "preserved",
                    )
                self.assertFalse((data_dir / "migration-backups").exists())

    def test_prepare_encrypted_sqlite_restores_original_when_migration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            db_path = data_dir / "legacy.sqlite"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("CREATE TABLE legacy (value TEXT)")
                conn.execute("INSERT INTO legacy (value) VALUES ('still here')")
                conn.commit()

            with patch(
                "atlas_local.security._encrypt_sqlite_snapshot",
                side_effect=RuntimeError("simulated export failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "original database was left in place"):
                    prepare_encrypted_sqlite(db_path, data_dir=data_dir)

            self.assertEqual(db_path.read_bytes()[:16], b"SQLite format 3\x00")
            with closing(sqlite3.connect(db_path)) as conn:
                self.assertEqual(conn.execute("SELECT value FROM legacy").fetchone()[0], "still here")
            if sqlcipher_enabled():
                self.assertEqual(
                    len(list((data_dir / "migration-backups").glob("*.plaintext.sqlite"))),
                    1,
                )

    def test_prepare_encrypted_qdrant_storage_migrates_without_erasing_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            qdrant_dir = data_dir / "qdrant" / "collection" / "atlas_local_memory"
            qdrant_dir.mkdir(parents=True, exist_ok=True)
            marker_path = data_dir / "qdrant" / "meta.json"
            marker_path.write_text('{"preserved": true}', encoding="utf-8")
            db_path = qdrant_dir / "storage.sqlite"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("CREATE TABLE points (id TEXT PRIMARY KEY)")
                conn.execute("INSERT INTO points (id) VALUES ('point-1')")
                conn.commit()

            prepare_encrypted_qdrant_storage(data_dir / "qdrant", data_dir=data_dir)

            if sqlcipher_enabled():
                self.assertTrue(db_path.exists())
                self.assertNotEqual(db_path.read_bytes()[:16], b"SQLite format 3\x00")
                self.assertEqual(marker_path.read_text(encoding="utf-8"), '{"preserved": true}')
                with closing(open_application_sqlite(db_path, data_dir=data_dir)) as conn:
                    self.assertEqual(conn.execute("SELECT id FROM points").fetchone()[0], "point-1")
                self.assertFalse((data_dir / "migration-backups").exists())

    def test_prepare_encrypted_qdrant_storage_rolls_back_failed_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            qdrant_path = data_dir / "qdrant"
            qdrant_dir = qdrant_path / "collection" / "atlas_local_memory"
            qdrant_dir.mkdir(parents=True, exist_ok=True)
            db_path = qdrant_dir / "storage.sqlite"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("CREATE TABLE points (id TEXT PRIMARY KEY)")
                conn.execute("INSERT INTO points (id) VALUES ('point-1')")
                conn.commit()

            with patch(
                "atlas_local.security._migrate_plaintext_sqlite",
                side_effect=RuntimeError("simulated staged migration failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "original store was restored"):
                    prepare_encrypted_qdrant_storage(qdrant_path, data_dir=data_dir)

            self.assertTrue(db_path.exists())
            self.assertEqual(db_path.read_bytes()[:16], b"SQLite format 3\x00")
            with closing(sqlite3.connect(db_path)) as conn:
                self.assertEqual(conn.execute("SELECT id FROM points").fetchone()[0], "point-1")
            if sqlcipher_enabled():
                backups = list(
                    (data_dir / "migration-backups").glob("*.plaintext.qdrant")
                )
                self.assertEqual(len(backups), 1)
                self.assertTrue(
                    (
                        backups[0]
                        / "collection"
                        / "atlas_local_memory"
                        / "storage.sqlite"
                    ).exists()
                )

    def test_purge_legacy_migration_backups_removes_known_roots_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            data_dir = project_root / "data"
            default_backup = data_dir / "migration-backups"
            fallback_backup = data_dir / ".atlas-migration-backups"
            unrelated = data_dir / "keep"
            default_backup.mkdir(parents=True)
            fallback_backup.mkdir()
            unrelated.mkdir()
            (default_backup / "legacy.sqlite").write_text("secret", encoding="utf-8")
            (fallback_backup / "legacy.qdrant").write_text("secret", encoding="utf-8")
            (unrelated / "marker").write_text("preserved", encoding="utf-8")

            purge_legacy_migration_backups(
                data_dir=data_dir,
                storage_paths=(data_dir / "qdrant",),
            )

            self.assertFalse(default_backup.exists())
            self.assertFalse(fallback_backup.exists())
            self.assertEqual(
                (unrelated / "marker").read_text(encoding="utf-8"),
                "preserved",
            )

    def test_non_windows_secret_storage_encrypts_and_decrypts(self) -> None:
        with (
            patch("atlas_local.security.os.name", "posix"),
            patch("atlas_local.security._non_windows_secret_storage_supported", return_value=True),
            patch("atlas_local.security._get_or_create_non_windows_master_key", return_value=b"\x11" * 32),
        ):
            encrypted = protect_bytes(b"atlas-secret", entropy=b"profile-key")
            self.assertNotEqual(encrypted, b"atlas-secret")
            decrypted = unprotect_bytes(encrypted, entropy=b"profile-key")
            self.assertEqual(decrypted, b"atlas-secret")

    def test_non_windows_unprotect_keeps_legacy_plaintext_bytes(self) -> None:
        with patch("atlas_local.security.os.name", "posix"):
            self.assertEqual(unprotect_bytes(b"legacy-bytes"), b"legacy-bytes")

    def test_strict_non_windows_protection_rejects_plaintext_and_missing_keyring(
        self,
    ) -> None:
        with (
            patch("atlas_local.security.os.name", "posix"),
            patch(
                "atlas_local.security._non_windows_secret_storage_supported",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "secure OS key storage"):
                protect_bytes(b"secret", require_protection=True)
            with self.assertRaisesRegex(RuntimeError, "unprotected legacy format"):
                unprotect_bytes(b"legacy-bytes", require_protection=True)

    def test_secret_storage_status_reports_non_windows_keyring_support(self) -> None:
        with (
            patch("atlas_local.security.os.name", "posix"),
            patch("atlas_local.security._non_windows_secret_storage_supported", return_value=True),
        ):
            self.assertEqual(local_secret_storage_label(), "os-keyring")
            self.assertTrue(application_secret_protection_available())


if __name__ == "__main__":
    unittest.main()
