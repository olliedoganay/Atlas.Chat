import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from atlas_local.security import (
    application_secret_protection_available,
    get_or_create_storage_key,
    local_secret_storage_label,
    open_application_sqlite,
    prepare_encrypted_qdrant_storage,
    prepare_encrypted_sqlite,
    protect_bytes,
    sqlcipher_enabled,
    unprotect_bytes,
)


class SecurityStorageTests(unittest.TestCase):
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

    def test_prepare_encrypted_sqlite_migrates_data_and_keeps_recovery_copy(self) -> None:
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
                backups = list((data_dir / "migration-backups").glob("*.plaintext.sqlite"))
                self.assertEqual(len(backups), 1)
                if os.name != "nt":
                    self.assertEqual(backups[0].stat().st_mode & 0o077, 0)
                with closing(sqlite3.connect(backups[0])) as conn:
                    self.assertEqual(
                        conn.execute("SELECT value FROM legacy").fetchone()[0],
                        "preserved",
                    )

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
                backups = list((data_dir / "migration-backups").glob("*.plaintext.qdrant"))
                self.assertEqual(len(backups), 1)
                with closing(
                    sqlite3.connect(
                        backups[0] / "collection" / "atlas_local_memory" / "storage.sqlite"
                    )
                ) as conn:
                    self.assertEqual(conn.execute("SELECT id FROM points").fetchone()[0], "point-1")

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

    def test_secret_storage_status_reports_non_windows_keyring_support(self) -> None:
        with (
            patch("atlas_local.security.os.name", "posix"),
            patch("atlas_local.security._non_windows_secret_storage_supported", return_value=True),
        ):
            self.assertEqual(local_secret_storage_label(), "os-keyring")
            self.assertTrue(application_secret_protection_available())


if __name__ == "__main__":
    unittest.main()
