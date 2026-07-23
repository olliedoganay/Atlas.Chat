import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from qdrant_client.http import models
from qdrant_client.local.persistence import CollectionPersistence

from atlas_local.config import load_config
from atlas_local.memory.mem0_service import (
    Mem0Service,
    _local_collection_points,
    _reconcile_legacy_qdrant_collections,
)
from atlas_local.security import open_application_sqlite


class Mem0ServiceCollectionMigrationTests(unittest.TestCase):
    def test_import_does_not_create_mem0_state_outside_atlas_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home_dir = root / "home"
            data_dir = root / "data"
            home_dir.mkdir()
            env = dict(os.environ)
            env.pop("MEM0_DIR", None)
            env["HOME"] = str(home_dir)
            env["ATLAS_DATA_DIR"] = str(data_dir)
            source_dir = Path(__file__).resolve().parents[1] / "src"
            existing_python_path = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                f"{source_dir}{os.pathsep}{existing_python_path}"
                if existing_python_path
                else str(source_dir)
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from atlas_local.memory.mem0_service import Mem0Service",
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((home_dir / ".mem0").exists())
            self.assertFalse((data_dir / "mem0").exists())

    def test_mem0_telemetry_is_disabled_for_local_first_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(project_root=Path(tmp), env={})
            service = Mem0Service(config)
            from mem0.memory.telemetry import MEM0_TELEMETRY

            self.assertFalse(MEM0_TELEMETRY)
            self.assertTrue((config.data_dir / "mem0" / "config.json").exists())
            service.close()

    def test_legacy_collection_replaces_empty_current_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(project_root=Path(tmp), env={})
            collection_root = config.qdrant_path / "collection"
            legacy_dir = collection_root / "legacy_memory"
            current_dir = collection_root / config.mem0_collection
            self._create_collection(legacy_dir, point_count=3)
            self._create_collection(current_dir, point_count=0)
            self._write_meta(
                config.qdrant_path / "meta.json",
                {
                    "collections": {
                        "legacy_memory": {"vectors": {"size": 768}},
                        config.mem0_collection: {"vectors": {"size": 768}},
                    }
                },
            )

            _reconcile_legacy_qdrant_collections(config)

            self.assertFalse(legacy_dir.exists())
            self.assertTrue(current_dir.exists())
            self.assertEqual(_local_collection_points(current_dir), 3)
            metadata = json.loads(
                (config.qdrant_path / "meta.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("legacy_memory", metadata.get("collections", {}))
            self.assertIn(config.mem0_collection, metadata.get("collections", {}))

    def test_multiple_populated_legacy_collections_do_not_move(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(project_root=Path(tmp), env={})
            collection_root = config.qdrant_path / "collection"
            legacy_dir = collection_root / "legacy_memory"
            secondary_legacy_dir = collection_root / "legacy_memory_two"
            current_dir = collection_root / config.mem0_collection
            self._create_collection(legacy_dir, point_count=2)
            self._create_collection(secondary_legacy_dir, point_count=1)
            self._create_collection(current_dir, point_count=0)
            self._write_meta(
                config.qdrant_path / "meta.json",
                {
                    "collections": {
                        "legacy_memory": {"vectors": {"size": 768}},
                        "legacy_memory_two": {"vectors": {"size": 768}},
                        config.mem0_collection: {"vectors": {"size": 768}},
                    }
                },
            )

            _reconcile_legacy_qdrant_collections(config)

            self.assertTrue(legacy_dir.exists())
            self.assertTrue(secondary_legacy_dir.exists())
            self.assertEqual(_local_collection_points(current_dir), 0)
            metadata = json.loads(
                (config.qdrant_path / "meta.json").read_text(encoding="utf-8")
            )
            self.assertIn("legacy_memory", metadata.get("collections", {}))
            self.assertIn("legacy_memory_two", metadata.get("collections", {}))
            self.assertIn(config.mem0_collection, metadata.get("collections", {}))

    def test_constructor_does_not_require_ollama(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(project_root=Path(tmp), env={})
            service = Mem0Service(config)
            with patch(
                "atlas_local.memory.mem0_service.Memory.from_config",
                side_effect=ConnectionError("offline"),
            ) as factory:
                self.assertIsNone(service._memory)

            factory.assert_not_called()

    def test_memory_access_reports_ollama_unavailability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(project_root=Path(tmp), env={})
            service = Mem0Service(config)
            with patch(
                "atlas_local.memory.mem0_service.Memory.from_config",
                side_effect=ConnectionError("offline"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "memory service is unavailable"
                ):
                    service.list(user_id="research_user", limit=10)

    def test_search_and_list_scope_mem0_v2_queries_with_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(project_root=Path(tmp), env={})
            service = Mem0Service(config)
            calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

            class FakeMemory:
                def search(self, *args, **kwargs):
                    calls.append(("search", args, kwargs))
                    return {"results": []}

                def get_all(self, *args, **kwargs):
                    calls.append(("get_all", args, kwargs))
                    return {"results": []}

            service._memory = FakeMemory()  # type: ignore[assignment]

            self.assertEqual(
                service.search("local memory", user_id="research_user", limit=7),
                [],
            )
            self.assertEqual(service.list(user_id="research_user", limit=11), [])
            self.assertEqual(
                calls,
                [
                    (
                        "search",
                        ("local memory",),
                        {
                            "filters": {"user_id": "research_user"},
                            "top_k": 7,
                            "rerank": False,
                        },
                    ),
                    (
                        "get_all",
                        (),
                        {
                            "filters": {"user_id": "research_user"},
                            "top_k": 11,
                        },
                    ),
                ],
            )

    def test_delete_all_initializes_memory_before_deleting_user_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(project_root=Path(tmp), env={})
            service = Mem0Service(config)
            deleted_users: list[str] = []

            class FakeMemory:
                def delete_all(self, *, user_id: str) -> None:
                    deleted_users.append(user_id)

            with patch(
                "atlas_local.memory.mem0_service.Memory.from_config",
                return_value=FakeMemory(),
            ) as factory:
                service.delete_all(user_id="research_user")

            factory.assert_called_once()
            self.assertEqual(deleted_users, ["research_user"])

    def test_delete_requires_memory_owner_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(project_root=Path(tmp), env={})
            service = Mem0Service(config)
            deleted: list[str] = []

            class FakeMemory:
                def get(self, memory_id: str):
                    return {
                        "id": memory_id,
                        "memory": "private note",
                        "user_id": "owner",
                    }

                def delete(self, memory_id: str) -> None:
                    deleted.append(memory_id)

            with patch(
                "atlas_local.memory.mem0_service.Memory.from_config",
                return_value=FakeMemory(),
            ):
                with self.assertRaisesRegex(RuntimeError, "does not belong"):
                    service.delete("memory-1", user_id="other")
                service.delete("memory-1", user_id="owner")

            self.assertEqual(deleted, ["memory-1"])

    def test_reset_removes_memory_storage_without_initializing_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(project_root=Path(tmp), env={})
            service = Mem0Service(config)
            collection_file = (
                config.qdrant_path
                / "collection"
                / config.mem0_collection
                / "storage.sqlite"
            )
            collection_file.parent.mkdir(parents=True, exist_ok=True)
            collection_file.write_text("vector data", encoding="utf-8")
            config.mem0_history_db.write_text("history data", encoding="utf-8")

            with patch("atlas_local.memory.mem0_service.Memory.from_config") as factory:
                service.reset()

            factory.assert_not_called()
            self.assertTrue(config.qdrant_path.exists())
            self.assertEqual(list(config.qdrant_path.iterdir()), [])
            self.assertFalse(config.mem0_history_db.exists())
            self.assertTrue(config.mem0_history_db.parent.exists())

    def test_constructor_patches_qdrant_local_storage_to_encrypted_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(project_root=Path(tmp), env={})
            service = Mem0Service(config)
            collection_dir = config.qdrant_path / "collection" / config.mem0_collection
            persistence = CollectionPersistence(str(collection_dir))
            try:
                persistence.persist(
                    models.PointStruct(
                        id=1, vector=[1.0, 2.0], payload={"source": "test"}
                    )
                )
            finally:
                persistence.close()
                service.close()

            header = (collection_dir / "storage.sqlite").read_bytes()[:16]
            self.assertNotEqual(header, b"SQLite format 3\x00")

    @staticmethod
    def _create_collection(path: Path, *, point_count: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with closing(
            open_application_sqlite(path / "storage.sqlite", data_dir=path.parents[2])
        ) as conn:
            conn.execute("CREATE TABLE points (id TEXT PRIMARY KEY)")
            for index in range(point_count):
                conn.execute("INSERT INTO points (id) VALUES (?)", (str(index),))
            conn.commit()

    @staticmethod
    def _write_meta(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
