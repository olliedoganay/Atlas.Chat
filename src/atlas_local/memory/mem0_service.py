from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any

from qdrant_client.local import persistence as qdrant_persistence

from ..config import AppConfig
from ..security import (
    build_encrypted_sqlite_module,
    open_application_sqlite,
    prepare_encrypted_qdrant_storage,
)
from .models import MemoryCandidate, MemoryRecord, StoredMemory


# Mem0 writes an identity file as soon as it is imported, even when telemetry
# events are disabled. Keep the import lazy so the active Atlas data directory
# is known before any Mem0 module code can create local state.
Memory: Any | None = None
mem0_storage: Any | None = None
_mem0_setup_module: Any | None = None
_mem0_main_module: Any | None = None
_MEM0_RUNTIME_LOCK = threading.RLock()


def _load_mem0_runtime(data_dir: Path) -> tuple[Any, Any]:
    global Memory, mem0_storage, _mem0_main_module, _mem0_setup_module

    mem0_dir = (data_dir / "mem0").resolve()
    with _MEM0_RUNTIME_LOCK:
        os.environ["MEM0_TELEMETRY"] = "False"
        os.environ["MEM0_DIR"] = str(mem0_dir)

        if Memory is None:
            from mem0 import Memory as imported_memory
            from mem0.memory import main as imported_mem0_main
            from mem0.memory import setup as imported_mem0_setup
            from mem0.memory import storage as imported_mem0_storage

            Memory = imported_memory
            mem0_storage = imported_mem0_storage
            _mem0_main_module = imported_mem0_main
            _mem0_setup_module = imported_mem0_setup

        _secure_mem0_ollama_factories()

        # A source process can construct sequential AppConfig instances (for
        # example in tests or automation). Rebind Mem0's cached path before it
        # reads or writes its identity config so each instance remains inside
        # the active Atlas data directory.
        _mem0_setup_module.mem0_dir = str(mem0_dir)
        _mem0_main_module.mem0_dir = str(mem0_dir)
        _mem0_setup_module.setup_config()
        return Memory, mem0_storage


def _secure_mem0_ollama_factories() -> None:
    from mem0.utils.factory import EmbedderFactory, LlmFactory

    _, ollama_config_class = LlmFactory.provider_to_class["ollama"]
    LlmFactory.provider_to_class["ollama"] = (
        "atlas_local.memory.local_ollama.AtlasOllamaLLM",
        ollama_config_class,
    )
    EmbedderFactory.provider_to_class["ollama"] = (
        "atlas_local.memory.local_ollama.AtlasOllamaEmbedding"
    )


class Mem0Service:
    def __init__(self, config: AppConfig):
        self.config = config
        self._memory: Any | None = None
        self._lifecycle_lock = threading.RLock()
        _, runtime_storage = _load_mem0_runtime(config.data_dir)
        sqlite_module = build_encrypted_sqlite_module(data_dir=config.data_dir)
        runtime_storage.sqlite3 = sqlite_module
        qdrant_persistence.sqlite3 = sqlite_module
        prepare_encrypted_qdrant_storage(config.qdrant_path, data_dir=config.data_dir)
        _reconcile_legacy_qdrant_collections(config)

    def search(self, query: str, *, user_id: str, limit: int) -> list[StoredMemory]:
        memory = self._require_memory()
        filters = {"user_id": user_id}
        existing = memory.get_all(filters=filters, top_k=1)
        if not existing.get("results", []):
            return []
        response = memory.search(
            query,
            filters=filters,
            top_k=limit,
            rerank=False,
        )
        return [StoredMemory.from_dict(item) for item in response.get("results", [])]

    def add(
        self,
        candidate: MemoryCandidate | MemoryRecord,
        *,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text = (
            candidate.to_storage_text()
            if isinstance(candidate, MemoryCandidate)
            else candidate.text
        )
        return self._require_memory().add(
            text,
            user_id=user_id,
            metadata=metadata,
            infer=False,
        )

    def update(
        self, memory_id: str, text: str, *, metadata: dict[str, Any] | None = None
    ) -> None:
        self._require_memory().update(memory_id, text, metadata=metadata)

    def delete(self, memory_id: str, *, user_id: str) -> None:
        memory = self._require_memory()
        item = memory.get(memory_id)
        if not item:
            raise RuntimeError(f"Memory not found: {memory_id}")
        owner = str(item.get("user_id", "") or "")
        if owner != user_id:
            raise RuntimeError("Memory does not belong to this user.")
        memory.delete(memory_id)

    def list(self, *, user_id: str, limit: int = 20) -> list[StoredMemory]:
        response = self._require_memory().get_all(
            filters={"user_id": user_id},
            top_k=limit,
        )
        return [StoredMemory.from_dict(item) for item in response.get("results", [])]

    def delete_all(self, *, user_id: str) -> None:
        self._require_memory().delete_all(user_id=user_id)

    def reset(self) -> None:
        with self._lifecycle_lock:
            self.close()
            _remove_memory_path(self.config.qdrant_path)
            _remove_memory_path(self.config.mem0_history_db)
            self.config.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.config.mem0_history_db.parent.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        with self._lifecycle_lock:
            memory = self._memory
            if memory is None:
                return
            try:
                try:
                    memory.close()
                finally:
                    vector_store = getattr(memory, "vector_store", None)
                    client = getattr(vector_store, "client", None)
                    close = getattr(client, "close", None)
                    if callable(close):
                        close()
            finally:
                self._memory = None

    def _require_memory(self) -> Any:
        with self._lifecycle_lock:
            if self._memory is not None:
                return self._memory
            memory_runtime, _ = _load_mem0_runtime(self.config.data_dir)
            try:
                memory = memory_runtime.from_config(
                    {
                        "vector_store": {
                            "provider": "qdrant",
                            "config": {
                                "collection_name": self.config.mem0_collection,
                                "path": str(self.config.qdrant_path),
                                "embedding_model_dims": self.config.embed_dim,
                                "on_disk": True,
                            },
                        },
                        "llm": {
                            "provider": "ollama",
                            "config": {
                                # Mem0 still constructs an LLM client even though Atlas uses
                                # local extraction and disables reranking. Keep chat model
                                # selection outside memory setup.
                                "model": self.config.embed_model,
                                "temperature": 0.0,
                                "max_tokens": 600,
                                "ollama_base_url": self.config.ollama_url,
                            },
                        },
                        "embedder": {
                            "provider": "ollama",
                            "config": {
                                "model": self.config.embed_model,
                                "ollama_base_url": self.config.ollama_url,
                            },
                        },
                        "history_db_path": str(self.config.mem0_history_db),
                    }
                )
            except RuntimeError as exc:
                if "already accessed by another instance" in str(exc):
                    raise RuntimeError(
                        "Local Qdrant storage is locked by another Atlas process. "
                        "Run one CLI process at a time when using local-path Qdrant, "
                        "or switch to a remote Qdrant server for concurrent access."
                    ) from exc
                raise RuntimeError("Atlas memory service is unavailable.") from exc
            except Exception as exc:
                raise RuntimeError(
                    "Atlas memory service is unavailable. Make sure Ollama is running and the configured models are available."
                ) from exc
            self._memory = memory
            return memory


def _remove_memory_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _reconcile_legacy_qdrant_collections(config: AppConfig) -> None:
    collection_root = config.qdrant_path / "collection"
    if not collection_root.exists():
        return

    target_dir = collection_root / config.mem0_collection
    target_points = _local_collection_points(target_dir)
    meta_path = config.qdrant_path / "meta.json"
    metadata = _load_qdrant_metadata(meta_path)
    sibling_dirs = [
        path
        for path in collection_root.iterdir()
        if path.is_dir() and path != target_dir
    ]

    if target_points is None or target_points > 0:
        _write_qdrant_metadata(meta_path, metadata)
        return

    populated_candidates: list[tuple[Path, int]] = []
    for sibling_dir in sibling_dirs:
        sibling_points = _local_collection_points(sibling_dir)
        if sibling_points is None or sibling_points <= 0:
            continue
        populated_candidates.append((sibling_dir, sibling_points))

    if len(populated_candidates) != 1:
        _write_qdrant_metadata(meta_path, metadata)
        return

    source_dir, source_points = populated_candidates[0]
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    source_dir.replace(target_dir)
    _move_collection_metadata(
        metadata, from_name=source_dir.name, to_name=config.mem0_collection
    )
    target_points = source_points

    _write_qdrant_metadata(meta_path, metadata)


def _local_collection_points(collection_dir: Path) -> int | None:
    if not collection_dir.exists():
        return 0

    database_path = collection_dir / "storage.sqlite"
    if not database_path.exists():
        return 0

    try:
        with closing(
            open_application_sqlite(database_path, data_dir=collection_dir.parents[2])
        ) as conn:
            has_points_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'points'"
            ).fetchone()
            if not has_points_table:
                return 0
            row = conn.execute("SELECT COUNT(*) FROM points").fetchone()
            return int(row[0] or 0) if row else 0
    except sqlite3.Error:
        return None


def _load_qdrant_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}


def _write_qdrant_metadata(path: Path, payload: dict[str, Any]) -> None:
    if not payload:
        return
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _move_collection_metadata(
    payload: dict[str, Any], *, from_name: str, to_name: str
) -> None:
    collections = payload.get("collections")
    if not isinstance(collections, dict):
        return
    if to_name in collections:
        collections.pop(from_name, None)
        return
    if from_name in collections:
        collections[to_name] = collections.pop(from_name)
