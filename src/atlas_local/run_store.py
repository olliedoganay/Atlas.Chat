from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from .config import AppConfig
from .run_contract import (
    RunEvent,
    RunTraceItem,
    make_run_event,
    make_trace_item,
    now_timestamp,
)
from .security import (
    application_secret_protection_available,
    open_application_sqlite,
    protect_bytes,
    protect_bytes_with_key,
    unprotect_bytes,
    unprotect_bytes_with_key,
)
from .session import scoped_thread_id

ACTIVE_RUN_STATUSES = {"queued", "running", "cancelling"}
PASSWORDLESS_PROTECTION = "passwordless"
PASSWORD_PROTECTED = "password"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_PASSWORD_KEY_LENGTH = 32
_PASSWORD_MATERIAL_LENGTH = _PASSWORD_KEY_LENGTH * 2
_INDEX_FORMAT = "atlas-dpapi-index-v1"
_RUN_FORMAT_V1 = "atlas-dpapi-run-v1"
_RUN_FORMAT_V2 = "atlas-profile-run-v2"
_RUN_FORMAT_LEGACY_PLAINTEXT = "atlas-plaintext-run-legacy"
_PASSWORD_KDF_V2 = "atlas-scrypt-split-v2"
_SAFE_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SEARCH_INDEX_LIMIT = 500
_PERSISTED_STREAM_CHUNK_BYTES = 4096
_MAX_PERSISTED_STREAM_BYTES = 2_000_000
_MAX_PERSISTED_RUN_EVENTS = 8192
_TOKEN_EVENT_FLUSH_SECONDS = 0.2
_STREAM_EVENT_TYPES = {"token", "thinking_token"}


class RunStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.runs_dir = config.data_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            self.runs_dir.chmod(0o700)
        self._index_path = self.runs_dir / "index.json"
        self._search_index_path = self.runs_dir / "search.sqlite"
        self._lock = threading.RLock()
        self._user_keys: dict[str, bytearray] = {}
        self._pending_run_events: dict[str, list[RunEvent]] = {}
        self._pending_run_started_at: dict[str, float] = {}
        self._pending_run_stream_bytes: dict[str, int] = {}
        self._run_stream_bytes: dict[str, int] = {}
        self._next_event_sequences: dict[str, int] = {}
        if not self._index_path.exists():
            self._write_index({"threads": {}, "runs": {}, "users": {}})

    def create_run(
        self,
        *,
        mode: str,
        user_id: str,
        thread_id: str,
        chat_model: str,
        temperature: float | None,
        prompt: str,
        thread_title: str | None = None,
        status: str = "running",
        touch_thread: bool = True,
        history_after_message_count: int = 0,
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        with self._lock:
            index = self._read_index()
            existing_thread = index.get("threads", {}).get(
                self._thread_key(user_id, thread_id), {}
            )
            resolved_title = (
                (thread_title or "").strip()
                or existing_thread.get("title")
                or thread_id
            )
            artifact = {
                "run_id": run_id,
                "mode": mode,
                "user_id": user_id,
                "thread_id": thread_id,
                "thread_title": resolved_title,
                "chat_model": chat_model,
                "temperature": temperature,
                "prompt": prompt,
                "status": status,
                "history_after_message_count": max(
                    0, int(history_after_message_count or 0)
                ),
                "started_at": now_timestamp(),
                "completed_at": None,
                "answer": "",
                "reasoning": "",
                "events": [],
                "trace_items": [],
                "error": None,
            }
            if user_id not in index.get("users", {}):
                index["users"][user_id] = self._build_user_record(
                    user_id,
                    updated_at=artifact["started_at"],
                )
                self._cache_user_key_from_record(user_id, index["users"][user_id])
            self._write_run_file(run_id, artifact)
            index["runs"][run_id] = {
                "run_id": run_id,
                "mode": mode,
                "user_id": user_id,
                "thread_id": thread_id,
                "thread_title": resolved_title,
                "chat_model": chat_model,
                "temperature": temperature,
                "status": status,
                "history_after_message_count": artifact["history_after_message_count"],
                "started_at": artifact["started_at"],
                "artifact_format": _RUN_FORMAT_V2,
            }
            if touch_thread:
                index["threads"][self._thread_key(user_id, thread_id)] = {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "title": resolved_title,
                    "chat_model": chat_model,
                    "temperature": temperature,
                    "last_mode": mode,
                    "updated_at": artifact["started_at"],
                    "last_prompt": prompt[:120],
                    "last_run_id": run_id,
                }
                existing_user = index.get("users", {}).get(user_id)
                index["users"][user_id] = self._build_user_record(
                    user_id,
                    updated_at=artifact["started_at"],
                    existing=existing_user,
                )
            self._write_index(index)
            self._index_run_for_search(artifact)
        return artifact

    def append_event(
        self, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> RunEvent:
        with self._lock:
            if not self._run_path(run_id).exists():
                raise RuntimeError(f"Run not found: {run_id}")
            if (
                run_id not in self._next_event_sequences
                or run_id not in self._run_stream_bytes
            ):
                artifact = self._read_run_file(run_id)
                self._run_stream_bytes[run_id] = _artifact_stream_byte_count(
                    artifact
                )
            stream_text = _stream_event_text(event_type, payload)
            if stream_text:
                stream_bytes = len(stream_text.encode("utf-8"))
                current_stream_bytes = self._run_stream_bytes.get(run_id, 0)
                if (
                    current_stream_bytes + stream_bytes
                    > _MAX_PERSISTED_STREAM_BYTES
                ):
                    raise RuntimeError(
                        "Run stream exceeded Atlas's persisted output size limit."
                    )
            sequence = self._next_event_sequences.get(run_id, 1)
            event = make_run_event(
                event_type,
                payload,
                sequence=sequence,
            )
            self._next_event_sequences[run_id] = sequence + 1
            pending = self._pending_run_events.setdefault(run_id, [])
            pending.append(event)
            self._pending_run_started_at.setdefault(run_id, time.monotonic())
            if stream_text:
                self._run_stream_bytes[run_id] = (
                    self._run_stream_bytes.get(run_id, 0) + stream_bytes
                )
                self._pending_run_stream_bytes[run_id] = (
                    self._pending_run_stream_bytes.get(run_id, 0)
                    + stream_bytes
                )
            should_flush = (
                event_type not in _STREAM_EVENT_TYPES
                or self._pending_run_stream_bytes.get(run_id, 0)
                >= _PERSISTED_STREAM_CHUNK_BYTES
                or time.monotonic() - self._pending_run_started_at[run_id]
                >= _TOKEN_EVENT_FLUSH_SECONDS
            )
            if should_flush:
                self._flush_pending_run_locked(run_id)
        return event

    def append_trace_item(self, run_id: str, item: dict[str, Any]) -> RunTraceItem:
        enriched = make_trace_item(item)
        with self._lock:
            self._flush_pending_run_locked(run_id)
            artifact = self.get_run(run_id)
            artifact["trace_items"].append(enriched)
            self._write_run_file(run_id, artifact)
        return enriched

    def complete_run(
        self,
        run_id: str,
        *,
        answer: str,
        terminal_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact, _event = self.complete_run_with_event(
            run_id,
            answer=answer,
            terminal_payload=terminal_payload,
        )
        return artifact

    def complete_run_with_event(
        self,
        run_id: str,
        *,
        answer: str,
        terminal_payload: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], RunEvent]:
        payload = {
            "answer": answer,
            **(terminal_payload or {}),
        }
        return self._finish_run_with_event(
            run_id,
            status="completed",
            answer=answer,
            error_message=None,
            event_type="run_completed",
            event_payload=payload,
        )

    def fail_run(self, run_id: str, *, error: str) -> dict[str, Any]:
        artifact, _event = self.fail_run_with_event(run_id, error=error)
        return artifact

    def fail_run_with_event(
        self,
        run_id: str,
        *,
        error: str,
    ) -> tuple[dict[str, Any], RunEvent]:
        return self._finish_run_with_event(
            run_id,
            status="failed",
            answer=None,
            error_message=error,
            event_type="run_failed",
            event_payload={"error": error},
        )

    def _finish_run_with_event(
        self,
        run_id: str,
        *,
        status: str,
        answer: str | None,
        error_message: str | None,
        event_type: str,
        event_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], RunEvent]:
        with self._lock:
            self._flush_pending_run_locked(run_id)
            artifact = self.get_run(run_id)
            existing_terminal = next(
                (
                    event
                    for event in reversed(artifact.get("events", []))
                    if event.get("type") in {"run_completed", "run_failed"}
                ),
                None,
            )
            if existing_terminal is not None:
                return artifact, existing_terminal

            existing_status = str(artifact.get("status", "") or "")
            if existing_status in {"completed", "failed"}:
                status = existing_status
                if status == "completed":
                    event_type = "run_completed"
                    event_payload = {
                        "answer": str(artifact.get("answer", "") or ""),
                    }
                else:
                    event_type = "run_failed"
                    event_payload = {
                        "error": str(artifact.get("error", "") or ""),
                    }
            elif existing_status == "cancelling" and status == "completed":
                status = "failed"
                answer = None
                error_message = "Run stopped by user."
                event_type = "run_failed"
                event_payload = {"error": error_message}
            timestamp = str(artifact.get("completed_at", "") or "") or now_timestamp()
            artifact["status"] = status
            artifact["completed_at"] = timestamp
            if status == "completed" and answer is not None:
                artifact["answer"] = answer
            if status == "failed" and error_message is not None:
                artifact["error"] = error_message

            sequence = self._next_event_sequences.get(
                run_id,
                len(artifact.get("events", [])) + 1,
            )
            terminal_event = make_run_event(
                event_type,
                event_payload,
                timestamp=timestamp,
                sequence=sequence,
            )
            artifact.setdefault("events", []).append(terminal_event)
            self._next_event_sequences[run_id] = sequence + 1
            self._write_run_file(run_id, artifact)
            index = self._read_index()
            if run_id in index["runs"]:
                index["runs"][run_id]["status"] = status
                index["runs"][run_id]["completed_at"] = artifact["completed_at"]
                index["runs"][run_id]["thread_title"] = artifact.get(
                    "thread_title", artifact.get("thread_id", "")
                )
            self._write_index(index)
            self._index_run_for_search(artifact)
        return artifact, terminal_event

    def mark_run_running(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            self._flush_pending_run_locked(run_id)
            artifact = self.get_run(run_id)
            artifact["status"] = "running"
            self._write_run_file(run_id, artifact)
            index = self._read_index()
            if run_id in index["runs"]:
                index["runs"][run_id]["status"] = "running"
            self._write_index(index)
        return artifact

    def update_run_history_after_message_count(
        self,
        run_id: str,
        *,
        history_after_message_count: int,
    ) -> dict[str, Any]:
        """Move a queued chat run's search positions to its execution snapshot."""

        resolved_count = max(0, int(history_after_message_count or 0))
        with self._lock:
            self._flush_pending_run_locked(run_id)
            artifact = self.get_run(run_id)
            if artifact.get("history_after_message_count") == resolved_count:
                return artifact

            artifact["history_after_message_count"] = resolved_count
            self._write_run_file(run_id, artifact)
            index = self._read_index()
            indexed_run = index.get("runs", {}).get(run_id)
            if isinstance(indexed_run, dict):
                indexed_run["history_after_message_count"] = resolved_count
                self._write_index(index)
            self._index_run_for_search(artifact)
        return artifact

    def mark_run_cancelling(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            self._flush_pending_run_locked(run_id)
            artifact = self.get_run(run_id)
            if artifact.get("status") not in {"completed", "failed"}:
                artifact["status"] = "cancelling"
                self._write_run_file(run_id, artifact)
            index = self._read_index()
            if run_id in index["runs"] and index["runs"][run_id].get("status") not in {
                "completed",
                "failed",
            }:
                index["runs"][run_id]["status"] = "cancelling"
            self._write_index(index)
        return artifact

    def mark_run_cancelling_with_event(
        self,
        run_id: str,
    ) -> tuple[dict[str, Any], RunEvent | None]:
        """Atomically claim cancellation and persist its stopping event."""
        with self._lock:
            self._flush_pending_run_locked(run_id)
            artifact = self.get_run(run_id)
            if artifact.get("status") in {"completed", "failed"}:
                return artifact, None
            existing_stopping = next(
                (
                    event
                    for event in reversed(artifact.get("events", []))
                    if event.get("type") == "stage_changed"
                    and event.get("payload", {}).get("stage") == "stopping"
                ),
                None,
            )
            artifact["status"] = "cancelling"
            stopping_event: RunEvent | None = existing_stopping
            if stopping_event is None:
                sequence = self._next_event_sequences.get(
                    run_id,
                    len(artifact.get("events", [])) + 1,
                )
                stopping_event = make_run_event(
                    "stage_changed",
                    {"stage": "stopping"},
                    sequence=sequence,
                )
                artifact.setdefault("events", []).append(stopping_event)
                self._next_event_sequences[run_id] = sequence + 1
            self._write_run_file(run_id, artifact)
            index = self._read_index()
            if run_id in index["runs"] and index["runs"][run_id].get(
                "status"
            ) not in {"completed", "failed"}:
                index["runs"][run_id]["status"] = "cancelling"
            self._write_index(index)
        return artifact, stopping_event

    def list_incomplete_runs(
        self,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            index = self._read_index()
            items: list[dict[str, Any]] = []
            for run_id, item in index.get("runs", {}).items():
                if item.get("status") not in ACTIVE_RUN_STATUSES:
                    continue
                if user_id is not None and item.get("user_id") != user_id:
                    continue
                copied = dict(item)
                copied.setdefault("run_id", run_id)
                items.append(copied)
            return items

    def fail_incomplete_runs(
        self,
        *,
        error: str,
        user_id: str | None = None,
        run_ids: set[str] | None = None,
        skip_locked: bool = False,
    ) -> list[str]:
        recovered: list[str] = []
        with self._lock:
            index = self._read_index()
            for run_id, item in index.get("runs", {}).items():
                if item.get("status") not in ACTIVE_RUN_STATUSES:
                    continue
                if user_id is not None and item.get("user_id") != user_id:
                    continue
                if run_ids is not None and run_id not in run_ids:
                    continue
                item_user_id = str(item.get("user_id", "") or "")
                user = index.get("users", {}).get(item_user_id, {})
                if (
                    skip_locked
                    and user.get("protection") == PASSWORD_PROTECTED
                    and item_user_id not in self._user_keys
                ):
                    continue
                timestamp = now_timestamp()
                try:
                    path = self._run_path(run_id)
                except RuntimeError:
                    self._mark_unreadable_run_recovered(
                        run_id=run_id,
                        item=item,
                        timestamp=timestamp,
                        error=error,
                        path=None,
                    )
                    recovered.append(run_id)
                    continue
                if path.exists():
                    try:
                        artifact = self.get_run(run_id)
                        events = artifact.get("events", [])
                        if not isinstance(events, list):
                            raise RuntimeError(
                                "The run artifact events payload is invalid."
                            )
                        terminal_event = next(
                            (
                                event
                                for event in reversed(events)
                                if isinstance(event, dict)
                                and event.get("type")
                                in {"run_completed", "run_failed"}
                            ),
                            None,
                        )
                        artifact_status = str(artifact.get("status", "") or "")
                        terminal_artifact = terminal_event is not None or artifact_status in {
                            "completed",
                            "failed",
                        }
                        if terminal_event is not None:
                            artifact_status = (
                                "completed"
                                if terminal_event.get("type") == "run_completed"
                                else "failed"
                            )
                        if terminal_artifact and artifact_status not in {
                            "completed",
                            "failed",
                        }:
                            raise RuntimeError(
                                f"Run {run_id} has an invalid terminal state."
                            )
                    except Exception:
                        self._mark_unreadable_run_recovered(
                            run_id=run_id,
                            item=item,
                            timestamp=timestamp,
                            error=error,
                            path=path,
                        )
                        recovered.append(run_id)
                        continue

                    if terminal_artifact:
                        artifact["status"] = artifact_status
                        artifact["completed_at"] = (
                            str(artifact.get("completed_at", "") or "")
                            or (
                                str(terminal_event.get("timestamp", "") or "")
                                if terminal_event is not None
                                else ""
                            )
                            or timestamp
                        )
                        if terminal_event is None:
                            if artifact_status == "completed":
                                event_type = "run_completed"
                                event_payload = {
                                    "answer": str(artifact.get("answer", "") or "")
                                }
                            else:
                                event_type = "run_failed"
                                event_payload = {
                                    "error": str(artifact.get("error", "") or "")
                                }
                            artifact.setdefault("events", []).append(
                                make_run_event(
                                    event_type,
                                    event_payload,
                                    timestamp=artifact["completed_at"],
                                )
                            )
                        self._write_run_file(run_id, artifact)
                        item["status"] = artifact_status
                        item["completed_at"] = artifact["completed_at"]
                        item["thread_title"] = artifact.get(
                            "thread_title", artifact.get("thread_id", "")
                        )
                        item["artifact_format"] = _RUN_FORMAT_V2
                        self._index_run_for_search(artifact)
                        continue
                else:
                    artifact = self._recovery_artifact_from_index(
                        run_id=run_id,
                        item=item,
                        timestamp=timestamp,
                    )
                artifact["status"] = "failed"
                artifact["completed_at"] = timestamp
                artifact["error"] = error
                artifact.setdefault("events", [])
                artifact["events"].append(
                    make_run_event("run_failed", {"error": error}, timestamp=timestamp)
                )
                self._write_run_file(run_id, artifact)
                item["status"] = "failed"
                item["completed_at"] = timestamp
                item["artifact_format"] = _RUN_FORMAT_V2
                recovered.append(run_id)
            self._write_index(index)
        return recovered

    def _mark_unreadable_run_recovered(
        self,
        *,
        run_id: str,
        item: dict[str, Any],
        timestamp: str,
        error: str,
        path: Path | None,
    ) -> None:
        quarantine_path = ""
        if path is not None and path.exists():
            try:
                quarantine_dir = self.runs_dir / "quarantine"
                quarantine_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                if os.name != "nt":
                    quarantine_dir.chmod(0o700)
                destination = quarantine_dir / f"{run_id}.{uuid.uuid4().hex}.json"
                os.replace(path, destination)
                if os.name != "nt":
                    destination.chmod(0o600)
                quarantine_path = str(destination.relative_to(self.runs_dir))
            except OSError:
                quarantine_path = ""

        recovery_error = (
            f"{error} The stored run artifact was unreadable and was quarantined."
            if quarantine_path
            else f"{error} The stored run artifact was unreadable."
        )
        recovery = {
            "reason": "unreadable_run_artifact",
            "quarantined_artifact": quarantine_path,
        }
        self._discard_pending_run(run_id, forget_sequence=True)
        item["status"] = "failed"
        item["completed_at"] = timestamp
        item["artifact_format"] = _RUN_FORMAT_V2
        item["recovery"] = recovery

        if not quarantine_path:
            return
        artifact = self._recovery_artifact_from_index(
            run_id=run_id,
            item=item,
            timestamp=timestamp,
        )
        artifact["status"] = "failed"
        artifact["completed_at"] = timestamp
        artifact["error"] = recovery_error
        artifact["recovery"] = recovery
        artifact["events"].append(
            make_run_event(
                "run_failed",
                {"error": recovery_error},
                timestamp=timestamp,
            )
        )
        try:
            self._delete_search_entries(run_id=run_id)
        except Exception:
            # A stale optional search index must not make startup recovery fail.
            pass
        try:
            self._write_run_file(run_id, artifact)
        except Exception:
            # Recovery is deliberately per artifact. The encrypted index still
            # records this run as failed and the original remains quarantined.
            return

    @staticmethod
    def _recovery_artifact_from_index(
        *,
        run_id: str,
        item: dict[str, Any],
        timestamp: str,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "mode": item.get("mode", "chat"),
            "user_id": item.get("user_id", ""),
            "thread_id": item.get("thread_id", ""),
            "thread_title": item.get(
                "thread_title", item.get("thread_id", "")
            ),
            "chat_model": item.get("chat_model", ""),
            "temperature": item.get("temperature"),
            "prompt": "",
            "status": item.get("status", "failed"),
            "history_after_message_count": int(
                item.get("history_after_message_count", 0) or 0
            ),
            "started_at": item.get("started_at", timestamp),
            "completed_at": None,
            "answer": "",
            "reasoning": "",
            "events": [],
            "trace_items": [],
            "error": None,
        }

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            artifact = self._read_run_file(run_id)
            return self._merge_pending_events(run_id, artifact)

    def get_run_metadata(self, run_id: str) -> dict[str, Any]:
        """Return indexed ownership/status metadata without decrypting the artifact."""

        with self._lock:
            item = self._read_index().get("runs", {}).get(run_id)
            if not isinstance(item, dict):
                raise RuntimeError(f"Run not found: {run_id}")
            copied = dict(item)
            copied.setdefault("run_id", run_id)
            return copied

    def flush_pending_events(self) -> None:
        with self._lock:
            for run_id in list(self._pending_run_events):
                self._flush_pending_run_locked(run_id)

    def _read_run_file(self, run_id: str) -> dict[str, Any]:
        path = self._run_path(run_id)
        if not path.exists():
            raise RuntimeError(f"Run not found: {run_id}")
        payload = _read_json_with_retry(path)
        decoded = self._decode_run_payload(run_id, payload)
        decoded.setdefault(
            "reasoning",
            _reconstruct_stream_text(decoded.get("events", []), "thinking_token"),
        )
        sequences_changed = self._register_event_sequences(run_id, decoded)
        if payload.get("format") != _RUN_FORMAT_V2 or sequences_changed:
            self._write_encoded_run_file(
                run_id,
                _compact_artifact_for_storage(decoded),
            )
            self._set_indexed_run_format(run_id, _RUN_FORMAT_V2)
        return decoded

    def _merge_pending_events(
        self,
        run_id: str,
        artifact: dict[str, Any],
    ) -> dict[str, Any]:
        pending = list(self._pending_run_events.get(run_id, []))
        if not pending:
            return artifact
        artifact.setdefault("events", []).extend(pending)
        for event in pending:
            event_type = str(event.get("type", "") or "")
            payload = event.get("payload", {})
            if event_type == "token":
                artifact["answer"] = (
                    f"{str(artifact.get('answer', '') or '')}{payload.get('text', '')}"
                )
            elif event_type == "thinking_token":
                artifact["reasoning"] = (
                    f"{str(artifact.get('reasoning', '') or '')}{payload.get('text', '')}"
                )
        return artifact

    def _flush_pending_run_locked(self, run_id: str) -> None:
        if not self._pending_run_events.get(run_id):
            return
        artifact = self._merge_pending_events(run_id, self._read_run_file(run_id))
        self._write_run_file(run_id, artifact)

    def list_threads(self, *, user_id: str | None = None) -> list[dict[str, Any]]:
        index = self._read_index()
        items = list(index.get("threads", {}).values())
        if user_id:
            items = [item for item in items if item.get("user_id") == user_id]
        items.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return items

    def list_runs_for_thread(
        self, *, user_id: str | None, thread_id: str
    ) -> list[dict[str, Any]]:
        index = self._read_index()
        run_ids = [
            run_id
            for run_id, item in index.get("runs", {}).items()
            if item.get("thread_id") == thread_id
            and (user_id is None or item.get("user_id") == user_id)
        ]
        artifacts: list[dict[str, Any]] = []
        for run_id in run_ids:
            try:
                artifacts.append(self.get_run(run_id))
            except RuntimeError:
                continue
        artifacts.sort(
            key=lambda item: (
                str(item.get("started_at", "") or ""),
                str(item.get("run_id", "") or ""),
            )
        )
        return artifacts

    def list_users(self) -> list[dict[str, Any]]:
        index = self._read_index()
        items = list(index.get("users", {}).values())
        items.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return items

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        index = self._read_index()
        item = index.get("users", {}).get(user_id)
        if not item:
            return None
        return dict(item)

    def create_user(
        self, user_id: str, *, password: str | None = None
    ) -> dict[str, Any]:
        with self._lock:
            index = self._read_index()
            existing = index.get("users", {}).get(user_id)
            if existing:
                raise RuntimeError(f"User already exists: {user_id}")
            resolved_password = (password or "").strip() or None
            item = self._build_user_record(
                user_id,
                updated_at=now_timestamp(),
                password=resolved_password,
            )
            index["users"][user_id] = item
            self._write_index(index)
        return dict(item)

    def upsert_user(self, user_id: str) -> dict[str, Any]:
        with self._lock:
            index = self._read_index()
            item = self._build_user_record(
                user_id,
                updated_at=now_timestamp(),
                existing=index.get("users", {}).get(user_id),
            )
            index["users"][user_id] = item
            self._write_index(index)
        return dict(item)

    def verify_user_password(self, user_id: str, password: str) -> bool:
        user = self.get_user(user_id)
        if not user:
            raise RuntimeError(f"User not found: {user_id}")
        if user.get("protection") != PASSWORD_PROTECTED:
            return True
        try:
            salt = _decode_password_field(user, "password_salt")
            expected = _decode_password_field(user, "password_hash")
        except TypeError, ValueError:
            return False
        if user.get("password_kdf") == _PASSWORD_KDF_V2:
            actual, _key_encryption_key = _derive_password_material(password, salt)
        else:
            actual = _derive_password_hash(password, salt)
        return hmac.compare_digest(actual, expected)

    def unlock_user_key(self, user_id: str, *, password: str | None = None) -> None:
        user = self.get_user(user_id)
        if not user:
            raise RuntimeError(f"User not found: {user_id}")
        self._discard_user_key(user_id)
        if user.get("protection") == PASSWORD_PROTECTED:
            if not application_secret_protection_available():
                raise RuntimeError(
                    "Password-protected profiles require available OS secret storage."
                )
            resolved_password = (password or "").strip()
            if not resolved_password:
                raise RuntimeError("Password is required for this user.")
            if not self.verify_user_password(user_id, resolved_password):
                raise RuntimeError("Password did not match this user.")
            wrapped_key = _decode_wrapped_profile_key(user)
            if user.get("password_kdf") == _PASSWORD_KDF_V2:
                salt = _decode_password_field(user, "password_salt")
                _verifier, key_encryption_key = _derive_password_material(
                    resolved_password,
                    salt,
                )
                try:
                    key = unprotect_bytes_with_key(
                        wrapped_key,
                        key=key_encryption_key,
                        aad=_profile_key_aad(user_id),
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        "The protected profile key failed authentication."
                    ) from exc
            else:
                key = unprotect_bytes(
                    wrapped_key,
                    entropy=_derive_user_entropy(resolved_password, user),
                    require_protection=not self.config.allow_legacy_plaintext_migration,
                )
                self._migrate_password_user_record(
                    user_id,
                    password=resolved_password,
                    profile_key=key,
                )
        else:
            key = unprotect_bytes(
                _decode_wrapped_profile_key(user),
                require_protection=not self.config.allow_legacy_plaintext_migration,
            )
        if len(key) != _PASSWORD_KEY_LENGTH:
            raise RuntimeError("The profile key has an invalid length.")
        self._user_keys[user_id] = bytearray(key)

    def lock_user_key(self, user_id: str) -> None:
        self._discard_user_key(user_id)

    def lock_all_user_keys(self) -> None:
        self._discard_all_user_keys()

    def is_user_key_unlocked(self, user_id: str) -> bool:
        user = self.get_user(user_id)
        if not user:
            return False
        if user.get("protection") != PASSWORD_PROTECTED:
            return True
        return user_id in self._user_keys

    def get_thread(self, *, user_id: str, thread_id: str) -> dict[str, Any] | None:
        index = self._read_index()
        thread = index.get("threads", {}).get(self._thread_key(user_id, thread_id))
        if thread:
            return dict(thread)
        run_id = ""
        for item in index.get("runs", {}).values():
            if item.get("user_id") == user_id and item.get("thread_id") == thread_id:
                run_id = str(item.get("run_id", ""))
                break
        if not run_id:
            return None
        try:
            artifact = self.get_run(run_id)
        except RuntimeError:
            return None
        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "title": artifact.get("thread_title") or thread_id,
            "chat_model": artifact.get("chat_model", ""),
            "temperature": artifact.get("temperature", self.config.chat_temperature),
            "last_mode": artifact.get("mode", ""),
            "updated_at": artifact.get("completed_at")
            or artifact.get("started_at", ""),
            "last_prompt": str(artifact.get("prompt", ""))[:120],
            "last_run_id": run_id,
        }

    def rename_thread(
        self, *, user_id: str, thread_id: str, title: str
    ) -> dict[str, Any]:
        resolved_title = title.strip() or thread_id
        with self._lock:
            index = self._read_index()
            thread_key = self._thread_key(user_id, thread_id)
            thread = dict(index.get("threads", {}).get(thread_key) or {})
            if not thread:
                thread = {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "chat_model": "",
                    "temperature": None,
                    "last_mode": "chat",
                    "updated_at": now_timestamp(),
                    "last_prompt": "",
                    "last_run_id": "",
                }
            thread["title"] = resolved_title
            thread["updated_at"] = now_timestamp()
            thread.setdefault("temperature", self.config.chat_temperature)
            index["threads"][thread_key] = thread
            for run_id, item in index.get("runs", {}).items():
                if (
                    item.get("user_id") == user_id
                    and item.get("thread_id") == thread_id
                ):
                    item["thread_title"] = resolved_title
                    path = self._run_path(run_id)
                    if path.exists():
                        artifact = self.get_run(run_id)
                        artifact["thread_title"] = resolved_title
                        self._write_run_file(run_id, artifact)
                        item["artifact_format"] = _RUN_FORMAT_V2
                        self._index_run_for_search(artifact)
            self._write_index(index)
        return dict(thread)

    def upsert_thread(
        self,
        *,
        user_id: str,
        thread_id: str,
        title: str,
        chat_model: str,
        temperature: float | None,
        last_mode: str = "chat",
        updated_at: str | None = None,
        last_prompt: str = "",
        last_run_id: str = "",
    ) -> dict[str, Any]:
        resolved_updated_at = updated_at or now_timestamp()
        thread = {
            "user_id": user_id,
            "thread_id": thread_id,
            "title": title.strip() or thread_id,
            "chat_model": chat_model,
            "temperature": temperature,
            "last_mode": last_mode,
            "updated_at": resolved_updated_at,
            "last_prompt": last_prompt[:120],
            "last_run_id": last_run_id,
        }
        with self._lock:
            index = self._read_index()
            index["threads"][self._thread_key(user_id, thread_id)] = thread
            index["users"][user_id] = self._build_user_record(
                user_id,
                updated_at=resolved_updated_at,
                existing=index.get("users", {}).get(user_id),
            )
            self._write_index(index)
        return dict(thread)

    def delete_thread(self, *, user_id: str | None, thread_id: str) -> None:
        with self._lock:
            index = self._read_index()
            thread_keys = [
                key
                for key, item in index.get("threads", {}).items()
                if item.get("thread_id") == thread_id
                and (user_id is None or item.get("user_id") == user_id)
            ]
            run_ids = [
                run_id
                for run_id, item in index.get("runs", {}).items()
                if item.get("thread_id") == thread_id
                and (user_id is None or item.get("user_id") == user_id)
            ]
            for key in thread_keys:
                index["threads"].pop(key, None)
            for run_id in run_ids:
                index["runs"].pop(run_id, None)
                self._discard_pending_run(run_id, forget_sequence=True)
                path = self._run_path(run_id)
                if path.exists():
                    path.unlink()
                self._delete_quarantined_run_files(run_id)
            self._write_index(index)
            self._delete_search_entries(user_id=user_id, thread_id=thread_id)

    def delete_user(self, user_id: str) -> None:
        with self._lock:
            index = self._read_index()
            thread_keys = [
                key
                for key, item in index.get("threads", {}).items()
                if item.get("user_id") == user_id
            ]
            run_ids = [
                run_id
                for run_id, item in index.get("runs", {}).items()
                if item.get("user_id") == user_id
            ]
            for key in thread_keys:
                index["threads"].pop(key, None)
            index.get("users", {}).pop(user_id, None)
            for run_id in run_ids:
                index["runs"].pop(run_id, None)
                self._discard_pending_run(run_id, forget_sequence=True)
                path = self._run_path(run_id)
                if path.exists():
                    path.unlink()
                self._delete_quarantined_run_files(run_id)
            self._discard_user_key(user_id)
            self._write_index(index)
            self._delete_search_entries(user_id=user_id)

    def reset_all(self) -> None:
        with self._lock:
            self._pending_run_events.clear()
            self._pending_run_started_at.clear()
            self._pending_run_stream_bytes.clear()
            self._run_stream_bytes.clear()
            self._next_event_sequences.clear()
            for item in self.runs_dir.iterdir():
                if item.name == "index.json":
                    continue
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            self._discard_all_user_keys()
            self._write_index({"threads": {}, "runs": {}, "users": {}})
            self._delete_search_index()

    def refresh_search_index(self, *, user_id: str) -> None:
        index = self._read_index()
        with closing(self._open_search_index()) as conn:
            indexed_run_ids = {
                str(row[0])
                for row in conn.execute(
                    "SELECT DISTINCT run_id FROM run_search_entries WHERE user_id = ? AND run_id <> ''",
                    (user_id,),
                ).fetchall()
            }
        for run_id, item in index.get("runs", {}).items():
            if item.get("user_id") != user_id or item.get("mode") != "chat":
                continue
            expected_rows = 2 if item.get("status") == "completed" else 1
            if (
                run_id in indexed_run_ids
                and self._search_entry_count(user_id=user_id, run_id=run_id)
                >= expected_rows
            ):
                continue
            try:
                artifact = self.get_run(run_id)
            except RuntimeError:
                continue
            self._index_run_for_search(artifact)

    def search_messages(
        self, *, user_id: str, query: str, limit: int = _SEARCH_INDEX_LIMIT
    ) -> list[dict[str, Any]]:
        normalized_query = query.casefold().strip()
        if len(normalized_query) < 2:
            return []
        self.refresh_search_index(user_id=user_id)
        with closing(self._open_search_index()) as conn:
            rows = self._search_messages_fts(
                conn, user_id=user_id, query=query, limit=limit
            )
            if not rows:
                rows = self._search_messages_like(
                    conn, user_id=user_id, query=query, limit=limit
                )
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            content = str(row.get("content", "") or "")
            if normalized_query not in content.casefold():
                continue
            entry_key = str(row.get("entry_key", "") or "")
            if entry_key in seen:
                continue
            seen.add(entry_key)
            results.append(row)
        return results

    def replace_thread_search_messages(
        self,
        *,
        user_id: str,
        thread_id: str,
        title: str,
        chat_model: str,
        updated_at: str,
        messages: list[dict[str, Any]],
    ) -> None:
        with closing(self._open_search_index()) as conn:
            self._delete_search_entries_with_connection(
                conn, user_id=user_id, thread_id=thread_id
            )
            for index, message in enumerate(messages):
                role = str(message.get("role", "") or "")
                content = str(message.get("content", "") or "").strip()
                if role not in {"user", "assistant"} or not content:
                    continue
                entry = {
                    "entry_key": f"thread:{scoped_thread_id(user_id, thread_id)}:{index}",
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "run_id": "",
                    "role": role,
                    "history_index": index,
                    "thread_title": title,
                    "chat_model": chat_model,
                    "updated_at": updated_at,
                    "content": content,
                }
                self._insert_search_entry(conn, entry)
            conn.commit()

    def _index_run_for_search(self, artifact: dict[str, Any]) -> None:
        if artifact.get("mode") != "chat":
            return
        user_id = str(artifact.get("user_id", "") or "").strip()
        thread_id = str(artifact.get("thread_id", "") or "").strip()
        run_id = str(artifact.get("run_id", "") or "").strip()
        if not user_id or not thread_id or not run_id:
            return
        history_after_message_count = max(
            0, int(artifact.get("history_after_message_count", 0) or 0)
        )
        prompt = str(artifact.get("prompt", "") or "").strip()
        answer = str(artifact.get("answer", "") or "").strip()
        base = {
            "user_id": user_id,
            "thread_id": thread_id,
            "run_id": run_id,
            "thread_title": str(artifact.get("thread_title", "") or thread_id),
            "chat_model": str(artifact.get("chat_model", "") or ""),
            "updated_at": str(
                artifact.get("completed_at") or artifact.get("started_at") or ""
            ),
        }
        with closing(self._open_search_index()) as conn:
            self._delete_search_entries_with_connection(conn, run_id=run_id)
            if prompt:
                self._insert_search_entry(
                    conn,
                    {
                        **base,
                        "entry_key": f"{run_id}:user",
                        "role": "user",
                        "history_index": max(0, history_after_message_count - 1),
                        "content": prompt,
                    },
                )
            if answer:
                self._insert_search_entry(
                    conn,
                    {
                        **base,
                        "entry_key": f"{run_id}:assistant",
                        "role": "assistant",
                        "history_index": history_after_message_count,
                        "content": answer,
                    },
                )
            conn.commit()

    def _open_search_index(self) -> Any:
        self._search_index_path.parent.mkdir(parents=True, exist_ok=True)
        conn = open_application_sqlite(
            self._search_index_path, data_dir=self.config.data_dir
        )
        self._ensure_search_index_schema(conn)
        return conn

    def _ensure_search_index_schema(self, conn: Any) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_search_entries (
                entry_key TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                role TEXT NOT NULL,
                history_index INTEGER,
                thread_title TEXT NOT NULL,
                chat_model TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_run_search_entries_user_thread
            ON run_search_entries(user_id, thread_id, updated_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_run_search_entries_user_run
            ON run_search_entries(user_id, run_id)
            """
        )
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS run_search_fts USING fts5(
                    entry_key UNINDEXED,
                    user_id UNINDEXED,
                    thread_id UNINDEXED,
                    run_id UNINDEXED,
                    role UNINDEXED,
                    history_index UNINDEXED,
                    thread_title,
                    chat_model UNINDEXED,
                    updated_at UNINDEXED,
                    content,
                    tokenize='unicode61'
                )
                """
            )
        except Exception:
            pass
        conn.commit()

    def _insert_search_entry(self, conn: Any, entry: dict[str, Any]) -> None:
        values = (
            str(entry.get("entry_key", "") or ""),
            str(entry.get("user_id", "") or ""),
            str(entry.get("thread_id", "") or ""),
            str(entry.get("run_id", "") or ""),
            str(entry.get("role", "") or ""),
            entry.get("history_index"),
            str(entry.get("thread_title", "") or ""),
            str(entry.get("chat_model", "") or ""),
            str(entry.get("updated_at", "") or ""),
            str(entry.get("content", "") or ""),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO run_search_entries (
                entry_key,
                user_id,
                thread_id,
                run_id,
                role,
                history_index,
                thread_title,
                chat_model,
                updated_at,
                content
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        if self._search_index_has_fts(conn):
            conn.execute("DELETE FROM run_search_fts WHERE entry_key = ?", (values[0],))
            conn.execute(
                """
                INSERT INTO run_search_fts (
                    entry_key,
                    user_id,
                    thread_id,
                    run_id,
                    role,
                    history_index,
                    thread_title,
                    chat_model,
                    updated_at,
                    content
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )

    def _search_entry_count(self, *, user_id: str, run_id: str) -> int:
        with closing(self._open_search_index()) as conn:
            row = conn.execute(
                "SELECT count(*) FROM run_search_entries WHERE user_id = ? AND run_id = ?",
                (user_id, run_id),
            ).fetchone()
        return int(row[0] if row else 0)

    def _delete_search_entries(
        self,
        *,
        user_id: str | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        if not self._search_index_path.exists():
            return
        with closing(self._open_search_index()) as conn:
            self._delete_search_entries_with_connection(
                conn, user_id=user_id, thread_id=thread_id, run_id=run_id
            )
            conn.commit()

    def _delete_search_entries_with_connection(
        self,
        conn: Any,
        *,
        user_id: str | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        clauses: list[str] = []
        values: list[str] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            values.append(user_id)
        if thread_id is not None:
            clauses.append("thread_id = ?")
            values.append(thread_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        if not clauses:
            conn.execute("DELETE FROM run_search_entries")
            if self._search_index_has_fts(conn):
                conn.execute("DELETE FROM run_search_fts")
            return
        where = " AND ".join(clauses)
        conn.execute(f"DELETE FROM run_search_entries WHERE {where}", tuple(values))
        if self._search_index_has_fts(conn):
            conn.execute(f"DELETE FROM run_search_fts WHERE {where}", tuple(values))

    def _delete_search_index(self) -> None:
        self._search_index_path.unlink(missing_ok=True)

    def _search_messages_fts(
        self, conn: Any, *, user_id: str, query: str, limit: int
    ) -> list[dict[str, Any]]:
        fts_query = _build_fts_query(query)
        if not fts_query or not self._search_index_has_fts(conn):
            return []
        try:
            rows = conn.execute(
                """
                SELECT
                    entry_key,
                    user_id,
                    thread_id,
                    run_id,
                    role,
                    history_index,
                    thread_title,
                    chat_model,
                    updated_at,
                    content
                FROM run_search_fts
                WHERE run_search_fts MATCH ? AND user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (fts_query, user_id, max(1, int(limit))),
            ).fetchall()
        except Exception:
            return []
        return [_search_row_to_dict(row) for row in rows]

    def _search_messages_like(
        self, conn: Any, *, user_id: str, query: str, limit: int
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT
                entry_key,
                user_id,
                thread_id,
                run_id,
                role,
                history_index,
                thread_title,
                chat_model,
                updated_at,
                content
            FROM run_search_entries
            WHERE user_id = ? AND content LIKE ? ESCAPE '\\'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, f"%{_escape_like_query(query)}%", max(1, int(limit))),
        ).fetchall()
        return [_search_row_to_dict(row) for row in rows]

    def _search_index_has_fts(self, conn: Any) -> bool:
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'run_search_fts'"
            ).fetchone()
            return bool(row)
        except Exception:
            return False

    def _read_index(self) -> dict[str, Any]:
        payload = _read_json_with_retry(self._index_path)
        migrated_plaintext_index = False
        if payload.get("format") == _INDEX_FORMAT:
            try:
                encoded = str(payload.get("payload", "") or "").strip()
                decrypted = unprotect_bytes(
                    base64.b64decode(encoded.encode("ascii"), validate=True),
                    require_protection=not self.config.allow_legacy_plaintext_migration,
                )
                payload = json.loads(decrypted.decode("utf-8"))
            except (RuntimeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "The run index failed authentication or is unavailable."
                ) from exc
        elif "format" in payload:
            raise RuntimeError("The run index format is not supported.")
        elif self.config.allow_legacy_plaintext_migration:
            migrated_plaintext_index = True
        else:
            raise RuntimeError(
                "Atlas found a legacy plaintext run index. For a one-time migration, "
                "start Atlas with ATLAS_ALLOW_LEGACY_PLAINTEXT_MIGRATION=1, then remove it."
            )
        if not isinstance(payload, dict):
            raise RuntimeError("The run index payload is invalid.")
        for key in ("threads", "runs", "users"):
            value = payload.get(key, {})
            if not isinstance(value, dict):
                raise RuntimeError("The run index payload is invalid.")
        payload.setdefault("threads", {})
        payload.setdefault("runs", {})
        payload.setdefault("users", {})
        migrated_threads: dict[str, dict[str, Any]] = {}
        for legacy_key, item in payload["threads"].items():
            user_id = str(item.get("user_id", "") or "")
            thread_id = str(item.get("thread_id", "") or "")
            if not user_id or not thread_id:
                migrated_threads[str(legacy_key)] = item
                continue
            migrated_threads[self._thread_key(user_id, thread_id)] = item
        payload["threads"] = migrated_threads
        for item in payload["threads"].values():
            item.setdefault("title", item.get("thread_id", ""))
            item.setdefault("temperature", self.config.chat_temperature)
        for item in payload["runs"].values():
            item.setdefault("thread_title", item.get("thread_id", ""))
            item.setdefault("temperature", self.config.chat_temperature)
            item.setdefault("history_after_message_count", 0)
            user_id = item.get("user_id")
            if user_id and user_id not in payload["users"]:
                payload["users"][user_id] = self._build_user_record(
                    user_id,
                    updated_at=item.get("started_at", ""),
                )
        for item in payload["threads"].values():
            user_id = item.get("user_id")
            if user_id and user_id not in payload["users"]:
                payload["users"][user_id] = self._build_user_record(
                    user_id,
                    updated_at=item.get("updated_at", ""),
                )
        payload["users"] = {
            user_id: self._build_user_record(
                user_id,
                updated_at=item.get("updated_at", ""),
                existing=item,
            )
            for user_id, item in payload["users"].items()
        }
        formats_changed = False
        for run_id, item in payload["runs"].items():
            if item.get("artifact_format"):
                continue
            try:
                raw_run = _read_json_with_retry(self._run_path(str(run_id)))
            except (OSError, RuntimeError, ValueError):
                continue
            if raw_run.get("format") in {_RUN_FORMAT_V1, _RUN_FORMAT_V2}:
                item["artifact_format"] = raw_run["format"]
                formats_changed = True
            elif "format" not in raw_run:
                item["artifact_format"] = _RUN_FORMAT_LEGACY_PLAINTEXT
                formats_changed = True
        if formats_changed or migrated_plaintext_index:
            self._write_index(payload)
        return payload

    def _write_index(self, payload: dict[str, Any]) -> None:
        encrypted = protect_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            require_protection=True,
        )
        _atomic_write_json(
            self._index_path,
            {
                "format": _INDEX_FORMAT,
                "payload": base64.b64encode(encrypted).decode("ascii"),
            },
        )

    def _write_run_file(self, run_id: str, payload: dict[str, Any]) -> None:
        self._register_event_sequences(run_id, payload)
        storage_payload = _compact_artifact_for_storage(payload)
        self._write_encoded_run_file(run_id, storage_payload)
        self._discard_pending_run(run_id)

    def _set_indexed_run_format(self, run_id: str, run_format: str) -> None:
        index = self._read_index()
        item = index.get("runs", {}).get(run_id)
        if not isinstance(item, dict) or item.get("artifact_format") == run_format:
            return
        item["artifact_format"] = run_format
        self._write_index(index)

    def _register_event_sequences(
        self,
        run_id: str,
        payload: dict[str, Any],
    ) -> bool:
        events = payload.get("events", [])
        if not isinstance(events, list):
            raise RuntimeError("The run artifact events payload is invalid.")
        changed = False
        next_sequence = 1
        for event in events:
            if not isinstance(event, dict):
                raise RuntimeError("The run artifact contains an invalid event.")
            sequence = event.get("sequence")
            sequence_end = event.get("sequence_end")
            width = 1
            if (
                type(sequence) is int
                and type(sequence_end) is int
                and sequence >= 1
                and sequence_end >= sequence
            ):
                width = sequence_end - sequence + 1
            if sequence != next_sequence:
                event["sequence"] = next_sequence
                changed = True
            resolved_sequence_end = next_sequence + width - 1
            if width > 1:
                if event.get("sequence_end") != resolved_sequence_end:
                    event["sequence_end"] = resolved_sequence_end
                    changed = True
            elif "sequence_end" in event:
                event.pop("sequence_end", None)
                changed = True
            next_sequence = resolved_sequence_end + 1
        pending_next_sequence = max(
            (
                int(event.get("sequence", 0)) + 1
                for event in self._pending_run_events.get(run_id, [])
                if type(event.get("sequence")) is int
            ),
            default=1,
        )
        self._next_event_sequences[run_id] = max(
            self._next_event_sequences.get(run_id, 1),
            next_sequence,
            pending_next_sequence,
        )
        return changed

    def _write_encoded_run_file(self, run_id: str, payload: dict[str, Any]) -> None:
        path = self._run_path(run_id)
        user_id = str(payload.get("user_id", "") or "").strip()
        if not user_id:
            _atomic_write_json(path, payload)
            return
        key = self._require_user_key(user_id)
        encrypted = protect_bytes_with_key(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            key=key,
            aad=_run_payload_aad(user_id, run_id),
        )
        _atomic_write_json(
            path,
            {
                "format": _RUN_FORMAT_V2,
                "user_id": user_id,
                "payload": base64.b64encode(encrypted).decode("ascii"),
            },
        )

    def _discard_pending_run(
        self,
        run_id: str,
        *,
        forget_sequence: bool = False,
    ) -> None:
        self._pending_run_events.pop(run_id, None)
        self._pending_run_started_at.pop(run_id, None)
        self._pending_run_stream_bytes.pop(run_id, None)
        if forget_sequence:
            self._next_event_sequences.pop(run_id, None)
            self._run_stream_bytes.pop(run_id, None)

    def _delete_quarantined_run_files(self, run_id: str) -> None:
        quarantine_dir = self.runs_dir / "quarantine"
        if not quarantine_dir.is_dir():
            return
        for path in quarantine_dir.glob(f"{run_id}.*.json"):
            path.unlink(missing_ok=True)

    @staticmethod
    def _thread_key(user_id: str, thread_id: str) -> str:
        return scoped_thread_id(user_id, thread_id)

    def _run_path(self, run_id: str) -> Path:
        resolved = str(run_id or "")
        if (
            not _SAFE_RUN_ID_PATTERN.fullmatch(resolved)
            or ".." in resolved
            or "/" in resolved
            or "\\" in resolved
        ):
            raise RuntimeError("Run id contains unsafe path characters.")
        return self.runs_dir / f"{resolved}.json"

    def _build_user_record(
        self,
        user_id: str,
        *,
        updated_at: str | None = None,
        existing: dict[str, Any] | None = None,
        password: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "user_id": user_id,
            "updated_at": updated_at
            or (existing or {}).get("updated_at")
            or now_timestamp(),
            "protection": PASSWORDLESS_PROTECTION,
            "password_hash": None,
            "password_salt": None,
            "password_kdf": None,
            "wrapped_profile_key": None,
        }
        if existing:
            record["protection"] = str(
                existing.get("protection", PASSWORDLESS_PROTECTION)
                or PASSWORDLESS_PROTECTION
            )
            record["password_hash"] = existing.get("password_hash")
            record["password_salt"] = existing.get("password_salt")
            record["password_kdf"] = existing.get("password_kdf")
            record["wrapped_profile_key"] = existing.get("wrapped_profile_key")
        if password:
            profile_key = os.urandom(_PASSWORD_KEY_LENGTH)
            record = self._password_user_record(
                user_id,
                updated_at=str(record["updated_at"]),
                password=password,
                profile_key=profile_key,
            )
        elif record["wrapped_profile_key"] is None:
            record["wrapped_profile_key"] = base64.b64encode(
                protect_bytes(
                    os.urandom(_PASSWORD_KEY_LENGTH),
                    require_protection=True,
                )
            ).decode("ascii")
        if record["protection"] != PASSWORD_PROTECTED:
            record["protection"] = PASSWORDLESS_PROTECTION
            record["password_salt"] = None
            record["password_hash"] = None
            record["password_kdf"] = None
        return record

    def _cache_user_key_from_record(
        self, user_id: str, user: dict[str, Any], *, password: str | None = None
    ) -> None:
        if user.get("protection") == PASSWORD_PROTECTED:
            if not password:
                return
            self.unlock_user_key(user_id, password=password)
            return
        else:
            key = unprotect_bytes(
                _decode_wrapped_profile_key(user),
                require_protection=not self.config.allow_legacy_plaintext_migration,
            )
        if len(key) != _PASSWORD_KEY_LENGTH:
            raise RuntimeError("The profile key has an invalid length.")
        self._discard_user_key(user_id)
        self._user_keys[user_id] = bytearray(key)

    def _password_user_record(
        self,
        user_id: str,
        *,
        updated_at: str,
        password: str,
        profile_key: bytes,
    ) -> dict[str, Any]:
        if not application_secret_protection_available():
            raise RuntimeError(
                "Password-protected profiles require available OS secret storage."
            )
        if len(profile_key) != _PASSWORD_KEY_LENGTH:
            raise RuntimeError("The profile key has an invalid length.")
        salt = os.urandom(16)
        verifier, key_encryption_key = _derive_password_material(password, salt)
        wrapped_key = protect_bytes_with_key(
            profile_key,
            key=key_encryption_key,
            aad=_profile_key_aad(user_id),
        )
        return {
            "user_id": user_id,
            "updated_at": updated_at,
            "protection": PASSWORD_PROTECTED,
            "password_hash": base64.b64encode(verifier).decode("ascii"),
            "password_salt": base64.b64encode(salt).decode("ascii"),
            "password_kdf": _PASSWORD_KDF_V2,
            "wrapped_profile_key": base64.b64encode(wrapped_key).decode("ascii"),
        }

    def _migrate_password_user_record(
        self,
        user_id: str,
        *,
        password: str,
        profile_key: bytes,
    ) -> None:
        with self._lock:
            index = self._read_index()
            current = index.get("users", {}).get(user_id)
            if not current or current.get("password_kdf") == _PASSWORD_KDF_V2:
                return
            index["users"][user_id] = self._password_user_record(
                user_id,
                updated_at=str(current.get("updated_at", "") or now_timestamp()),
                password=password,
                profile_key=profile_key,
            )
            self._write_index(index)

    def _require_user_key(self, user_id: str) -> bytes:
        cached = self._user_keys.get(user_id)
        if cached is not None:
            return bytes(cached)
        user = self.get_user(user_id)
        if not user:
            raise RuntimeError(f"User not found: {user_id}")
        if user.get("protection") == PASSWORD_PROTECTED:
            raise RuntimeError("Unlock this user before continuing.")
        self.unlock_user_key(user_id)
        cached = self._user_keys.get(user_id)
        if cached is None:
            raise RuntimeError(f"Profile key is not available for user: {user_id}")
        return bytes(cached)

    def _discard_user_key(self, user_id: str) -> None:
        cached = self._user_keys.pop(user_id, None)
        if cached is None:
            return
        for index in range(len(cached)):
            cached[index] = 0

    def _discard_all_user_keys(self) -> None:
        for user_id in list(self._user_keys):
            self._discard_user_key(user_id)

    def _decode_run_payload(
        self, run_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        run_format = payload.get("format")
        if run_format not in {_RUN_FORMAT_V1, _RUN_FORMAT_V2}:
            if "format" in payload:
                raise RuntimeError("The run artifact format is not supported.")
            if not self.config.allow_legacy_plaintext_migration:
                raise RuntimeError(
                    "Atlas found a legacy plaintext run artifact. For a one-time "
                    "migration, start Atlas with "
                    "ATLAS_ALLOW_LEGACY_PLAINTEXT_MIGRATION=1, then remove it."
                )
            self._validate_legacy_plaintext_run(run_id, payload)
            return payload
        user_id = str(payload.get("user_id", "") or "").strip()
        if not user_id:
            raise RuntimeError("The run artifact user id is invalid.")
        key = self._require_user_key(user_id)
        try:
            encrypted = base64.b64decode(
                str(payload.get("payload", "") or "").encode("ascii"),
                validate=True,
            )
            if run_format == _RUN_FORMAT_V2:
                decrypted = unprotect_bytes_with_key(
                    encrypted,
                    key=key,
                    aad=_run_payload_aad(user_id, run_id),
                )
            else:
                decrypted = unprotect_bytes(
                    encrypted,
                    entropy=key,
                    require_protection=not self.config.allow_legacy_plaintext_migration,
                )
            decoded = json.loads(decrypted.decode("utf-8"))
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("The run artifact failed authentication.") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("The run artifact payload is invalid.")
        if (
            str(decoded.get("run_id", "") or "") != run_id
            or str(decoded.get("user_id", "") or "").strip() != user_id
        ):
            raise RuntimeError("The run artifact identity is invalid.")
        return decoded

    def _validate_legacy_plaintext_run(
        self,
        run_id: str,
        payload: dict[str, Any],
    ) -> None:
        index_item = self._read_index().get("runs", {}).get(run_id)
        if not isinstance(index_item, dict):
            raise RuntimeError(
                "The legacy plaintext run artifact is not present in the index."
            )
        if index_item.get("artifact_format") != _RUN_FORMAT_LEGACY_PLAINTEXT:
            raise RuntimeError(
                "The legacy plaintext run artifact is not allowed by the index."
            )
        required_types = {
            "run_id": str,
            "user_id": str,
            "thread_id": str,
            "mode": str,
            "status": str,
            "events": list,
            "trace_items": list,
        }
        if any(
            key not in payload or not isinstance(payload.get(key), expected_type)
            for key, expected_type in required_types.items()
        ):
            raise RuntimeError("The legacy plaintext run artifact is invalid.")
        if str(payload.get("run_id", "") or "") != run_id:
            raise RuntimeError("The legacy plaintext run identity is invalid.")
        for key in ("user_id", "thread_id", "mode"):
            value = str(payload.get(key, "") or "")
            indexed_value = str(index_item.get(key, "") or "")
            if not value or value != indexed_value:
                raise RuntimeError("The legacy plaintext run identity is invalid.")


def _stream_event_text(event_type: str, payload: Any) -> str:
    if event_type not in _STREAM_EVENT_TYPES or not isinstance(payload, dict):
        return ""
    text = payload.get("text", "")
    return text if isinstance(text, str) else ""


def _event_sequence_end(event: dict[str, Any]) -> int:
    sequence = event.get("sequence")
    if type(sequence) is not int or sequence < 1:
        return 0
    sequence_end = event.get("sequence_end")
    if type(sequence_end) is int and sequence_end >= sequence:
        return sequence_end
    return sequence


def _reconstruct_stream_text(events: Any, event_type: str) -> str:
    if not isinstance(events, list):
        return ""
    parts: list[str] = []
    snapshot_key = "answer_text" if event_type == "token" else "thinking_text"
    for event in events:
        if not isinstance(event, dict):
            continue
        current_type = str(event.get("type", "") or "")
        payload = event.get("payload", {})
        if current_type == event_type:
            parts.append(_stream_event_text(current_type, payload))
        elif current_type == "stream_snapshot" and isinstance(payload, dict):
            value = payload.get(snapshot_key, "")
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def _artifact_stream_byte_count(artifact: dict[str, Any]) -> int:
    events = artifact.get("events", [])
    answer = artifact.get("answer")
    reasoning = artifact.get("reasoning")
    answer_text = (
        answer
        if isinstance(answer, str)
        else _reconstruct_stream_text(events, "token")
    )
    reasoning_text = (
        reasoning
        if isinstance(reasoning, str)
        else _reconstruct_stream_text(events, "thinking_token")
    )
    return len(answer_text.encode("utf-8")) + len(reasoning_text.encode("utf-8"))


def _compact_artifact_for_storage(payload: dict[str, Any]) -> dict[str, Any]:
    compacted = dict(payload)
    events = payload.get("events", [])
    if not isinstance(events, list):
        raise RuntimeError("The run artifact events payload is invalid.")
    compacted_events = _coalesce_stream_events(events)
    if len(compacted_events) > _MAX_PERSISTED_RUN_EVENTS:
        compacted_events = _collapse_stream_segments(compacted_events)
        compacted["stream_events_compacted"] = True
    if len(compacted_events) > _MAX_PERSISTED_RUN_EVENTS:
        compacted_events = _truncate_event_history(compacted_events)
        compacted["event_history_truncated"] = True
    compacted["events"] = compacted_events
    reasoning = payload.get("reasoning")
    compacted["reasoning"] = (
        reasoning
        if isinstance(reasoning, str)
        else _reconstruct_stream_text(events, "thinking_token")
    )
    return compacted


def _coalesce_stream_events(events: list[Any]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    active_event: dict[str, Any] | None = None
    active_parts: list[str] = []
    active_bytes = 0
    active_type = ""
    active_end = 0

    def finish_active_event() -> None:
        nonlocal active_event, active_parts, active_bytes, active_type, active_end
        if active_event is not None:
            active_event["payload"]["text"] = "".join(active_parts)
        active_event = None
        active_parts = []
        active_bytes = 0
        active_type = ""
        active_end = 0

    for raw_event in events:
        if not isinstance(raw_event, dict):
            raise RuntimeError("The run artifact contains an invalid event.")
        event = dict(raw_event)
        payload = raw_event.get("payload", {})
        if isinstance(payload, dict):
            event["payload"] = dict(payload)
        event_type = str(event.get("type", "") or "")
        text = _stream_event_text(event_type, event.get("payload"))
        event_payload = event.get("payload")
        sequence = event.get("sequence")
        text_bytes = len(text.encode("utf-8")) if text else 0
        can_compact = (
            bool(text)
            and isinstance(event_payload, dict)
            and set(event_payload) == {"text"}
            and type(sequence) is int
            and sequence >= 1
        )
        if (
            can_compact
            and active_event is not None
            and active_type == event_type
            and sequence == active_end + 1
            and active_bytes + text_bytes <= _PERSISTED_STREAM_CHUNK_BYTES
        ):
            active_parts.append(text)
            active_bytes += text_bytes
            active_end = _event_sequence_end(event)
            if active_end > int(active_event.get("sequence", 0) or 0):
                active_event["sequence_end"] = active_end
            continue

        finish_active_event()
        compacted.append(event)
        if can_compact:
            active_event = event
            active_parts = [text]
            active_bytes = text_bytes
            active_type = event_type
            active_end = _event_sequence_end(event)

    finish_active_event()
    return compacted


def _collapse_stream_segments(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    segment: list[dict[str, Any]] = []

    def flush_segment() -> None:
        if not segment:
            return
        first = segment[0]
        last = segment[-1]
        next_sequence = int(first.get("sequence", 0) or 0)
        segment_end = _event_sequence_end(last)
        for event_type in ("thinking_token", "token"):
            matching = [
                event for event in segment if event.get("type") == event_type
            ]
            if not matching:
                continue
            logical_width = sum(
                max(
                    1,
                    _event_sequence_end(event)
                    - int(event.get("sequence", 0) or 0)
                    + 1,
                )
                for event in matching
            )
            sequence_end = min(
                segment_end,
                next_sequence + logical_width - 1,
            )
            compacted_event: dict[str, Any] = {
                "type": event_type,
                "timestamp": str(matching[0].get("timestamp", "") or ""),
                "payload": {
                    "text": _reconstruct_stream_text(matching, event_type),
                },
                "sequence": next_sequence,
            }
            if sequence_end > next_sequence:
                compacted_event["sequence_end"] = sequence_end
            collapsed.append(compacted_event)
            next_sequence = sequence_end + 1
        segment.clear()

    for event in events:
        if event.get("type") in _STREAM_EVENT_TYPES:
            segment.append(event)
            continue
        flush_segment()
        collapsed.append(event)
    flush_segment()
    return collapsed


def _truncate_event_history(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terminal_events = [
        event
        for event in events
        if event.get("type") in {"run_completed", "run_failed"}
    ]
    terminal_event = terminal_events[-1] if terminal_events else None
    nonterminal_events = [
        event
        for event in events
        if event.get("type") not in {"run_completed", "run_failed"}
    ]
    keep_count = max(
        0,
        _MAX_PERSISTED_RUN_EVENTS - (2 if terminal_event is not None else 1),
    )
    kept = nonterminal_events[:keep_count]
    dropped = nonterminal_events[keep_count:]
    if dropped:
        marker: dict[str, Any] = {
            "type": "event_history_truncated",
            "timestamp": str(dropped[-1].get("timestamp", "") or ""),
            "payload": {"dropped_event_count": len(dropped)},
            "sequence": dropped[0].get("sequence"),
        }
        sequence_end = _event_sequence_end(dropped[-1])
        if sequence_end > int(marker.get("sequence", 0) or 0):
            marker["sequence_end"] = sequence_end
        kept.append(marker)
    if terminal_event is not None:
        kept.append(terminal_event)
    return kept


def _search_row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "entry_key": str(row[0] or ""),
        "user_id": str(row[1] or ""),
        "thread_id": str(row[2] or ""),
        "run_id": str(row[3] or ""),
        "role": str(row[4] or ""),
        "history_index": None if row[5] is None else int(row[5]),
        "thread_title": str(row[6] or ""),
        "chat_model": str(row[7] or ""),
        "updated_at": str(row[8] or ""),
        "content": str(row[9] or ""),
    }


def _escape_like_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_fts_query(value: str) -> str:
    tokens = []
    current = []
    for char in value.casefold():
        if char.isalnum() or char == "_":
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    tokens = [token for token in tokens if len(token) >= 2]
    if not tokens:
        return ""
    return " AND ".join(f"{token}*" for token in tokens[:8])


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.parent.chmod(0o700)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    last_error: PermissionError | None = None

    for attempt in range(6):
        temp_path = path.with_name(
            f"{path.stem}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            descriptor = os.open(
                temp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            if os.name != "nt":
                path.chmod(0o600)
                try:
                    directory = os.open(path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                except OSError:
                    pass
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
        finally:
            temp_path.unlink(missing_ok=True)

    if last_error is not None:
        raise last_error


def _read_json_with_retry(path: Path) -> dict[str, Any]:
    last_error: PermissionError | None = None
    for attempt in range(6):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return json.loads(path.read_text(encoding="utf-8"))


def _derive_password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_PASSWORD_KEY_LENGTH,
    )


def _derive_password_material(password: str, salt: bytes) -> tuple[bytes, bytes]:
    material = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_PASSWORD_MATERIAL_LENGTH,
    )
    return material[:_PASSWORD_KEY_LENGTH], material[_PASSWORD_KEY_LENGTH:]


def _decode_password_field(user: dict[str, Any], field: str) -> bytes:
    encoded = str(user.get(field, "") or "").strip()
    if not encoded:
        raise ValueError(f"{field} is missing")
    decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
    expected_length = 16 if field == "password_salt" else _PASSWORD_KEY_LENGTH
    if len(decoded) != expected_length:
        raise ValueError(f"{field} has an invalid length")
    return decoded


def _decode_wrapped_profile_key(user: dict[str, Any]) -> bytes:
    encoded = str(user.get("wrapped_profile_key", "") or "").strip()
    if not encoded:
        raise RuntimeError("The wrapped profile key is missing.")
    try:
        wrapped = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("The wrapped profile key is invalid.") from exc
    if not wrapped:
        raise RuntimeError("The wrapped profile key is empty.")
    return wrapped


def _profile_key_aad(user_id: str) -> bytes:
    return f"atlas-profile-key-v2:{user_id}".encode("utf-8")


def _run_payload_aad(user_id: str, run_id: str) -> bytes:
    return f"atlas-run-v2:{user_id}:{run_id}".encode("utf-8")


def _derive_user_entropy(password: str, user: dict[str, Any]) -> bytes:
    try:
        salt = _decode_password_field(user, "password_salt")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Password salt is invalid for this protected user.") from exc
    return _derive_password_hash(password, salt)
