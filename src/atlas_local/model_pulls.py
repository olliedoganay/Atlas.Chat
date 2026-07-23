from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from uuid import uuid4

MAX_PULL_HISTORY = 50


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ModelPull:
    pull_id: str
    model: str
    status: str = "queued"
    detail: str = "Queued"
    completed: int = 0
    total: int = 0
    started_at: str = field(default_factory=_timestamp)
    updated_at: str = field(default_factory=_timestamp)
    error: str = ""
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _response: BinaryIO | None = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        progress = None
        if self.total > 0:
            progress = max(0.0, min(1.0, self.completed / self.total))
        return {
            "pull_id": self.pull_id,
            "model": self.model,
            "status": self.status,
            "detail": self.detail,
            "completed": self.completed,
            "total": self.total,
            "progress": progress,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "error": self.error or None,
        }


class ModelPullManager:
    def __init__(self, *, max_concurrent: int = 1):
        self._lock = threading.RLock()
        self._pulls: dict[str, ModelPull] = {}
        self._max_concurrent = max(1, int(max_concurrent))

    def start(self, *, ollama_url: str, model: str) -> dict[str, Any]:
        resolved_model = model.strip()
        if not resolved_model or len(resolved_model) > 200:
            raise RuntimeError("Choose a valid local model name.")
        with self._lock:
            self._prune_history()
            for pull in self._pulls.values():
                if pull.model == resolved_model and pull.status in {"queued", "pulling"}:
                    return pull.public()
            active = sum(1 for pull in self._pulls.values() if pull.status in {"queued", "pulling"})
            if active >= self._max_concurrent:
                raise RuntimeError("Another model download is already active.")
            pull = ModelPull(pull_id=str(uuid4()), model=resolved_model)
            self._pulls[pull.pull_id] = pull
        threading.Thread(
            target=self._run,
            args=(pull, ollama_url.rstrip("/") + "/"),
            daemon=True,
            name=f"atlas-model-pull-{pull.pull_id[:8]}",
        ).start()
        return pull.public()

    def _prune_history(self) -> None:
        terminal = sorted(
            (
                pull
                for pull in self._pulls.values()
                if pull.status in {"completed", "failed", "cancelled"}
            ),
            key=lambda item: item.updated_at,
        )
        excess = max(0, len(self._pulls) - MAX_PULL_HISTORY + 1)
        for pull in terminal[:excess]:
            self._pulls.pop(pull.pull_id, None)

    def get(self, pull_id: str) -> dict[str, Any]:
        with self._lock:
            pull = self._pulls.get(pull_id)
            if pull is None:
                raise RuntimeError("Model download was not found.")
            return pull.public()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                pull.public()
                for pull in sorted(self._pulls.values(), key=lambda item: item.started_at, reverse=True)
            ]

    def cancel(self, pull_id: str) -> dict[str, Any]:
        with self._lock:
            pull = self._pulls.get(pull_id)
            if pull is None:
                raise RuntimeError("Model download was not found.")
            if pull.status not in {"queued", "pulling"}:
                return pull.public()
            pull._cancel.set()
            response = pull._response
        if response is not None:
            try:
                response.close()
            except OSError:
                pass
        self._update(pull, status="cancelled", detail="Download cancelled")
        return pull.public()

    def shutdown(self) -> None:
        with self._lock:
            pull_ids = [
                pull.pull_id for pull in self._pulls.values() if pull.status in {"queued", "pulling"}
            ]
        for pull_id in pull_ids:
            self.cancel(pull_id)

    def _run(self, pull: ModelPull, ollama_url: str) -> None:
        self._update(pull, status="pulling", detail="Connecting to Ollama")
        body = json.dumps({"name": pull.model, "stream": True}).encode("utf-8")
        request = Request(
            urljoin(ollama_url, "api/pull"),
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
        )
        try:
            response = urlopen(request, timeout=30)
            with self._lock:
                pull._response = response
            with response:
                for raw_line in response:
                    if pull._cancel.is_set():
                        return
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        continue
                    error = str(payload.get("error", "") or "").strip()
                    if error:
                        raise RuntimeError(error)
                    self._update(
                        pull,
                        detail=str(payload.get("status", "") or "Downloading"),
                        completed=int(payload.get("completed", 0) or 0),
                        total=int(payload.get("total", 0) or 0),
                    )
            if not pull._cancel.is_set():
                self._update(pull, status="completed", detail="Model ready")
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            if not pull._cancel.is_set():
                message = str(getattr(exc, "reason", "") or exc).strip() or "Model download failed."
                self._update(pull, status="failed", detail="Download failed", error=message)
        finally:
            with self._lock:
                pull._response = None

    def _update(self, pull: ModelPull, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(pull, key, value)
            pull.updated_at = _timestamp()


_MODEL_PULL_MANAGER = ModelPullManager()


def get_model_pull_manager() -> ModelPullManager:
    return _MODEL_PULL_MANAGER
