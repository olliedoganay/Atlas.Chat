from __future__ import annotations

import base64
import binascii
import io
import queue
import re
import shutil
import threading
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import islice
from time import monotonic
from types import SimpleNamespace
from typing import Any, Callable
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pypdf import PdfReader

from .config import (
    CHAT_PROVIDER_LABELS,
    DEFAULT_OLLAMA_URL,
    OPENAI_COMPATIBLE_PROVIDER_DEFAULT_URLS,
    AppConfig,
    chat_provider_label,
    is_ollama_chat_provider,
    load_config,
    normalize_chat_provider,
)
from .code_runner import shutdown_runner
from .discovery import build_discovery_report
from .graph.builder import AgentApplication, build_chat_application
from .graph.builder import post_synthesis_node_sequence, pre_synthesis_node_sequence
from .graph.context import GraphContext
from .graph.nodes import _build_answer_messages, _finalize_answer_text, _latest_user_text
from .llm import (
    OLLAMA_CONTEXT_WINDOW_PRESETS,
    OllamaCatalogSnapshot,
    format_runtime_error,
    inspect_local_models,
)
from .local_provider import normalize_local_provider_base_url
from .memory.models import MemoryRecord
from .model_pulls import get_model_pull_manager
from .provider_settings import (
    clear_provider_api_key,
    load_provider_settings,
    save_provider_settings,
)
from .run_contract import RunEvent, RunHub
from .run_store import ACTIVE_RUN_STATUSES, PASSWORD_PROTECTED, RunStore
from .security import (
    application_secret_protection_available,
    local_secret_storage_label,
    open_application_sqlite,
    purge_legacy_migration_backups,
    sqlcipher_enabled,
)
from .session import legacy_scoped_thread_id, parse_scoped_thread_id, scoped_thread_id
from .text_normalization import MojibakeRepairStream
from .version import atlas_version


DEFAULT_COMPACTION_TIMEOUT_SECONDS = 25.0
COMPACTION_CANCEL_POLL_SECONDS = 0.05
COMPACTION_PIN_LIMIT = 36
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 8.0
MAX_ATTACHMENTS = 8
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_TOTAL_ATTACHMENT_BYTES = 25 * 1024 * 1024
MAX_ATTACHMENT_TEXT_CHARS = 14_000
MAX_ATTACHMENT_CONTEXT_CHARS = 42_000
MAX_PDF_EXTRACTED_CHARS = 18_000
MAX_PDF_PAGES = 64
MAX_MODEL_STREAM_CHARS = 2_000_000
_INTERRUPTED_RUN_ERROR = "Atlas backend restarted while this run was active."


class _CompactionModelUnavailable(RuntimeError):
    """Raised when compaction cannot obtain a trustworthy model summary."""


_MEDIA_TYPE_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$"
)
_IMAGE_MEDIA_TYPE_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-png": "image/png",
}
_SUPPORTED_IMAGE_MEDIA_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/bmp",
        "image/x-ms-bmp",
        "image/tiff",
        "image/x-tiff",
        "image/vnd.microsoft.icon",
        "image/x-icon",
    }
)


@dataclass
class _QueuedRunJob:
    mode: str
    run_id: str
    prompt: str
    user_id: str
    thread_id: str
    chat_model: str
    temperature: float | None
    reasoning_mode: str | None
    cross_chat_memory: bool
    auto_compact_long_chats: bool
    attachments: list[dict[str, Any]]


@dataclass
class _CompactionCallBudget:
    remaining: int = 1

    def claim(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


@dataclass
class AtlasBackendService:
    config: AppConfig
    app: AgentApplication
    run_store: RunStore
    run_hub: RunHub
    _control_lock: threading.Lock = field(default_factory=threading.Lock)
    _active_run_id: str | None = None
    _cancelled_runs: set[str] = field(default_factory=set)
    _pending_runs: list[_QueuedRunJob] = field(default_factory=list)
    _worker_wakeup: threading.Event = field(default_factory=threading.Event)
    _worker_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _shutdown_requested: bool = field(default=False, init=False, repr=False)
    _shutdown_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _shutdown_complete: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _shutdown_started: bool = field(default=False, init=False, repr=False)
    _shutdown_errors: list[Exception] = field(default_factory=list, init=False, repr=False)
    _close_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _close_complete: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _close_started: bool = field(default=False, init=False, repr=False)
    _model_catalog_cache: tuple[float, OllamaCatalogSnapshot] | None = field(default=None, init=False, repr=False)
    _unlocked_users: set[str] = field(default_factory=set)
    _mutating_users: set[str] = field(default_factory=set, init=False, repr=False)
    _active_profile_operations: dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _stream_profile_operations: dict[tuple[str, int], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _global_mutation_active: bool = field(default=False, init=False, repr=False)
    _deferred_recovery_runs: dict[str, set[str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _last_chat_model: str = field(default="", init=False, repr=False)
    _known_chat_models: set[str] = field(default_factory=set, init=False, repr=False)

    @classmethod
    def create(cls, config: AppConfig | None = None) -> "AtlasBackendService":
        resolved = config or load_config()
        service = cls(
            config=resolved,
            app=build_chat_application(resolved),
            run_store=RunStore(resolved),
            run_hub=RunHub(),
        )
        service._recover_incomplete_runs()
        service._ensure_worker_started()
        return service

    def close(self) -> None:
        self._ensure_runtime_state()
        with self._close_lock:
            if self._close_complete.is_set():
                return
            if self._close_started:
                close_complete = self._close_complete
            else:
                self._close_started = True
                close_complete = None

        if close_complete is not None:
            close_complete.wait()
            return

        try:
            self._close_once()
        finally:
            self._close_complete.set()

    def begin_shutdown(self) -> None:
        """Quiesce active work without closing resources used by HTTP requests."""

        self._ensure_runtime_state()
        with self._shutdown_lock:
            if self._shutdown_complete.is_set():
                return
            if self._shutdown_started:
                shutdown_complete = self._shutdown_complete
            else:
                self._shutdown_started = True
                shutdown_complete = None

        if shutdown_complete is not None:
            shutdown_complete.wait()
            return

        try:
            self._begin_shutdown_once()
        finally:
            self._shutdown_complete.set()

    def _begin_shutdown_once(self) -> None:
        error_message = "Atlas shut down while this run was active."
        cleanup_errors: list[Exception] = []

        def attempt(callback) -> None:
            try:
                callback()
            except Exception as exc:  # pragma: no cover - defensive best-effort cleanup
                cleanup_errors.append(exc)

        active_run_id: str | None = None
        pending_run_ids: list[str] = []
        with self._control_lock:
            self._shutdown_requested = True
            active_run_id = self._active_run_id
            pending_run_ids = [job.run_id for job in self._pending_runs]
            self._pending_runs.clear()
            if active_run_id:
                self._cancelled_runs.add(active_run_id)
        for run_id in pending_run_ids:
            attempt(
                lambda run_id=run_id: self._fail_run_with_terminal_event(
                    run_id,
                    error=error_message,
                )
            )
        if active_run_id:
            attempt(lambda: self.run_store.mark_run_cancelling(active_run_id))
            attempt(self.app.llm_provider.abort_active_requests)
        attempt(get_model_pull_manager().shutdown)
        attempt(shutdown_runner)
        self._worker_wakeup.set()
        self._shutdown_errors.extend(cleanup_errors)

    def _close_once(self) -> None:
        self.begin_shutdown()
        error_message = "Atlas shut down while this run was active."
        cleanup_errors = list(self._shutdown_errors)

        def attempt(callback) -> None:
            try:
                callback()
            except Exception as exc:  # pragma: no cover - defensive best-effort cleanup
                cleanup_errors.append(exc)

        with self._control_lock:
            active_run_id = self._active_run_id
        worker = self._worker_thread
        if (
            worker is not None
            and worker is not threading.current_thread()
            and worker.is_alive()
        ):
            worker.join(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        if worker is not None and not worker.is_alive():
            self._worker_thread = None
        elif worker is not None and worker.is_alive():
            if active_run_id:
                attempt(
                    lambda: self._fail_run_with_terminal_event(
                        active_run_id,
                        error=error_message,
                    )
                )
            # The desktop shell has a bounded process-exit fallback. Keep
            # database and memory resources open for an uncooperative daemon
            # worker rather than closing them underneath code that may still
            # resume before process termination.
            if cleanup_errors:
                raise RuntimeError("Atlas backend did not close cleanly.") from cleanup_errors[0]
            return

        attempt(self.app.close)
        flush_pending_events = getattr(self.run_store, "flush_pending_events", None)
        if callable(flush_pending_events):
            attempt(flush_pending_events)
        self._unlocked_users.clear()
        lock_all_user_keys = getattr(self.run_store, "lock_all_user_keys", None)
        if callable(lock_all_user_keys):
            attempt(lock_all_user_keys)

        if cleanup_errors:
            raise RuntimeError("Atlas backend did not close cleanly.") from cleanup_errors[0]

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "product": "Atlas Chat"}

    def status(self) -> dict[str, Any]:
        self._ensure_runtime_state()
        with self._control_lock:
            busy = self._active_run_id is not None or bool(self._pending_runs)
        protected_locally = application_secret_protection_available()
        return {
            "status": "ok",
            "product_name": "Atlas Chat",
            "version": atlas_version(),
            "backend": "Atlas Chat local runtime",
            "default_chat_temperature": self.config.chat_temperature,
            "chat_temperature": self.config.chat_temperature,
            "embed_model": self.config.embed_model,
            "ollama_url": self.config.ollama_url,
            "chat_provider": getattr(self.config, "chat_provider", "ollama"),
            "chat_provider_label": chat_provider_label(getattr(self.config, "chat_provider", "ollama")),
            "chat_base_url": getattr(self.config, "chat_base_url", self.config.ollama_url),
            "runtime_mode": "chat-only",
            "compaction_timeout_seconds": self._compaction_timeout_seconds(),
            "busy": busy,
            "security": {
                "profile_key_protection": local_secret_storage_label(),
                "run_artifacts_encrypted_at_rest": protected_locally,
                "run_index_encrypted_at_rest": protected_locally,
                "packaged_logs_default": "off",
                "sqlite_encrypted_at_rest": protected_locally and sqlcipher_enabled(),
                "sqlite_paths": [
                    str(self.config.langgraph_checkpoint_db),
                    str(self.config.mem0_history_db),
                ],
                "vector_store": "local-qdrant",
                "vector_store_encrypted_at_rest": protected_locally and sqlcipher_enabled(),
                "vector_store_path": str(self.config.qdrant_path),
            },
        }

    def _ensure_runtime_state(self) -> None:
        if not hasattr(self, "_control_lock") or self._control_lock is None:
            self._control_lock = threading.Lock()
        if not hasattr(self, "_active_run_id"):
            self._active_run_id = None
        if not hasattr(self, "_cancelled_runs") or self._cancelled_runs is None:
            self._cancelled_runs = set()
        if not hasattr(self, "_pending_runs") or self._pending_runs is None:
            self._pending_runs = []
        if not hasattr(self, "_worker_wakeup") or self._worker_wakeup is None:
            self._worker_wakeup = threading.Event()
        if not hasattr(self, "_shutdown_requested"):
            self._shutdown_requested = False
        if not hasattr(self, "_shutdown_lock") or self._shutdown_lock is None:
            self._shutdown_lock = threading.Lock()
        if not hasattr(self, "_shutdown_complete") or self._shutdown_complete is None:
            self._shutdown_complete = threading.Event()
        if not hasattr(self, "_shutdown_started"):
            self._shutdown_started = False
        if not hasattr(self, "_shutdown_errors") or self._shutdown_errors is None:
            self._shutdown_errors = []
        if not hasattr(self, "_close_lock") or self._close_lock is None:
            self._close_lock = threading.Lock()
        if not hasattr(self, "_close_complete") or self._close_complete is None:
            self._close_complete = threading.Event()
        if not hasattr(self, "_close_started"):
            self._close_started = False
        if not hasattr(self, "_model_catalog_cache"):
            self._model_catalog_cache = None
        if not hasattr(self, "_worker_thread"):
            self._worker_thread = None
        if not hasattr(self, "_unlocked_users") or self._unlocked_users is None:
            self._unlocked_users = set()
        if not hasattr(self, "_mutating_users") or self._mutating_users is None:
            self._mutating_users = set()
        if (
            not hasattr(self, "_active_profile_operations")
            or self._active_profile_operations is None
        ):
            self._active_profile_operations = {}
        if (
            not hasattr(self, "_stream_profile_operations")
            or self._stream_profile_operations is None
        ):
            self._stream_profile_operations = {}
        if not hasattr(self, "_global_mutation_active"):
            self._global_mutation_active = False
        if (
            not hasattr(self, "_deferred_recovery_runs")
            or self._deferred_recovery_runs is None
        ):
            self._deferred_recovery_runs = {}
        if not hasattr(self, "_last_chat_model"):
            self._last_chat_model = ""
        if not hasattr(self, "_known_chat_models") or self._known_chat_models is None:
            self._known_chat_models = set()

    def _ensure_worker_started(self) -> None:
        self._ensure_runtime_state()
        with self._close_lock:
            if self._close_started:
                raise RuntimeError("Atlas backend is shutting down.")
        with self._control_lock:
            if self._shutdown_requested:
                raise RuntimeError("Atlas backend is shutting down.")
            worker = self._worker_thread
            if worker is not None and worker.is_alive():
                return
            self._shutdown_requested = False
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="atlas-run-worker",
                daemon=True,
            )
            self._worker_thread.start()

    def _recover_incomplete_runs(self) -> None:
        self._ensure_runtime_state()
        fail_incomplete_runs = getattr(self.run_store, "fail_incomplete_runs", None)
        if not callable(fail_incomplete_runs):
            return
        list_incomplete_runs = getattr(self.run_store, "list_incomplete_runs", None)
        if callable(list_incomplete_runs):
            is_user_key_unlocked = getattr(
                self.run_store,
                "is_user_key_unlocked",
                None,
            )
            for item in list_incomplete_runs():
                user_id = str(item.get("user_id", "") or "").strip()
                run_id = str(item.get("run_id", "") or "").strip()
                user = self._lookup_user_record(user_id) if user_id else None
                key_unlocked = (
                    bool(is_user_key_unlocked(user_id))
                    if user_id and callable(is_user_key_unlocked)
                    else False
                )
                if (
                    user
                    and user.get("protection") == PASSWORD_PROTECTED
                    and not key_unlocked
                    and run_id
                ):
                    self._deferred_recovery_runs.setdefault(user_id, set()).add(
                        run_id
                    )
        fail_incomplete_runs(
            error=_INTERRUPTED_RUN_ERROR,
            skip_locked=True,
        )

    def _recover_deferred_runs_for_user(self, user_id: str) -> None:
        run_ids = set(self._deferred_recovery_runs.get(user_id, set()))
        if not run_ids:
            return
        fail_incomplete_runs = getattr(self.run_store, "fail_incomplete_runs", None)
        if not callable(fail_incomplete_runs):
            return
        fail_incomplete_runs(
            error=_INTERRUPTED_RUN_ERROR,
            user_id=user_id,
            run_ids=run_ids,
        )
        self._deferred_recovery_runs.pop(user_id, None)

    def _worker_loop(self) -> None:
        while True:
            self._worker_wakeup.wait()
            self._worker_wakeup.clear()

            while True:
                job = self._claim_next_job()
                if job is None:
                    with self._control_lock:
                        if self._shutdown_requested:
                            return
                    break

                try:
                    if job.mode == "chat":
                        self._execute_run(
                            run_id=job.run_id,
                            prompt=job.prompt,
                            user_id=job.user_id,
                            thread_id=job.thread_id,
                            chat_model=job.chat_model,
                            temperature=job.temperature,
                            reasoning_mode=job.reasoning_mode,
                            cross_chat_memory=job.cross_chat_memory,
                            auto_compact_long_chats=job.auto_compact_long_chats,
                            attachments=job.attachments,
                        )
                    elif job.mode == "compact":
                        self._execute_compact_run(
                            run_id=job.run_id,
                            user_id=job.user_id,
                            thread_id=job.thread_id,
                            chat_model=job.chat_model,
                        )
                    else:
                        raise RuntimeError(f"Unsupported queued run mode: {job.mode}")
                except Exception as exc:  # pragma: no cover - last-resort worker safety net
                    self._fail_run_from_exception(job.run_id, exc)
                finally:
                    with self._control_lock:
                        if self._active_run_id == job.run_id:
                            self._active_run_id = None
                        self._cancelled_runs.discard(job.run_id)
                    self._worker_wakeup.set()

    def _claim_next_job(self) -> _QueuedRunJob | None:
        with self._control_lock:
            if self._shutdown_requested or self._active_run_id is not None or not self._pending_runs:
                return None
            job = self._pending_runs.pop(0)
            self._active_run_id = job.run_id
            self.run_store.mark_run_running(job.run_id)
            return job

    def list_models(self) -> dict[str, Any]:
        catalog = self._get_model_catalog()
        context_window = self._ollama_context_window_payload(catalog)
        return {
            "default_temperature": self.config.chat_temperature,
            "temperature_presets": [
                {"label": f"{value:.1f}", "value": value}
                for value in _temperature_dropdown_values()
            ],
            "context_window_presets": list(OLLAMA_CONTEXT_WINDOW_PRESETS),
            "ollama_context_window": context_window,
            "ollama_online": catalog.ollama_online,
            "has_local_models": catalog.has_local_models,
            "provider": catalog.provider,
            "provider_label": catalog.provider_label,
            "provider_base_url": catalog.provider_base_url
            or getattr(self.config, "chat_base_url", None)
            or getattr(self.config, "ollama_url", "http://127.0.0.1:11434"),
            "provider_online": catalog.provider_online or catalog.ollama_online,
            "has_chat_models": catalog.has_local_models,
            "supports_context_window": catalog.supports_context_window,
            "supports_model_unload": catalog.supports_model_unload,
            "catalog_source": catalog.source,
            "loaded_models": self._loaded_ollama_models(catalog),
            "models": [item.name for item in catalog.models],
            "model_details": [item.to_dict() for item in catalog.models],
        }

    def get_provider_settings(self) -> dict[str, Any]:
        saved = load_provider_settings(self.config.data_dir)
        return self._provider_settings_payload(
            provider=getattr(self.config, "chat_provider", "ollama"),
            base_url=getattr(self.config, "chat_base_url", self.config.ollama_url),
            has_api_key=bool(getattr(self.config, "chat_api_key", None)),
            key_unavailable=saved.get("api_key_unavailable") == "true",
            base_url_invalid=saved.get("base_url_invalid") == "true",
        )

    def save_provider_settings(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str | None,
        preserve_existing_key: bool,
    ) -> dict[str, Any]:
        resolved_provider = normalize_chat_provider(provider)
        if provider.strip().lower() not in CHAT_PROVIDER_LABELS:
            raise RuntimeError("Choose a supported local model provider.")
        resolved_base_url = normalize_local_provider_base_url(
            base_url.strip() or self._provider_default_base_url(resolved_provider)
        )
        result = save_provider_settings(
            self.config.data_dir,
            provider=resolved_provider,
            base_url=resolved_base_url,
            api_key=api_key,
            preserve_existing_key=preserve_existing_key,
        )
        return self._provider_settings_payload(
            provider=resolved_provider,
            base_url=resolved_base_url,
            has_api_key=bool(result.get("has_api_key")),
            restart_required=True,
        )

    def clear_provider_api_key(self) -> dict[str, Any]:
        clear_provider_api_key(self.config.data_dir)
        payload = self.get_provider_settings()
        payload["has_api_key"] = False
        payload["restart_required"] = True
        return payload

    def start_model_pull(self, *, model: str) -> dict[str, Any]:
        if not is_ollama_chat_provider(getattr(self.config, "chat_provider", "ollama")):
            raise RuntimeError("Model downloads are available when Ollama is selected.")
        return get_model_pull_manager().start(
            ollama_url=getattr(self.config, "chat_base_url", self.config.ollama_url),
            model=model,
        )

    def list_model_pulls(self) -> list[dict[str, Any]]:
        return get_model_pull_manager().list()

    def get_model_pull(self, pull_id: str) -> dict[str, Any]:
        return get_model_pull_manager().get(pull_id)

    def cancel_model_pull(self, pull_id: str) -> dict[str, Any]:
        return get_model_pull_manager().cancel(pull_id)

    def _provider_settings_payload(
        self,
        *,
        provider: str,
        base_url: str,
        has_api_key: bool,
        key_unavailable: bool = False,
        base_url_invalid: bool = False,
        restart_required: bool = False,
    ) -> dict[str, Any]:
        resolved_provider = normalize_chat_provider(provider)
        return {
            "provider": resolved_provider,
            "provider_label": chat_provider_label(resolved_provider),
            "base_url": base_url,
            "has_api_key": has_api_key,
            "api_key_unavailable": key_unavailable,
            "base_url_invalid": base_url_invalid,
            "secure_key_storage_available": application_secret_protection_available(),
            "restart_required": restart_required,
            "providers": [
                {
                    "id": provider_id,
                    "label": label,
                    "default_base_url": self._provider_default_base_url(provider_id),
                }
                for provider_id, label in CHAT_PROVIDER_LABELS.items()
            ],
        }

    def _provider_default_base_url(self, provider: str) -> str:
        if is_ollama_chat_provider(provider):
            return getattr(self.config, "ollama_url", DEFAULT_OLLAMA_URL)
        return OPENAI_COMPATIBLE_PROVIDER_DEFAULT_URLS[provider]

    def set_ollama_context_window(self, *, context_window: int | None) -> dict[str, Any]:
        setter = getattr(self.app.llm_provider, "set_ollama_context_window", None)
        if not callable(setter):
            raise RuntimeError("This runtime cannot change Ollama context windows.")
        with self._control_lock:
            self._ensure_model_runtime_idle_locked("change the context window")
            saved_value = setter(context_window)
        catalog = self._get_model_catalog(ttl_seconds=0.0)
        return {
            "configured_context_window": saved_value,
            "ollama_context_window": self._ollama_context_window_payload(catalog),
        }

    def unload_ollama_model(self, *, model: str) -> dict[str, Any]:
        unloader = getattr(self.app.llm_provider, "unload_model", None)
        if not callable(unloader):
            raise RuntimeError("This runtime cannot stop Ollama models.")
        with self._control_lock:
            self._ensure_model_runtime_idle_locked("stop a model")
            payload = unloader(model)
        payload["loaded_models"] = self._loaded_ollama_models(self._get_model_catalog())
        return payload

    def _ensure_model_runtime_idle_locked(self, action: str) -> None:
        if self._active_run_id is not None or self._pending_runs:
            raise RuntimeError(
                f"Wait for active and queued runs to finish before Atlas can {action}."
            )

    def _loaded_ollama_models(self, catalog: OllamaCatalogSnapshot | None = None) -> list[str]:
        provider = getattr(getattr(self, "app", None), "llm_provider", None)
        getter = getattr(provider, "loaded_models", None)
        if not callable(getter):
            return []
        allowed_models = {item.name for item in catalog.models} if catalog is not None else None
        try:
            loaded: list[str] = []
            for model in getter():
                model_name = str(model).strip()
                if not model_name:
                    continue
                if allowed_models is not None and model_name not in allowed_models:
                    continue
                loaded.append(model_name)
            return loaded
        except Exception:
            return []

    def _prepare_ollama_model_switch(self, target_model: str) -> list[str]:
        if not is_ollama_chat_provider(getattr(getattr(self, "config", None), "chat_provider", "ollama")):
            return []
        target = str(target_model or "").strip()
        if not target:
            return []
        provider = getattr(getattr(self, "app", None), "llm_provider", None)
        unloader = getattr(provider, "unload_model", None)
        if not callable(unloader):
            return []

        known_models = sorted(
            str(model).strip()
            for model in getattr(self, "_known_chat_models", set())
            if str(model).strip()
        )
        candidates = [
            model
            for model in self._loaded_ollama_models()
            if not _looks_like_embedding_model(model)
        ]
        last_model = str(getattr(self, "_last_chat_model", "") or "").strip()
        if last_model:
            candidates.append(last_model)
        candidates.extend(known_models)

        unloaded: list[str] = []
        target_key = target.casefold()
        seen: set[str] = set()
        for model in candidates:
            model_name = str(model or "").strip()
            model_key = model_name.casefold()
            if not model_name or model_key == target_key or model_key in seen:
                continue
            seen.add(model_key)
            try:
                unloader(model_name)
            except Exception:
                continue
            unloaded.append(model_name)
        return unloaded

    def _remember_ollama_chat_model(self, model: str) -> None:
        model_name = str(model or "").strip()
        if not model_name:
            return
        if not hasattr(self, "_known_chat_models") or self._known_chat_models is None:
            self._known_chat_models = set()
        self._known_chat_models.add(model_name)
        self._last_chat_model = model_name

    def _cleanup_ollama_model_after_resource_failure(self, chat_model: str, error_message: str) -> bool:
        if not is_ollama_chat_provider(getattr(getattr(self, "config", None), "chat_provider", "ollama")):
            return False
        model_name = str(chat_model or "").strip()
        if not model_name or not _looks_like_ollama_resource_failure(error_message):
            return False
        provider = getattr(getattr(self, "app", None), "llm_provider", None)
        unloader = getattr(provider, "unload_model", None)
        if not callable(unloader):
            return False
        try:
            unloader(model_name)
        except Exception:
            return False
        return True

    def _ollama_context_window_payload(self, catalog: OllamaCatalogSnapshot) -> dict[str, Any]:
        if not catalog.supports_context_window:
            return {
                "configured_context_window": None,
                "effective_context_window": None,
                "source": "provider",
            }
        provider = getattr(getattr(self, "app", None), "llm_provider", None)
        configured_getter = getattr(provider, "ollama_context_window", None)
        configured_value = configured_getter() if callable(configured_getter) else None
        configured_window = int(configured_value or 0) or None
        detected_window = None
        if provider is not None:
            for item in catalog.models:
                try:
                    detected_window = int(provider.effective_context_window(item.name) or 0) or None
                except Exception:
                    detected_window = None
                if detected_window:
                    break
        return {
            "configured_context_window": configured_window,
            "effective_context_window": configured_window or detected_window,
            "source": "configured" if configured_window else "ollama",
        }

    def discovery(self) -> dict[str, Any]:
        return build_discovery_report(self.config, self._get_model_catalog())

    def list_users(self) -> list[dict[str, Any]]:
        self._ensure_runtime_state()
        items = self.run_store.list_users()
        items.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return [self._sanitize_user_summary(item) for item in items]

    def create_user(self, *, user_id: str, password: str | None = None) -> dict[str, Any]:
        self._ensure_runtime_state()
        resolved = user_id.strip()
        if not resolved:
            raise RuntimeError("User id is required.")
        resolved_password = (password or "").strip() or None
        with self._profile_mutation(resolved, action="create"):
            item = self.run_store.create_user(resolved, password=resolved_password)
            if item.get("protection") == PASSWORD_PROTECTED:
                self.run_store.unlock_user_key(resolved, password=resolved_password)
                self._unlocked_users.add(resolved)
            return self._sanitize_user_summary(item)

    def unlock_user(self, *, user_id: str, password: str | None = None) -> dict[str, Any]:
        self._ensure_runtime_state()
        resolved = user_id.strip()
        if not resolved:
            raise RuntimeError("User id is required.")
        with self._profile_mutation(resolved, action="unlock"):
            user = self._lookup_user_record(resolved)
            if not user:
                raise RuntimeError(f"User not found: {resolved}")
            if user.get("protection") == PASSWORD_PROTECTED:
                resolved_password = (password or "").strip()
                if not resolved_password:
                    raise RuntimeError("Password is required for this user.")
                try:
                    self.run_store.unlock_user_key(
                        resolved,
                        password=resolved_password,
                    )
                    self._recover_deferred_runs_for_user(resolved)
                except Exception:
                    self._unlocked_users.discard(resolved)
                    self.run_store.lock_user_key(resolved)
                    raise
                else:
                    self._unlocked_users.add(resolved)
            else:
                self.run_store.unlock_user_key(resolved)
            return self._sanitize_user_summary(user)

    def lock_user(self, *, user_id: str) -> dict[str, Any]:
        self._ensure_runtime_state()
        resolved = user_id.strip()
        if not resolved:
            raise RuntimeError("User id is required.")
        with self._profile_mutation(resolved, action="lock"):
            user = self._lookup_user_record(resolved)
            if not user:
                raise RuntimeError(f"User not found: {resolved}")
            self._unlocked_users.discard(resolved)
            self.run_store.lock_user_key(resolved)
            return self._sanitize_user_summary(user)

    def add_memory(self, *, user_id: str, text: str) -> dict[str, Any]:
        resolved_user_id = user_id.strip()
        resolved_text = text.strip()
        if not resolved_user_id:
            raise RuntimeError("User id is required.")
        if not resolved_text:
            raise RuntimeError("Memory text is required.")
        with self._profile_operation(resolved_user_id):
            response = self.app.memory_service.add(
                MemoryRecord(claim_id=f"manual:{uuid4()}", text=resolved_text),
                user_id=resolved_user_id,
                metadata={"source": "manual", "kind": "memory_note"},
            )
            results = response.get("results", []) if isinstance(response, dict) else []
            memory_id = ""
            if results and isinstance(results[0], dict):
                memory_id = str(results[0].get("id", "")).strip()
            self.run_store.upsert_user(resolved_user_id)
            return {
                "status": "ok",
                "user_id": resolved_user_id,
                "memory_id": memory_id,
                "text": resolved_text,
            }

    def delete_memory(self, *, user_id: str, memory_id: str) -> dict[str, Any]:
        resolved_user_id = user_id.strip()
        resolved_memory_id = memory_id.strip()
        if not resolved_user_id:
            raise RuntimeError("User id is required.")
        if not resolved_memory_id:
            raise RuntimeError("Memory id is required.")
        with self._profile_operation(resolved_user_id):
            self.app.memory_service.delete(
                resolved_memory_id,
                user_id=resolved_user_id,
            )
            return {
                "status": "ok",
                "user_id": resolved_user_id,
                "memory_id": resolved_memory_id,
            }

    def list_threads(self, *, user_id: str | None = None) -> list[dict[str, Any]]:
        self._ensure_runtime_state()
        if user_id:
            with self._profile_operation(user_id):
                return self._list_threads_data(user_id=user_id)

        # The unscoped workspace view spans every currently accessible profile.
        # Pin each profile before re-reading the stores so a concurrent lock or
        # delete cannot complete and then have stale private metadata returned.
        initial_items = self.run_store.list_threads(user_id=None)
        initial_checkpoint_threads = self._list_checkpoint_threads()
        candidate_user_ids = {
            str(item.get("user_id", "") or "").strip()
            for item in initial_items
        }
        candidate_user_ids.update(
            checkpoint_user_id
            for checkpoint_user_id, _thread_id in initial_checkpoint_threads
        )
        candidate_user_ids.discard("")

        with ExitStack() as profile_operations:
            accessible_user_ids: set[str] = set()
            for candidate_user_id in sorted(candidate_user_ids):
                if not self._is_profile_accessible(candidate_user_id):
                    continue
                try:
                    profile_operations.enter_context(
                        self._profile_operation(candidate_user_id)
                    )
                except RuntimeError:
                    with self._control_lock:
                        global_mutation_active = self._global_mutation_active
                        profile_mutating = candidate_user_id in self._mutating_users
                    if global_mutation_active:
                        raise
                    if profile_mutating or not self._is_profile_accessible(
                        candidate_user_id
                    ):
                        continue
                    raise
                accessible_user_ids.add(candidate_user_id)

            return self._list_threads_data(
                user_id=None,
                accessible_user_ids=accessible_user_ids,
            )

    def _list_threads_data(
        self,
        *,
        user_id: str | None,
        accessible_user_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        items = self.run_store.list_threads(user_id=user_id)
        if not user_id:
            items = [
                item
                for item in items
                if str(item.get("user_id", "") or "").strip()
                in (accessible_user_ids or set())
            ]
        checkpoint_threads = self._list_checkpoint_threads()
        seen = {(item["user_id"], item["thread_id"]) for item in items}
        for checkpoint_user_id, thread_id in checkpoint_threads:
            key = (checkpoint_user_id, thread_id)
            if key in seen:
                continue
            if user_id and checkpoint_user_id != user_id:
                continue
            if not user_id and checkpoint_user_id not in (
                accessible_user_ids or set()
            ):
                continue
            items.append(
                {
                    "user_id": checkpoint_user_id,
                    "thread_id": thread_id,
                    "title": thread_id,
                    "chat_model": "",
                    "temperature": None,
                    "last_mode": "chat",
                    "updated_at": "",
                    "last_prompt": "",
                    "last_run_id": "",
                }
            )
            seen.add(key)
        items.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return items

    def rename_thread(self, *, user_id: str, thread_id: str, title: str) -> dict[str, Any]:
        resolved_user_id = user_id.strip()
        resolved_thread_id = thread_id.strip()
        if not resolved_user_id:
            raise RuntimeError("User id is required.")
        if not resolved_thread_id:
            raise RuntimeError("Thread id is required.")
        with self._profile_operation(resolved_user_id):
            return self.run_store.rename_thread(
                user_id=resolved_user_id,
                thread_id=resolved_thread_id,
                title=title,
            )

    def duplicate_thread(self, *, user_id: str, thread_id: str) -> dict[str, Any]:
        resolved_user_id = user_id.strip()
        resolved_thread_id = thread_id.strip()
        if not resolved_user_id:
            raise RuntimeError("User id is required.")
        if not resolved_thread_id:
            raise RuntimeError("Thread id is required.")
        with self._profile_operation(resolved_user_id):
            return self._duplicate_thread_data(
                user_id=resolved_user_id,
                thread_id=resolved_thread_id,
            )

    def _duplicate_thread_data(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> dict[str, Any]:
        source_thread = self.run_store.get_thread(user_id=user_id, thread_id=thread_id)
        if not source_thread:
            raise RuntimeError(f"Thread not found: {thread_id}")

        snapshot = self._get_snapshot(user_id=user_id, thread_id=thread_id)
        history_messages = list(snapshot.values.get("messages", []))
        duplicate_thread_id = f"atlas-{datetime.now(UTC).strftime('%Y-%m-%d-%H-%M-%S')}-{uuid4().hex[:4]}"
        duplicate_title = _duplicate_thread_title(source_thread.get("title") or thread_id)
        duplicate_session_id = scoped_thread_id(user_id, duplicate_thread_id)
        timeline_events = _persistent_thread_timeline_events(snapshot.values.get("timeline_events", []))

        if history_messages:
            self.app.graph.update_state(
                {"configurable": {"thread_id": duplicate_session_id}},
                {
                    "messages": history_messages,
                    "thread_summary": str(snapshot.values.get("thread_summary", "") or ""),
                    "compacted_message_count": int(snapshot.values.get("compacted_message_count", 0) or 0),
                    "detected_context_window": int(snapshot.values.get("detected_context_window", 0) or 0),
                    "timeline_events": timeline_events,
                },
                as_node="persist",
            )

        thread_record = self.run_store.upsert_thread(
            user_id=user_id,
            thread_id=duplicate_thread_id,
            title=duplicate_title,
            chat_model=str(source_thread.get("chat_model", "") or ""),
            temperature=self._record_temperature(source_thread, fallback_on_missing=self.config.chat_temperature),
            last_mode=str(source_thread.get("last_mode", "chat") or "chat"),
            last_prompt=str(source_thread.get("last_prompt", "") or ""),
        )
        replace_search_messages = getattr(self.run_store, "replace_thread_search_messages", None)
        if callable(replace_search_messages):
            replace_search_messages(
                user_id=user_id,
                thread_id=duplicate_thread_id,
                title=str(thread_record.get("title", "") or duplicate_title),
                chat_model=str(thread_record.get("chat_model", "") or ""),
                updated_at=str(thread_record.get("updated_at", "") or ""),
                messages=_messages_to_search_index_items(history_messages),
            )
        return thread_record

    def branch_thread(self, *, user_id: str, thread_id: str, after_message_count: int) -> dict[str, Any]:
        resolved_user_id = user_id.strip()
        resolved_thread_id = thread_id.strip()
        if not resolved_user_id:
            raise RuntimeError("User id is required.")
        if not resolved_thread_id:
            raise RuntimeError("Thread id is required.")
        with self._profile_operation(resolved_user_id):
            return self._branch_thread_data(
                user_id=resolved_user_id,
                thread_id=resolved_thread_id,
                after_message_count=after_message_count,
            )

    def _branch_thread_data(
        self,
        *,
        user_id: str,
        thread_id: str,
        after_message_count: int,
    ) -> dict[str, Any]:
        source_thread = self.run_store.get_thread(user_id=user_id, thread_id=thread_id)
        if not source_thread:
            raise RuntimeError(f"Thread not found: {thread_id}")

        snapshot = self._get_snapshot(user_id=user_id, thread_id=thread_id)
        history_messages = list(snapshot.values.get("messages", []))
        branch_count = max(0, min(int(after_message_count or 0), len(history_messages)))
        branch_messages = history_messages[:branch_count]
        branch_thread_id = f"atlas-{datetime.now(UTC).strftime('%Y-%m-%d-%H-%M-%S')}-{uuid4().hex[:4]}"
        branch_title = _branch_thread_title(source_thread.get("title") or thread_id)
        branch_session_id = scoped_thread_id(user_id, branch_thread_id)

        if branch_messages:
            self.app.graph.update_state(
                {"configurable": {"thread_id": branch_session_id}},
                {
                    "messages": branch_messages,
                    "thread_summary": "",
                    "compacted_message_count": 0,
                    "detected_context_window": 0,
                    "timeline_events": [],
                },
                as_node="persist",
            )

        thread_record = self.run_store.upsert_thread(
            user_id=user_id,
            thread_id=branch_thread_id,
            title=branch_title,
            chat_model=str(source_thread.get("chat_model", "") or ""),
            temperature=self._record_temperature(source_thread, fallback_on_missing=self.config.chat_temperature),
            last_mode="chat",
            last_prompt=_latest_user_text({"messages": branch_messages}) if branch_messages else "",
        )
        replace_search_messages = getattr(self.run_store, "replace_thread_search_messages", None)
        if callable(replace_search_messages):
            replace_search_messages(
                user_id=user_id,
                thread_id=branch_thread_id,
                title=str(thread_record.get("title", "") or branch_title),
                chat_model=str(thread_record.get("chat_model", "") or ""),
                updated_at=str(thread_record.get("updated_at", "") or ""),
                messages=_messages_to_search_index_items(branch_messages),
            )
        return thread_record

    def get_thread_context_usage(
        self, *, user_id: str, thread_id: str
    ) -> dict[str, Any]:
        with self._profile_operation(user_id):
            return self._get_thread_context_usage_data(
                user_id=user_id,
                thread_id=thread_id,
            )

    def _get_thread_context_usage_data(
        self, *, user_id: str, thread_id: str
    ) -> dict[str, Any]:
        snapshot = self._get_snapshot(user_id=user_id, thread_id=thread_id)
        messages = list(snapshot.values.get("messages", []))
        thread_summary = str(snapshot.values.get("thread_summary", "") or "")
        compacted_message_count = int(snapshot.values.get("compacted_message_count", 0) or 0)
        detected_context_window = int(snapshot.values.get("detected_context_window", 0) or 0)

        chat_model = ""
        try:
            thread_record = self.run_store.get_thread(user_id=user_id, thread_id=thread_id)
        except Exception:
            thread_record = None
        if thread_record:
            chat_model = str(thread_record.get("chat_model", "") or "")
            if not chat_model:
                chat_model = self._thread_last_run_model(thread_record)

        effective_context_window = 0
        if chat_model:
            try:
                effective_context_window = int(
                    self.app.llm_provider.effective_context_window(chat_model) or 0
                )
            except Exception:
                effective_context_window = 0
        if effective_context_window <= 0:
            effective_context_window = detected_context_window

        summary_tokens = self._count_messages_tokens(
            model=chat_model,
            messages=[SystemMessage(content=f"Conversation summary from earlier in this thread:\n{thread_summary}")]
            if thread_summary.strip()
            else [],
        )
        raw_message_tokens = self._count_messages_tokens(
            model=chat_model,
            messages=messages[max(0, min(compacted_message_count, len(messages))) :],
        )
        representation_tokens = summary_tokens + raw_message_tokens

        auto_compact_ratio = 0.72
        auto_compact_threshold = _auto_compact_threshold(effective_context_window)
        auto_compact_margin_tokens = (
            max(0, auto_compact_threshold - representation_tokens)
            if auto_compact_threshold > 0
            else 0
        )

        return {
            "thread_id": thread_id,
            "user_id": user_id,
            "chat_model": chat_model,
            "context_window": effective_context_window,
            "auto_compact_ratio": auto_compact_ratio,
            "auto_compact_threshold": auto_compact_threshold,
            "auto_compact_margin_tokens": auto_compact_margin_tokens,
            "representation_tokens": representation_tokens,
            "summary_tokens": summary_tokens,
            "raw_message_tokens": raw_message_tokens,
            "compacted_message_count": compacted_message_count,
            "recent_raw_message_count": max(
                0,
                len(messages) - max(0, min(compacted_message_count, len(messages))),
            ),
            "message_count": len(messages),
        }

    def get_thread_history(self, *, user_id: str, thread_id: str) -> list[dict[str, Any]]:
        with self._profile_operation(user_id):
            return self._get_thread_history_data(
                user_id=user_id,
                thread_id=thread_id,
            )

    def _get_thread_history_data(
        self, *, user_id: str, thread_id: str
    ) -> list[dict[str, Any]]:
        snapshot = self._get_snapshot(user_id=user_id, thread_id=thread_id)
        history: list[dict[str, Any]] = []
        timeline_events = _sorted_thread_timeline_events(
            _persistent_thread_timeline_events(snapshot.values.get("timeline_events", []))
        )
        pending_events = list(timeline_events)
        for index, message in enumerate(snapshot.values.get("messages", []), start=1):
            role = "system"
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            content, attachments = _message_to_history_parts(message)
            history.append(
                {
                    "role": role,
                    "content": content,
                    "attachments": attachments,
                    "history_index": index - 1,
                }
            )
            while pending_events and int(pending_events[0].get("after_message_count", 0) or 0) == index:
                history.append(_timeline_event_to_history_item(pending_events.pop(0)))
        while pending_events:
            history.append(_timeline_event_to_history_item(pending_events.pop(0)))
        return history

    def search_threads(
        self,
        *,
        user_id: str,
        query: str,
        current_thread_id: str | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        if not user_id.strip():
            raise RuntimeError("User id is required.")
        with self._profile_operation(user_id):
            return self._search_threads_data(
                user_id=user_id,
                query=query,
                current_thread_id=current_thread_id,
                limit=limit,
            )

    def _search_threads_data(
        self,
        *,
        user_id: str,
        query: str,
        current_thread_id: str | None,
        limit: int,
    ) -> dict[str, Any]:
        normalized_query = _normalize_search_query(query)
        if len(normalized_query) < 2:
            return {
                "query": query.strip(),
                "current_thread_id": current_thread_id or "",
                "current_thread_results": [],
                "other_thread_results": [],
            }

        thread_items = self.list_threads(user_id=user_id)
        indexed_message_matches = self._search_indexed_messages(
            user_id=user_id,
            query=query,
            limit=max(50, limit * 12),
        )
        current_results: list[dict[str, Any]] = []
        other_results: list[dict[str, Any]] = []

        for thread in thread_items:
            thread_id = str(thread.get("thread_id", "") or "").strip()
            if not thread_id:
                continue
            matches = self._search_thread_matches(
                user_id=user_id,
                thread=thread,
                normalized_query=normalized_query,
                indexed_messages=indexed_message_matches.get(thread_id, []) if indexed_message_matches is not None else None,
            )
            if thread_id == (current_thread_id or ""):
                current_results.extend(matches)
            else:
                other_results.extend(matches)

        return {
            "query": query.strip(),
            "current_thread_id": current_thread_id or "",
            "current_thread_results": _sort_search_results(current_results, current_thread=True)[: max(1, limit)],
            "other_thread_results": _sort_search_results(other_results, current_thread=False)[: max(1, limit * 2)],
        }

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._accessible_run_operation(run_id) as artifact:
            user_id = str(artifact.get("user_id", "") or "").strip() or None
            thread_id = str(artifact.get("thread_id", "") or "").strip()
            if not thread_id:
                artifact["diagnostics"] = _build_run_diagnostics(artifact)
                return artifact
            snapshot = self._get_snapshot(user_id=user_id, thread_id=thread_id)
            artifact["thread_summary"] = str(snapshot.values.get("thread_summary", "") or "")
            artifact["compacted_message_count"] = int(snapshot.values.get("compacted_message_count", 0) or 0)
            artifact["detected_context_window"] = int(snapshot.values.get("detected_context_window", 0) or 0)
            artifact["diagnostics"] = _build_run_diagnostics(artifact)
            return artifact

    def list_thread_runs(self, *, user_id: str, thread_id: str) -> list[dict[str, Any]]:
        with self._profile_operation(user_id):
            artifacts = self.run_store.list_runs_for_thread(user_id=user_id, thread_id=thread_id)
            for artifact in artifacts:
                artifact["diagnostics"] = _build_run_diagnostics(artifact)
            return artifacts

    def _search_thread_matches(
        self,
        *,
        user_id: str,
        thread: dict[str, Any],
        normalized_query: str,
        indexed_messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        thread_id = str(thread.get("thread_id", "") or "").strip()
        title = str(thread.get("title", "") or thread_id).strip() or thread_id
        last_prompt = str(thread.get("last_prompt", "") or "").strip()
        updated_at = str(thread.get("updated_at", "") or "")
        chat_model = str(thread.get("chat_model", "") or "")

        results: list[dict[str, Any]] = []
        title_match = normalized_query in title.casefold()
        prompt_match = bool(last_prompt) and normalized_query in last_prompt.casefold()
        if title_match or prompt_match:
            results.append(
                {
                    "thread_id": thread_id,
                    "thread_title": title,
                    "chat_model": chat_model,
                    "updated_at": updated_at,
                    "match_type": "thread",
                    "role": None,
                    "history_index": None,
                    "snippet": _build_thread_search_snippet(title=title, last_prompt=last_prompt, normalized_query=normalized_query),
                }
            )

        if indexed_messages is None:
            history = self.get_thread_history(user_id=user_id, thread_id=thread_id)
            indexed_messages = [
                {
                    "role": str(item.get("role", "") or ""),
                    "content": str(item.get("content", "") or ""),
                    "history_index": item.get("history_index", history_index),
                    "updated_at": updated_at,
                    "chat_model": chat_model,
                }
                for history_index, item in enumerate(history)
            ]
        for item in indexed_messages:
            role = str(item.get("role", "") or "")
            if role not in {"user", "assistant"}:
                continue
            content = str(item.get("content", "") or "").strip()
            if not content or normalized_query not in content.casefold():
                continue
            results.append(
                {
                    "thread_id": thread_id,
                    "thread_title": title,
                    "chat_model": str(item.get("chat_model", "") or chat_model),
                    "updated_at": str(item.get("updated_at", "") or updated_at),
                    "match_type": "message",
                    "role": role,
                    "history_index": item.get("history_index"),
                    "snippet": _build_search_snippet(content, normalized_query),
                }
            )
        return results

    def _search_indexed_messages(self, *, user_id: str, query: str, limit: int) -> dict[str, list[dict[str, Any]]] | None:
        search_messages = getattr(self.run_store, "search_messages", None)
        if not callable(search_messages):
            return None
        rows = search_messages(user_id=user_id, query=query, limit=limit)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            thread_id = str(row.get("thread_id", "") or "").strip()
            if not thread_id:
                continue
            grouped.setdefault(thread_id, []).append(row)
        return grouped

    def subscribe(self, run_id: str) -> queue.Queue[RunEvent]:
        self._ensure_runtime_state()
        operation = self._accessible_run_operation(run_id)
        operation.__enter__()
        try:
            subscriber = self.run_hub.subscribe(run_id)
            with self._control_lock:
                self._stream_profile_operations[(run_id, id(subscriber))] = operation
            return subscriber
        except BaseException:
            operation.__exit__(None, None, None)
            raise

    def unsubscribe(self, run_id: str, subscriber: queue.Queue[RunEvent]) -> None:
        self._ensure_runtime_state()
        try:
            self.run_hub.unsubscribe(run_id, subscriber)
        finally:
            with self._control_lock:
                operation = self._stream_profile_operations.pop(
                    (run_id, id(subscriber)),
                    None,
                )
            if operation is not None:
                operation.__exit__(None, None, None)

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        self._ensure_worker_started()
        artifact = self._get_accessible_run(run_id)
        already_cancelling = False

        with self._control_lock:
            try:
                artifact = self.run_store.get_run(run_id)
            except Exception:
                pass
            queued_index = next(
                (index for index, job in enumerate(self._pending_runs) if job.run_id == run_id),
                None,
            )
            active_match = self._active_run_id == run_id

            if queued_index is not None and not active_match:
                self._pending_runs.pop(queued_index)
                self._cancelled_runs.discard(run_id)
                queued_cancelled = True
            elif artifact.get("status") in {"completed", "failed"}:
                return {
                    "status": artifact.get("status", "unknown"),
                    "run_id": run_id,
                    "detail": "Run is not active.",
                }
            elif run_id in self._cancelled_runs:
                queued_cancelled = False
                already_cancelling = True
            elif artifact.get("status") in {"queued", "running", "cancelling"} or active_match:
                self._cancelled_runs.add(run_id)
                queued_cancelled = False
            else:
                return {
                    "status": artifact.get("status", "unknown"),
                    "run_id": run_id,
                    "detail": "Run is not active.",
                }

        if queued_cancelled:
            error_message = "Run stopped by user."
            self._emit_stage(run_id, "stopping")
            self._fail_run_with_terminal_event(run_id, error=error_message)
            return {"status": "cancelling", "run_id": run_id}

        if already_cancelling:
            return {"status": "cancelling", "run_id": run_id}

        claim_cancellation = getattr(
            self.run_store,
            "mark_run_cancelling_with_event",
            None,
        )
        if callable(claim_cancellation):
            artifact, stopping_event = claim_cancellation(run_id)
            if artifact.get("status") in {"completed", "failed"}:
                with self._control_lock:
                    self._cancelled_runs.discard(run_id)
                return {
                    "status": artifact.get("status", "unknown"),
                    "run_id": run_id,
                    "detail": "Run finished before cancellation could be claimed.",
                }
            if stopping_event is not None:
                self.run_hub.publish(run_id, stopping_event)
        else:
            artifact = self.run_store.mark_run_cancelling(run_id)
            if artifact.get("status") in {"completed", "failed"}:
                with self._control_lock:
                    self._cancelled_runs.discard(run_id)
                return {
                    "status": artifact.get("status", "unknown"),
                    "run_id": run_id,
                    "detail": "Run finished before cancellation could be claimed.",
                }
            self._emit_stage(run_id, "stopping")
        with self._control_lock:
            if self._active_run_id == run_id:
                # Keep the active-run check and provider abort atomic with
                # worker handoff. Otherwise this cancellation can race the
                # worker starting the next queued run and abort that request.
                self.app.llm_provider.abort_active_requests()
        return {"status": "cancelling", "run_id": run_id}

    def start_chat(
        self,
        *,
        prompt: str,
        user_id: str,
        thread_id: str,
        chat_model: str | None = None,
        temperature: float | None = None,
        reasoning_mode: str | None = None,
        thread_title: str | None = None,
        cross_chat_memory: bool = True,
        auto_compact_long_chats: bool = True,
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._ensure_user_unlocked(user_id)
        validated_attachments = _validated_attachments(attachments or [])
        if not prompt.strip() and not validated_attachments:
            raise RuntimeError("Prompt or a supported attachment is required.")
        return self._start_run(
            mode="chat",
            prompt=prompt,
            user_id=user_id,
            thread_id=thread_id,
            chat_model=chat_model,
            temperature=temperature,
            reasoning_mode=reasoning_mode,
            thread_title=thread_title,
            cross_chat_memory=cross_chat_memory,
            auto_compact_long_chats=auto_compact_long_chats,
            attachments=validated_attachments,
        )

    def start_manual_compact(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> dict[str, Any]:
        self._ensure_user_unlocked(user_id)
        thread = self.run_store.get_thread(user_id=user_id, thread_id=thread_id)
        snapshot = self._get_snapshot(user_id=user_id, thread_id=thread_id)
        if not list(snapshot.values.get("messages", [])) and (not thread or not thread.get("last_run_id")):
            raise RuntimeError("This thread does not have enough history to compact yet.")
        return self._start_run(
            mode="compact",
            prompt="",
            user_id=user_id,
            thread_id=thread_id,
            chat_model=None,
            temperature=None,
            reasoning_mode="off",
            thread_title=str(thread.get("title", "") or thread_id),
            cross_chat_memory=True,
            auto_compact_long_chats=True,
            attachments=[],
        )

    def list_memories(self, *, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._profile_operation(user_id):
            return [
                item.__dict__
                for item in self.app.list_memories(user_id=user_id, limit=limit)
            ]

    def reset_thread(self, *, thread_id: str, user_id: str) -> dict[str, Any]:
        resolved_user_id = user_id.strip()
        resolved_thread_id = thread_id.strip()
        if not resolved_user_id:
            raise RuntimeError("User id is required.")
        if not resolved_thread_id:
            raise RuntimeError("Thread id is required.")
        with self._profile_mutation(resolved_user_id, action="reset a thread"):
            self._ensure_user_unlocked(resolved_user_id)
            return self._reset_thread_data(
                user_id=resolved_user_id,
                thread_id=resolved_thread_id,
            )

    def _reset_thread_data(self, *, user_id: str, thread_id: str) -> dict[str, Any]:
        thread = self.run_store.get_thread(
            user_id=user_id,
            thread_id=thread_id,
        )
        checkpoint_owned = (
            user_id,
            thread_id,
        ) in set(self._list_checkpoint_threads())
        if not thread and not checkpoint_owned:
            raise RuntimeError(f"Thread not found: {thread_id}")
        runtime_thread_ids = {
            scoped_thread_id(user_id, thread_id),
            legacy_scoped_thread_id(user_id, thread_id),
        }
        if self._legacy_raw_thread_is_uniquely_owned(
            user_id=user_id,
            thread_id=thread_id,
        ):
            runtime_thread_ids.add(thread_id)
        with closing(open_application_sqlite(self.config.langgraph_checkpoint_db, data_dir=self.config.data_dir)) as conn:
            for runtime_thread_id in runtime_thread_ids:
                conn.execute("DELETE FROM writes WHERE thread_id = ?", (runtime_thread_id,))
                conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (runtime_thread_id,))
            conn.commit()
        self.run_store.delete_thread(
            user_id=user_id,
            thread_id=thread_id,
        )
        return {
            "status": "ok",
            "user_id": user_id,
            "thread_id": thread_id,
        }

    def reset_user(self, *, user_id: str, confirmation_user_id: str) -> dict[str, Any]:
        if confirmation_user_id != user_id:
            raise RuntimeError("User confirmation did not match the requested user id.")
        with self._profile_mutation(user_id, action="delete"):
            self._ensure_user_unlocked(user_id)
            try:
                thread_ids = {
                    item["thread_id"]
                    for item in self.run_store.list_threads(user_id=user_id)
                }
                thread_ids.update(
                    thread_id
                    for checkpoint_user_id, thread_id in self._list_checkpoint_threads()
                    if checkpoint_user_id == user_id
                )
                self.app.memory_service.delete_all(user_id=user_id)
                for thread_id in thread_ids:
                    self._reset_thread_data(thread_id=thread_id, user_id=user_id)
                self.run_store.delete_user(user_id)
                self._purge_legacy_migration_backups()
            finally:
                self._unlocked_users.discard(user_id)
                lock_user_key = getattr(self.run_store, "lock_user_key", None)
                if callable(lock_user_key):
                    lock_user_key(user_id)
        return {"status": "ok", "user_id": user_id}

    def _sanitize_user_summary(self, item: dict[str, Any]) -> dict[str, Any]:
        user_id = str(item.get("user_id", "") or "").strip()
        protection = str(item.get("protection", "passwordless") or "passwordless")
        is_user_key_unlocked = getattr(self.run_store, "is_user_key_unlocked", None)
        unlocked = True
        if callable(is_user_key_unlocked):
            unlocked = bool(is_user_key_unlocked(user_id))
        elif protection == PASSWORD_PROTECTED:
            unlocked = user_id in self._unlocked_users
        locked = protection == PASSWORD_PROTECTED and not unlocked
        return {
            "user_id": user_id,
            "updated_at": item.get("updated_at", ""),
            "protection": protection,
            "locked": locked,
        }

    def _ensure_user_unlocked(self, user_id: str) -> None:
        self._ensure_runtime_state()
        resolved = user_id.strip()
        if not resolved:
            raise RuntimeError("User id is required.")
        user = self._lookup_user_record(resolved)
        if not user:
            raise RuntimeError(f"User not found: {resolved}")
        if user.get("protection") == PASSWORD_PROTECTED:
            is_user_key_unlocked = getattr(self.run_store, "is_user_key_unlocked", None)
            key_is_unlocked = (
                bool(is_user_key_unlocked(resolved))
                if callable(is_user_key_unlocked)
                else resolved in self._unlocked_users
            )
            if resolved not in self._unlocked_users or not key_is_unlocked:
                self._unlocked_users.discard(resolved)
                raise RuntimeError("Unlock this user before continuing.")

    def _is_profile_accessible(self, user_id: str) -> bool:
        resolved = user_id.strip()
        if not resolved:
            return False
        user = self._lookup_user_record(resolved)
        if not user:
            return False
        if user.get("protection") != PASSWORD_PROTECTED:
            return True
        is_user_key_unlocked = getattr(self.run_store, "is_user_key_unlocked", None)
        key_is_unlocked = (
            bool(is_user_key_unlocked(resolved))
            if callable(is_user_key_unlocked)
            else resolved in self._unlocked_users
        )
        return resolved in self._unlocked_users and key_is_unlocked

    def _ensure_profile_has_no_active_runs(self, user_id: str, *, action: str) -> None:
        self._ensure_runtime_state()
        with self._control_lock:
            has_active_runs = self._profile_has_active_runs_locked(user_id)
        if has_active_runs:
            raise RuntimeError(
                f"Wait for this profile's active runs to finish before you {action} it."
            )

    def _profile_has_active_runs_locked(self, user_id: str) -> bool:
        resolved_user_id = user_id.strip()
        if any(
            str(getattr(job, "user_id", "") or "").strip() == resolved_user_id
            for job in self._pending_runs
        ):
            return True
        active_run_id = self._active_run_id
        if not active_run_id:
            return False
        get_run = getattr(self.run_store, "get_run", None)
        if not callable(get_run):
            return True
        try:
            artifact = get_run(active_run_id)
        except (KeyError, RuntimeError):
            return True
        return (
            str(artifact.get("user_id", "") or "").strip() == resolved_user_id
            and str(artifact.get("status", "") or "") in ACTIVE_RUN_STATUSES
        )

    def _raise_if_run_enqueue_blocked_locked(self, user_id: str) -> None:
        if self._shutdown_requested:
            raise RuntimeError("Atlas backend is shutting down.")
        if self._global_mutation_active:
            raise RuntimeError(
                "Atlas data reset is in progress. Try again when it finishes."
            )
        if user_id.strip() in self._mutating_users:
            raise RuntimeError(
                "This profile is being modified. Try again when it finishes."
            )

    @contextmanager
    def _profile_operation(self, user_id: str) -> Iterator[None]:
        self._ensure_runtime_state()
        resolved_user_id = user_id.strip()
        if not resolved_user_id:
            raise RuntimeError("User id is required.")
        with self._control_lock:
            if self._global_mutation_active:
                raise RuntimeError(
                    "Atlas data reset is in progress. Try again when it finishes."
                )
            if resolved_user_id in self._mutating_users:
                raise RuntimeError(
                    "This profile is being modified. Try again when it finishes."
                )
            self._active_profile_operations[resolved_user_id] = (
                self._active_profile_operations.get(resolved_user_id, 0) + 1
            )
        try:
            self._ensure_user_unlocked(resolved_user_id)
            yield
        finally:
            with self._control_lock:
                remaining = (
                    self._active_profile_operations.get(resolved_user_id, 1) - 1
                )
                if remaining > 0:
                    self._active_profile_operations[resolved_user_id] = remaining
                else:
                    self._active_profile_operations.pop(resolved_user_id, None)

    @contextmanager
    def _profile_mutation(self, user_id: str, *, action: str) -> Iterator[None]:
        self._ensure_runtime_state()
        resolved_user_id = user_id.strip()
        if not resolved_user_id:
            raise RuntimeError("User id is required.")
        with self._control_lock:
            if self._global_mutation_active:
                raise RuntimeError(
                    "Atlas data reset is in progress. Try again when it finishes."
                )
            if resolved_user_id in self._mutating_users:
                raise RuntimeError(
                    "This profile is already being modified. Try again when it finishes."
                )
            if self._active_profile_operations.get(resolved_user_id, 0) > 0:
                raise RuntimeError(
                    f"Wait for this profile's active operations to finish before you {action} it."
                )
            if self._profile_has_active_runs_locked(resolved_user_id):
                raise RuntimeError(
                    f"Wait for this profile's active runs to finish before you {action} it."
                )
            self._mutating_users.add(resolved_user_id)
        try:
            yield
        finally:
            with self._control_lock:
                self._mutating_users.discard(resolved_user_id)

    @contextmanager
    def _global_mutation(self) -> Iterator[None]:
        self._ensure_runtime_state()
        with self._control_lock:
            if self._global_mutation_active:
                raise RuntimeError(
                    "Atlas data reset is already in progress. Try again when it finishes."
                )
            if self._mutating_users:
                raise RuntimeError(
                    "Wait for profile changes to finish before you reset Atlas."
                )
            if self._active_profile_operations:
                raise RuntimeError(
                    "Wait for profile operations to finish before you reset Atlas."
                )
            if self._active_run_id is not None or self._pending_runs:
                raise RuntimeError(
                    "Wait for all active runs to finish before you reset Atlas."
                )
            self._global_mutation_active = True
        try:
            yield
        finally:
            with self._control_lock:
                self._global_mutation_active = False

    def _legacy_raw_thread_is_uniquely_owned(self, *, user_id: str, thread_id: str) -> bool:
        list_threads = getattr(self.run_store, "list_threads", None)
        if not callable(list_threads):
            return False
        owners = {
            str(item.get("user_id", "") or "")
            for item in list_threads(user_id=None)
            if str(item.get("thread_id", "") or "") == thread_id
        }
        return owners == {user_id}

    def _get_accessible_run(self, run_id: str) -> dict[str, Any]:
        with self._accessible_run_operation(run_id) as artifact:
            return artifact

    def _run_owner_id(self, run_id: str) -> str:
        get_run_metadata = getattr(self.run_store, "get_run_metadata", None)
        if callable(get_run_metadata):
            item = get_run_metadata(run_id)
        else:
            item = self.run_store.get_run(run_id)
        return str(item.get("user_id", "") or "").strip()

    @contextmanager
    def _accessible_run_operation(
        self,
        run_id: str,
    ) -> Iterator[dict[str, Any]]:
        user_id = self._run_owner_id(run_id)
        if not user_id:
            artifact = self.run_store.get_run(run_id)
            user_id = str(artifact.get("user_id", "") or "").strip()
            if not user_id:
                yield artifact
                return

        with self._profile_operation(user_id):
            # Re-read after acquiring the profile gate. A lock, delete, or reset
            # may have won the race after the metadata lookup above.
            artifact = self.run_store.get_run(run_id)
            actual_user_id = str(artifact.get("user_id", "") or "").strip()
            if actual_user_id != user_id:
                raise RuntimeError("Run ownership changed while it was being read.")
            yield artifact

    def _lookup_user_record(self, user_id: str) -> dict[str, Any] | None:
        get_user = getattr(self.run_store, "get_user", None)
        has_lookup = False
        if callable(get_user):
            has_lookup = True
            user = get_user(user_id)
            if user:
                return user
        list_users = getattr(self.run_store, "list_users", None)
        if callable(list_users):
            has_lookup = True
            for item in list_users():
                if str(item.get("user_id", "") or "").strip() == user_id:
                    return dict(item)
        if user_id and not has_lookup:
            return {"user_id": user_id, "protection": "passwordless", "updated_at": ""}
        return None

    def reset_all(self, *, confirmation: str) -> dict[str, Any]:
        if confirmation != "RESET ATLAS":
            raise RuntimeError("Reset confirmation did not match `RESET ATLAS`.")
        with self._global_mutation():
            with closing(open_application_sqlite(self.config.langgraph_checkpoint_db, data_dir=self.config.data_dir)) as conn:
                conn.execute("DELETE FROM writes")
                conn.execute("DELETE FROM checkpoints")
                conn.commit()
            self.app.memory_service.reset()
            runs_dir = self.config.data_dir / "runs"
            if runs_dir.exists():
                shutil.rmtree(runs_dir, ignore_errors=True)
            runs_dir.mkdir(parents=True, exist_ok=True)
            self.run_store.reset_all()
            self._unlocked_users.clear()
            self._deferred_recovery_runs.clear()
            self._purge_legacy_migration_backups()
        return {"status": "ok"}

    def _purge_legacy_migration_backups(self) -> None:
        purge_legacy_migration_backups(
            data_dir=self.config.data_dir,
            storage_paths=(
                self.config.langgraph_checkpoint_db,
                self.config.mem0_history_db,
                self.config.qdrant_path,
            ),
        )

    def _start_run(
        self,
        *,
        mode: str,
        prompt: str,
        user_id: str,
        thread_id: str,
        chat_model: str | None,
        temperature: float | None,
        reasoning_mode: str | None,
        thread_title: str | None,
        cross_chat_memory: bool,
        auto_compact_long_chats: bool,
        attachments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._ensure_runtime_state()
        with self._control_lock:
            self._raise_if_run_enqueue_blocked_locked(user_id)
        self._ensure_worker_started()
        history_after_message_count = self._thread_history_after_message_count(
            mode=mode,
            user_id=user_id,
            thread_id=thread_id,
        )

        resolved_chat_model = self._resolve_thread_model(
            user_id=user_id,
            thread_id=thread_id,
            requested_chat_model=chat_model,
        )
        if mode == "compact":
            resolved_temperature = 0.0
        else:
            resolved_temperature = self._resolve_thread_temperature(
                user_id=user_id,
                thread_id=thread_id,
                requested_temperature=temperature,
            )
        resolved_thread_title = self._resolve_thread_title(
            user_id=user_id,
            thread_id=thread_id,
            prompt=prompt,
            requested_thread_title=thread_title,
        )

        queue_position = 0
        with self._control_lock:
            self._raise_if_run_enqueue_blocked_locked(user_id)
            self._ensure_user_unlocked(user_id)
            artifact = self.run_store.create_run(
                mode=mode,
                user_id=user_id,
                thread_id=thread_id,
                chat_model=resolved_chat_model,
                temperature=resolved_temperature,
                prompt=prompt,
                thread_title=resolved_thread_title,
                status="queued",
                touch_thread=mode == "chat",
                history_after_message_count=history_after_message_count,
            )
            self._cancelled_runs.discard(artifact["run_id"])
            queue_position = len(self._pending_runs) + (1 if self._active_run_id else 0)
            self._pending_runs.append(
                _QueuedRunJob(
                    mode=mode,
                    run_id=artifact["run_id"],
                    prompt=prompt,
                    user_id=user_id,
                    thread_id=thread_id,
                    chat_model=resolved_chat_model,
                    temperature=resolved_temperature,
                    reasoning_mode=reasoning_mode,
                    cross_chat_memory=cross_chat_memory,
                    auto_compact_long_chats=auto_compact_long_chats,
                    attachments=attachments,
                )
            )
        if queue_position > 0:
            self._emit_event(
                artifact["run_id"],
                "run_queued",
                {
                    "mode": mode,
                    "thread_id": thread_id,
                    "queue_position": queue_position,
                },
            )
        self._emit_stage(artifact["run_id"], "queued")
        self._worker_wakeup.set()
        return {
            "run_id": artifact["run_id"],
            "status": artifact["status"],
            "mode": mode,
            "chat_model": resolved_chat_model,
            "temperature": resolved_temperature,
        }

    def _thread_history_after_message_count(self, *, mode: str, user_id: str, thread_id: str) -> int:
        snapshot = self._get_snapshot(user_id=user_id, thread_id=thread_id)
        message_count = len(list(snapshot.values.get("messages", [])))
        if mode == "chat":
            return message_count + 1
        return message_count

    def _execute_run(
        self,
        *,
        run_id: str,
        prompt: str,
        user_id: str,
        thread_id: str,
        chat_model: str,
        temperature: float | None,
        reasoning_mode: str | None,
        cross_chat_memory: bool,
        auto_compact_long_chats: bool,
        attachments: list[dict[str, Any]],
    ) -> None:
        session_id = scoped_thread_id(user_id, thread_id)
        config = {"configurable": {"thread_id": session_id}}
        snapshot = self._get_snapshot(user_id=user_id, thread_id=thread_id)
        prior_messages = list(snapshot.values.get("messages", []))
        self.run_store.update_run_history_after_message_count(
            run_id,
            history_after_message_count=len(prior_messages) + 1,
        )
        # start_chat validates and copies attachments before they enter the run queue.
        # Keeping that prepared representation avoids decoding images or parsing PDFs
        # a second time on the worker thread.
        validated_attachments = attachments
        image_attachments = [item for item in validated_attachments if item.get("kind") == "image"]
        if image_attachments and not self._model_supports_images(chat_model):
            raise RuntimeError(f"Model '{chat_model}' does not appear to support image input.")
        if _reasoning_mode_requested(reasoning_mode) and not self._model_supports_reasoning(chat_model):
            reasoning_mode = "off"
        self._prepare_ollama_model_switch(chat_model)
        self._remember_ollama_chat_model(chat_model)
        effective_context_window = self.app.llm_provider.effective_context_window(chat_model)
        new_user_message = HumanMessage(
            content=_build_user_message_content(prompt, validated_attachments),
            additional_kwargs={
                "atlas_user_prompt": _history_prompt_text(prompt, validated_attachments),
                "atlas_attachments": _history_attachment_metadata(validated_attachments),
            },
        )
        state: dict[str, Any] = dict(snapshot.values)
        state["messages"] = prior_messages + [new_user_message]
        state.setdefault("thread_summary", "")
        state.setdefault("compacted_message_count", 0)
        state["timeline_events"] = _persistent_thread_timeline_events(snapshot.values.get("timeline_events", []))
        state["detected_context_window"] = effective_context_window
        runtime = SimpleNamespace(
            context=GraphContext(
                user_id=user_id,
                thread_id=thread_id,
                session_id=session_id,
                chat_model=chat_model,
                chat_temperature=temperature,
                reasoning_mode=reasoning_mode,
                cross_chat_memory=cross_chat_memory,
                auto_compact_long_chats=auto_compact_long_chats,
                effective_context_window=effective_context_window,
            )
        )
        auto_compaction_budget = _CompactionCallBudget()
        try:
            self._raise_if_cancelled(run_id)
            self._emit_event(
                run_id,
                "run_started",
                {
                    "mode": "chat",
                    "thread_id": thread_id,
                    "chat_model": chat_model,
                    "temperature": temperature,
                },
            )

            self._emit_stage(run_id, "memory_retrieval")
            self._run_graph_node(run_id=run_id, node_name="retrieve_memories", state=state, runtime=runtime)
            self._emit_trace(
                run_id,
                stage="memory retrieval",
                rationale="Retrieved semantically related long-term memory for the current prompt.",
                inputs={"query": prompt},
                outputs={"count": len(state.get("retrieved_memories", [])), "items": state.get("retrieved_memories", [])[:5]},
            )

            self._raise_if_cancelled(run_id)
            self._apply_auto_compaction(
                run_id=run_id,
                config=config,
                state=state,
                runtime=runtime,
                after_message_count=len(prior_messages) + 1,
                protected_recent_count=1,
                checkpoint_when_compacted_count_at_most=len(prior_messages),
                model_call_budget=auto_compaction_budget,
            )

            self._raise_if_cancelled(run_id)
            # A cold local model can spend a meaningful amount of time loading
            # before it yields its first chunk. Keep that wait distinct from
            # active generation so the desktop UI can report truthful progress.
            self._emit_stage(run_id, "model_loading")
            answer = self._stream_answer(run_id=run_id, state=state, runtime=runtime)
            self._raise_if_cancelled(run_id)
            new_ai_message = AIMessage(content=answer)
            state["messages"] = prior_messages + [new_user_message, new_ai_message]
            state["answer"] = answer
            self._emit_trace(
                run_id,
                stage="synthesis",
                rationale="Synthesized the final answer from the current thread and retrieved memory context.",
                inputs={"memory_items": len(state.get("retrieved_memories", []))},
                outputs={"answer_chars": len(answer)},
            )

            self._apply_auto_compaction(
                run_id=run_id,
                config=config,
                state=state,
                runtime=runtime,
                after_message_count=len(state.get("messages", [])),
                protected_recent_count=2,
                checkpoint_when_compacted_count_at_most=None,
                model_call_budget=auto_compaction_budget,
            )

            self._emit_stage(run_id, "memory_persistence")
            self._run_graph_node(run_id=run_id, node_name="extract_updates", state=state, runtime=runtime)
            try:
                self._run_graph_node(
                    run_id=run_id,
                    node_name="persist",
                    state=state,
                    runtime=runtime,
                )
            except Exception as exc:
                # Durable cross-chat memory is supplementary. Once synthesis has
                # succeeded, a vector-store or embedder failure must not discard
                # the answer or omit this turn from thread history.
                self._raise_if_cancelled(run_id)
                state["memory_persistence_warnings"] = [
                    {
                        "category": "memory_service",
                        "error": (
                            str(exc).strip()
                            or "Atlas memory storage rejected the update."
                        )[:500],
                    }
                ]
            memory_warnings = [
                item
                for item in state.get("memory_persistence_warnings", [])
                if isinstance(item, dict)
            ][:8]
            if memory_warnings:
                self._emit_event(
                    run_id,
                    "run_warning",
                    {
                        "stage": "memory_persistence",
                        "message": (
                            "Atlas completed the answer, but could not save one or "
                            "more memory updates for future chats."
                        ),
                        "count": len(memory_warnings),
                        "errors": [
                            str(item.get("error", "") or "")[:500]
                            for item in memory_warnings
                        ],
                    },
                )
            self._emit_trace(
                run_id,
                stage="memory persistence",
                rationale=(
                    "Attempted to persist durable user facts without making "
                    "supplementary memory storage a prerequisite for preserving the answer."
                ),
                inputs={"candidates": len(state.get("update_candidates", []))},
                outputs={
                    "stored": len(state.get("persisted_memories", [])),
                    "warning_count": len(memory_warnings),
                    "warnings": memory_warnings,
                },
            )

            self.app.graph.update_state(
                config,
                {
                    "messages": [new_user_message, new_ai_message],
                    "answer": answer,
                    "thread_summary": str(state.get("thread_summary", "") or ""),
                    "compacted_message_count": int(state.get("compacted_message_count", 0) or 0),
                    "detected_context_window": int(state.get("detected_context_window", 0) or 0),
                    "timeline_events": list(state.get("timeline_events", [])),
                },
                as_node="persist",
            )

            self._complete_run_with_terminal_event(
                run_id,
                answer=state.get("answer", ""),
            )
        except Exception as exc:  # pragma: no cover - integration path
            error_message = "Run stopped by user." if self._is_cancelled(run_id) else str(exc)
            self._cleanup_ollama_model_after_resource_failure(chat_model, error_message)
            self._fail_run_with_terminal_event(run_id, error=error_message)

    def _execute_compact_run(
        self,
        *,
        run_id: str,
        user_id: str,
        thread_id: str,
        chat_model: str,
    ) -> None:
        session_id = scoped_thread_id(user_id, thread_id)
        config = {"configurable": {"thread_id": session_id}}
        snapshot = self._get_snapshot(user_id=user_id, thread_id=thread_id)
        all_messages = list(snapshot.values.get("messages", []))

        self._prepare_ollama_model_switch(chat_model)
        self._remember_ollama_chat_model(chat_model)
        effective_context_window = self.app.llm_provider.effective_context_window(chat_model)
        state: dict[str, Any] = dict(snapshot.values)
        state["messages"] = all_messages
        state.setdefault("thread_summary", "")
        state.setdefault("compacted_message_count", 0)
        state["timeline_events"] = _persistent_thread_timeline_events(snapshot.values.get("timeline_events", []))
        state["detected_context_window"] = effective_context_window
        runtime = SimpleNamespace(
            context=GraphContext(
                user_id=user_id,
                thread_id=thread_id,
                session_id=session_id,
                chat_model=chat_model,
                chat_temperature=0.0,
                cross_chat_memory=True,
                auto_compact_long_chats=True,
                effective_context_window=effective_context_window,
            )
        )

        try:
            if not all_messages:
                raise RuntimeError("This thread does not have enough history to compact yet.")
            self._raise_if_cancelled(run_id)
            self._emit_event(
                run_id,
                "run_started",
                {
                    "mode": "compact",
                    "thread_id": thread_id,
                    "chat_model": chat_model,
                    "temperature": 0.0,
                },
            )
            self._emit_stage(run_id, "compaction")

            prior_compacted_count = int(state.get("compacted_message_count", 0) or 0)
            representation_tokens_before = self._count_thread_representation_tokens(
                model=chat_model,
                messages=state.get("messages", []),
                thread_summary=str(state.get("thread_summary", "") or ""),
                compacted_message_count=prior_compacted_count,
            )
            compaction = self._manual_compact_context(
                state=state,
                runtime=runtime,
                cancel_check=lambda: self._is_cancelled(run_id),
            )
            self._raise_if_cancelled(run_id)
            state.update(compaction)
            manual_compaction_status = str(compaction.get("manual_compaction_status", "") or "")
            updated_compacted_count = int(state.get("compacted_message_count", 0) or 0)
            representation_tokens_after = self._count_thread_representation_tokens(
                model=chat_model,
                messages=state.get("messages", []),
                thread_summary=str(state.get("thread_summary", "") or ""),
                compacted_message_count=updated_compacted_count,
            )
            if updated_compacted_count <= prior_compacted_count and not (
                manual_compaction_status == "tightened_summary"
                and representation_tokens_after < representation_tokens_before
            ):
                if manual_compaction_status == "summary_not_useful":
                    raise RuntimeError(
                        "Manual compact skipped: the generated summary only saved a few tokens, so Atlas kept "
                        "the current context instead."
                    )
                if manual_compaction_status == "summary_too_large":
                    raise RuntimeError(
                        "Manual compact could not reduce this thread: the generated summary would be larger than "
                        "the messages it replaces."
                    )
                if manual_compaction_status == "summary_unavailable":
                    raise RuntimeError(
                        "Manual compact could not obtain a summary from the selected model, so Atlas kept the "
                        "thread unchanged."
                    )
                if manual_compaction_status == "input_too_large":
                    raise RuntimeError(
                        "Manual compact skipped: the oldest uncompacted conversation turn exceeds the safe "
                        "compaction input limit, so Atlas kept it intact."
                    )
                raise RuntimeError(
                    "Manual compact skipped: only the most recent messages remain raw, so there is no older "
                    "context to compact."
                )
            if representation_tokens_after >= representation_tokens_before:
                raise RuntimeError(
                    "Manual compact could not reduce this thread: the generated summary would be larger than "
                    "the messages it replaces."
                )

            self._raise_if_cancelled(run_id)
            self._persist_compaction_event(
                run_id=run_id,
                state=state,
                prior_compacted_count=prior_compacted_count,
                representation_tokens_before=representation_tokens_before,
                representation_tokens_after=representation_tokens_after,
                after_message_count=len(all_messages),
                reason="manual",
            )

            self.app.graph.update_state(
                config,
                {
                    "thread_summary": str(state.get("thread_summary", "") or ""),
                    "compacted_message_count": int(state.get("compacted_message_count", 0) or 0),
                    "detected_context_window": int(state.get("detected_context_window", 0) or 0),
                    "timeline_events": list(state.get("timeline_events", [])),
                },
                as_node="persist",
            )

            self._raise_if_cancelled(run_id)
            self._complete_run_with_terminal_event(
                run_id,
                answer="",
                terminal_payload={
                    "result": "compacted",
                },
            )
        except Exception as exc:  # pragma: no cover - integration path
            error_message = "Run stopped by user." if self._is_cancelled(run_id) else str(exc)
            self._cleanup_ollama_model_after_resource_failure(chat_model, error_message)
            self._fail_run_with_terminal_event(run_id, error=error_message)

    def _stream_answer(
        self,
        *,
        run_id: str,
        state: dict[str, Any],
        runtime: SimpleNamespace,
    ) -> str:
        messages = _build_answer_messages(
            state=state,
            runtime_context=runtime.context,
            token_counter=self._message_token_counter(runtime.context.chat_model),
            answer_prompt_template=getattr(self.app.nodes, "answer_prompt_template", ""),
        )

        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        stream_parser = _ThinkingStreamParser()
        answer_repair = MojibakeRepairStream()
        thinking_repair = MojibakeRepairStream()
        streamed_chars = 0
        generation_started = False

        def mark_generation_started(*parts: str) -> None:
            nonlocal generation_started
            if generation_started or not any(parts):
                return
            generation_started = True
            self._emit_stage(run_id, "generation")

        def reserve_stream_chars(*parts: str) -> None:
            nonlocal streamed_chars
            streamed_chars += sum(len(part) for part in parts)
            if streamed_chars <= MAX_MODEL_STREAM_CHARS:
                return
            abort_active_requests = getattr(
                self.app.llm_provider,
                "abort_active_requests",
                None,
            )
            if callable(abort_active_requests):
                try:
                    abort_active_requests()
                except Exception:
                    pass
            raise RuntimeError(
                "Model response exceeded Atlas's safe size limit. "
                "Ask for a shorter answer and retry."
            )

        self._raise_if_cancelled(run_id)
        try:
            for chunk in self.app.llm_provider.chat(
                runtime.context.chat_model,
                temperature=runtime.context.chat_temperature,
                reasoning=runtime.context.reasoning_mode,
            ).stream(messages):
                self._raise_if_cancelled(run_id)
                raw_answer_text, raw_thinking_text = _extract_chunk_stream_parts(chunk, stream_parser)
                thinking_text = thinking_repair.consume(raw_thinking_text)
                answer_text = answer_repair.consume(raw_answer_text)
                reserve_stream_chars(thinking_text, answer_text)
                mark_generation_started(thinking_text, answer_text)
                if thinking_text:
                    thinking_parts.append(thinking_text)
                    self._emit_event(run_id, "thinking_token", {"text": thinking_text})
                if answer_text:
                    answer_parts.append(answer_text)
                    self._emit_event(run_id, "token", {"text": answer_text})
                self._raise_if_cancelled(run_id)
        except Exception as exc:
            if self._is_cancelled(run_id):
                raise
            raise format_runtime_error(self.config, exc, chat_model=runtime.context.chat_model) from exc
        trailing_answer, trailing_thinking = stream_parser.flush()
        trailing_thinking = thinking_repair.consume(trailing_thinking) + thinking_repair.flush()
        trailing_answer = answer_repair.consume(trailing_answer) + answer_repair.flush()
        reserve_stream_chars(trailing_thinking, trailing_answer)
        mark_generation_started(trailing_thinking, trailing_answer)
        if trailing_thinking:
            thinking_parts.append(trailing_thinking)
            self._emit_event(run_id, "thinking_token", {"text": trailing_thinking})
        if trailing_answer:
            answer_parts.append(trailing_answer)
            self._emit_event(run_id, "token", {"text": trailing_answer})
        answer_body = "".join(answer_parts)
        answer = _finalize_answer_text(answer_body)
        suffix = answer[len(answer_body):]
        if suffix:
            self._emit_event(run_id, "token", {"text": suffix})
        if not answer:
            raise RuntimeError(
                "Model returned an empty response. If this thread is near the context limit, compact it "
                "and retry so Atlas can send a smaller prompt."
            )
        return answer

    def _run_graph_node(
        self,
        *,
        run_id: str,
        node_name: str,
        state: dict[str, Any],
        runtime: SimpleNamespace,
    ) -> None:
        allowed_nodes = set(pre_synthesis_node_sequence()) | set(post_synthesis_node_sequence())
        if node_name not in allowed_nodes:
            raise RuntimeError(f"Unsupported graph execution node: {node_name}")
        self._raise_if_cancelled(run_id)
        handler = getattr(self.app.nodes, node_name)
        state.update(handler(state, runtime))
        self._raise_if_cancelled(run_id)

    def _is_cancelled(self, run_id: str) -> bool:
        with self._control_lock:
            return run_id in self._cancelled_runs

    def _raise_if_cancelled(self, run_id: str) -> None:
        if self._is_cancelled(run_id):
            raise RuntimeError("Run stopped by user.")

    def _fail_run_from_exception(self, run_id: str, exc: Exception) -> None:
        error_message = "Run stopped by user." if self._is_cancelled(run_id) else str(exc)
        try:
            artifact = self.run_store.get_run(run_id)
        except Exception:
            artifact = {}
        if artifact.get("status") not in {"completed", "failed"}:
            self._fail_run_with_terminal_event(run_id, error=error_message)
            return
        existing_terminal_events = {
            str(event.get("type", "") or "")
            for event in artifact.get("events", [])
            if isinstance(event, dict)
        }
        if "run_completed" not in existing_terminal_events and "run_failed" not in existing_terminal_events:
            self._fail_run_with_terminal_event(run_id, error=error_message)

    def _persist_compaction_event(
        self,
        *,
        run_id: str,
        state: dict[str, Any],
        prior_compacted_count: int,
        representation_tokens_before: int,
        representation_tokens_after: int,
        after_message_count: int,
        reason: str,
    ) -> None:
        updated_compacted_count = int(state.get("compacted_message_count", 0) or 0)
        compaction_event = self._emit_event(
            run_id,
            "context_compacted",
            {
                "compacted_message_count": updated_compacted_count,
                "newly_compacted_message_count": max(0, updated_compacted_count - prior_compacted_count),
                "thread_summary": str(state.get("thread_summary", "") or ""),
                "detected_context_window": int(state.get("detected_context_window", 0) or 0),
                "history_representation_tokens_before_compaction": representation_tokens_before,
                "history_representation_tokens_after_compaction": representation_tokens_after,
                "compaction_reason": reason,
            },
        )
        state["timeline_events"] = list(state.get("timeline_events", [])) + [
            {
                "type": "context_compacted",
                "timestamp": compaction_event["timestamp"],
                "run_id": run_id,
                "after_message_count": after_message_count,
                **compaction_event["payload"],
            }
        ]

    def _checkpoint_compaction_state(self, *, config: dict[str, Any], state: dict[str, Any]) -> None:
        self.app.graph.update_state(
            config,
            {
                "thread_summary": str(state.get("thread_summary", "") or ""),
                "compacted_message_count": int(state.get("compacted_message_count", 0) or 0),
                "detected_context_window": int(state.get("detected_context_window", 0) or 0),
                "timeline_events": list(state.get("timeline_events", [])),
            },
            as_node="persist",
        )

    def _apply_auto_compaction(
        self,
        *,
        run_id: str,
        config: dict[str, Any],
        state: dict[str, Any],
        runtime: SimpleNamespace,
        after_message_count: int,
        protected_recent_count: int,
        checkpoint_when_compacted_count_at_most: int | None,
        model_call_budget: _CompactionCallBudget | None = None,
    ) -> bool:
        if not getattr(runtime.context, "auto_compact_long_chats", True):
            return False
        if model_call_budget is not None and model_call_budget.remaining <= 0:
            return False

        chat_model = str(getattr(runtime.context, "chat_model", "") or "")
        effective_context_window = int(getattr(runtime.context, "effective_context_window", 0) or 0)
        prior_compacted_count = int(state.get("compacted_message_count", 0) or 0)
        prior_thread_summary = str(state.get("thread_summary", "") or "")
        representation_tokens_before = self._count_thread_representation_tokens(
            model=chat_model,
            messages=state.get("messages", []),
            thread_summary=prior_thread_summary,
            compacted_message_count=prior_compacted_count,
        )
        if effective_context_window > 0:
            threshold = _auto_compact_threshold(effective_context_window)
            if threshold > 0 and representation_tokens_before >= threshold:
                self._emit_stage(run_id, "compaction")

        compaction = self._maybe_compact_context(
            state=state,
            runtime=runtime,
            protected_recent_count=protected_recent_count,
            current_representation_tokens=representation_tokens_before,
            cancel_check=lambda: self._is_cancelled(run_id),
            model_call_budget=model_call_budget,
        )
        if not compaction:
            return False

        state.update(compaction)
        updated_compacted_count = int(state.get("compacted_message_count", 0) or 0)
        representation_tokens_after = self._count_thread_representation_tokens(
            model=chat_model,
            messages=state.get("messages", []),
            thread_summary=str(state.get("thread_summary", "") or ""),
            compacted_message_count=updated_compacted_count,
        )
        if not (
            (
                updated_compacted_count > prior_compacted_count
                or str(state.get("thread_summary", "") or "") != prior_thread_summary
            )
            and representation_tokens_after < representation_tokens_before
        ):
            return False

        self._persist_compaction_event(
            run_id=run_id,
            state=state,
            prior_compacted_count=prior_compacted_count,
            representation_tokens_before=representation_tokens_before,
            representation_tokens_after=representation_tokens_after,
            after_message_count=after_message_count,
            reason="auto",
        )
        if (
            checkpoint_when_compacted_count_at_most is not None
            and updated_compacted_count <= checkpoint_when_compacted_count_at_most
        ):
            self._checkpoint_compaction_state(config=config, state=state)
        return True

    def _maybe_compact_context(
        self,
        *,
        state: dict[str, Any],
        runtime: SimpleNamespace,
        protected_recent_count: int = 1,
        current_representation_tokens: int | None = None,
        cancel_check: Callable[[], bool] | None = None,
        model_call_budget: _CompactionCallBudget | None = None,
    ) -> dict[str, Any]:
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Run stopped by user.")
        if not getattr(runtime.context, "auto_compact_long_chats", True):
            return {"detected_context_window": int(runtime.context.effective_context_window or 0)}

        effective_context_window = int(runtime.context.effective_context_window or 0)
        if effective_context_window <= 0:
            return {}

        all_messages = list(state.get("messages", []))
        threshold = _auto_compact_threshold(effective_context_window)
        target_after_compaction = _compaction_target_tokens(effective_context_window)
        updated_summary = str(state.get("thread_summary", "") or "")
        compacted_count = max(0, min(int(state.get("compacted_message_count", 0) or 0), len(all_messages)))
        if current_representation_tokens is None:
            current_representation_tokens = self._count_thread_representation_tokens(
                model=runtime.context.chat_model,
                messages=all_messages,
                thread_summary=updated_summary,
                compacted_message_count=compacted_count,
            )
        else:
            current_representation_tokens = max(0, int(current_representation_tokens))
        if threshold > 0 and current_representation_tokens < threshold:
            return {"detected_context_window": effective_context_window}

        protected_recent_count = max(0, min(int(protected_recent_count or 0), len(all_messages)))
        max_iterations = 3

        while (
            max_iterations > 0
            and (target_after_compaction <= 0 or current_representation_tokens > target_after_compaction)
        ):
            cutoff = len(all_messages) - protected_recent_count
            if cutoff <= compacted_count:
                if updated_summary.strip():
                    if model_call_budget is not None and not model_call_budget.claim():
                        break
                    tighten_kwargs: dict[str, Any] = {
                        "model": runtime.context.chat_model,
                        "existing_summary": updated_summary,
                        "target_words": _summary_target_words(
                            existing_summary=updated_summary,
                            messages=[],
                            replacement_tokens=current_representation_tokens,
                            effective_context_window=effective_context_window,
                        ),
                    }
                    if cancel_check is not None:
                        tighten_kwargs["cancel_check"] = cancel_check
                    try:
                        proposed_summary = self._tighten_thread_summary(
                            **tighten_kwargs,
                        )
                    except _CompactionModelUnavailable:
                        break
                    proposed_representation_tokens = self._count_thread_representation_tokens(
                        model=runtime.context.chat_model,
                        messages=all_messages,
                        thread_summary=proposed_summary,
                        compacted_message_count=compacted_count,
                    )
                    if _compaction_gain_is_acceptable(
                        before_tokens=current_representation_tokens,
                        after_tokens=proposed_representation_tokens,
                        effective_context_window=effective_context_window,
                        allow_shallow=current_representation_tokens >= threshold,
                    ):
                        updated_summary = proposed_summary
                        current_representation_tokens = proposed_representation_tokens
                        state["thread_summary"] = updated_summary
                break
            batch_messages, consumed = _select_messages_for_compaction(
                all_messages[compacted_count:cutoff],
                max_chars=max(6000, int(effective_context_window * 3)),
            )
            if not batch_messages or consumed <= 0:
                break
            if model_call_budget is not None and not model_call_budget.claim():
                break

            summarize_kwargs: dict[str, Any] = {
                "model": runtime.context.chat_model,
                "existing_summary": updated_summary,
                "messages": batch_messages,
                "target_words": _summary_target_words(
                    existing_summary=updated_summary,
                    messages=batch_messages,
                    replacement_tokens=current_representation_tokens,
                    effective_context_window=effective_context_window,
                ),
            }
            if cancel_check is not None:
                summarize_kwargs["cancel_check"] = cancel_check
            try:
                proposed_summary = self._summarize_message_batch(**summarize_kwargs)
            except _CompactionModelUnavailable:
                break
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Run stopped by user.")
            proposed_compacted_count = compacted_count + consumed
            best_summary: str | None = None
            best_representation_tokens = current_representation_tokens
            for candidate_summary in _compaction_summary_candidates(
                primary_summary=proposed_summary,
            ):
                candidate_representation_tokens = self._count_thread_representation_tokens(
                    model=runtime.context.chat_model,
                    messages=all_messages,
                    thread_summary=candidate_summary,
                    compacted_message_count=proposed_compacted_count,
                )
                if _compaction_gain_is_acceptable(
                    before_tokens=current_representation_tokens,
                    after_tokens=candidate_representation_tokens,
                    effective_context_window=effective_context_window,
                    allow_shallow=current_representation_tokens >= threshold,
                ) and candidate_representation_tokens < best_representation_tokens:
                    best_summary = candidate_summary
                    best_representation_tokens = candidate_representation_tokens
            if best_summary is None:
                break

            updated_summary = best_summary
            compacted_count = proposed_compacted_count
            current_representation_tokens = best_representation_tokens
            state["thread_summary"] = updated_summary
            state["compacted_message_count"] = compacted_count
            max_iterations -= 1

        return {
            "thread_summary": updated_summary,
            "compacted_message_count": compacted_count,
            "detected_context_window": effective_context_window,
        }

    def _message_token_counter(self, model: str):
        provider = getattr(getattr(self, "app", None), "llm_provider", None)
        counter = getattr(provider, "count_message_tokens", None)
        if not callable(counter):
            return None
        return lambda messages: counter(model, messages)

    def _count_messages_tokens(self, *, model: str, messages: list[Any]) -> int:
        if not messages:
            return 0
        token_counter = self._message_token_counter(model)
        if token_counter is not None:
            try:
                counted = int(token_counter(messages))
                if counted >= 0:
                    return counted
            except Exception:
                pass
        return _estimate_prompt_messages_tokens(messages)

    def _count_thread_representation_tokens(
        self,
        *,
        model: str,
        messages: list[Any],
        thread_summary: str,
        compacted_message_count: int,
    ) -> int:
        clamped_compacted_count = max(0, min(int(compacted_message_count or 0), len(messages)))
        total = self._count_messages_tokens(model=model, messages=messages[clamped_compacted_count:])
        return total + self._count_summary_tokens(
            model=model,
            thread_summary=thread_summary,
        )

    def _count_summary_tokens(self, *, model: str, thread_summary: str) -> int:
        summary = thread_summary.strip()
        if not summary:
            return 0
        return self._count_messages_tokens(
            model=model,
            messages=[SystemMessage(content=f"Conversation summary from earlier in this thread:\n{summary}")],
        )

    def _compaction_timeout_seconds(self) -> float:
        raw_value = getattr(getattr(self, "config", None), "compaction_timeout_seconds", None)
        try:
            timeout = float(raw_value if raw_value is not None else DEFAULT_COMPACTION_TIMEOUT_SECONDS)
        except (TypeError, ValueError):
            timeout = DEFAULT_COMPACTION_TIMEOUT_SECONDS
        return max(1.0, min(180.0, timeout))

    def _manual_compact_context(
        self,
        *,
        state: dict[str, Any],
        runtime: SimpleNamespace,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Run stopped by user.")
        effective_context_window = int(runtime.context.effective_context_window or 0)
        all_messages = list(state.get("messages", []))
        updated_summary = str(state.get("thread_summary", "") or "")
        compacted_count = max(0, min(int(state.get("compacted_message_count", 0) or 0), len(all_messages)))
        remaining_messages = all_messages[compacted_count:]
        no_change_payload = {
            "thread_summary": updated_summary,
            "compacted_message_count": compacted_count,
            "detected_context_window": effective_context_window,
            "manual_compaction_status": "nothing_to_compact",
        }
        current_representation_tokens = self._count_thread_representation_tokens(
            model=runtime.context.chat_model,
            messages=all_messages,
            thread_summary=updated_summary,
            compacted_message_count=compacted_count,
        )
        if len(remaining_messages) <= 2:
            if not updated_summary.strip():
                return no_change_payload
            tighten_kwargs: dict[str, Any] = {
                "model": runtime.context.chat_model,
                "existing_summary": updated_summary,
                "target_words": _summary_target_words(
                    existing_summary=updated_summary,
                    messages=[],
                    replacement_tokens=current_representation_tokens,
                    effective_context_window=effective_context_window,
                ),
            }
            if cancel_check is not None:
                tighten_kwargs["cancel_check"] = cancel_check
            try:
                proposed_summary = self._tighten_thread_summary(**tighten_kwargs)
            except _CompactionModelUnavailable:
                return {
                    **no_change_payload,
                    "manual_compaction_status": "summary_unavailable",
                }
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Run stopped by user.")
            proposed_representation_tokens = self._count_thread_representation_tokens(
                model=runtime.context.chat_model,
                messages=all_messages,
                thread_summary=proposed_summary,
                compacted_message_count=compacted_count,
            )
            if proposed_representation_tokens < current_representation_tokens:
                return {
                    "thread_summary": proposed_summary,
                    "compacted_message_count": compacted_count,
                    "detected_context_window": effective_context_window,
                    "manual_compaction_status": "tightened_summary",
                }
            if proposed_representation_tokens == current_representation_tokens:
                return {
                    **no_change_payload,
                    "manual_compaction_status": "summary_not_useful",
                }
            return {
                **no_change_payload,
                "manual_compaction_status": "summary_too_large",
            }

        max_chars = max(6000, int(max(effective_context_window, 1024) * 3))
        cutoff = len(all_messages) - 2
        batch_messages, consumed = _select_messages_for_compaction(
            all_messages[compacted_count:cutoff],
            max_chars=max_chars,
        )
        if not batch_messages or consumed <= 0:
            if _compaction_prefix_exceeds_limit(
                all_messages[compacted_count:cutoff],
                max_chars=max_chars,
            ):
                return {
                    **no_change_payload,
                    "manual_compaction_status": "input_too_large",
                }
            return no_change_payload

        replacement_tokens = self._count_messages_tokens(
            model=runtime.context.chat_model,
            messages=batch_messages,
        )
        summarize_kwargs: dict[str, Any] = {
            "model": runtime.context.chat_model,
            "existing_summary": updated_summary,
            "messages": batch_messages,
            "target_words": _summary_target_words(
                existing_summary=updated_summary,
                messages=batch_messages,
                replacement_tokens=replacement_tokens,
                effective_context_window=effective_context_window,
            ),
        }
        if cancel_check is not None:
            summarize_kwargs["cancel_check"] = cancel_check
        try:
            proposed_summary = self._summarize_message_batch(**summarize_kwargs)
        except _CompactionModelUnavailable:
            return {
                **no_change_payload,
                "manual_compaction_status": "summary_unavailable",
            }
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Run stopped by user.")

        proposed_compacted_count = compacted_count + consumed
        raw_tail_tokens = self._count_messages_tokens(
            model=runtime.context.chat_model,
            messages=all_messages[proposed_compacted_count:],
        )
        best_payload: dict[str, Any] | None = None
        best_representation_tokens = current_representation_tokens
        for candidate_summary in _compaction_summary_candidates(
            primary_summary=proposed_summary,
        ):
            proposed_representation_tokens = raw_tail_tokens + self._count_summary_tokens(
                model=runtime.context.chat_model,
                thread_summary=candidate_summary,
            )
            if (
                _compaction_gain_is_acceptable(
                    before_tokens=current_representation_tokens,
                    after_tokens=proposed_representation_tokens,
                    effective_context_window=effective_context_window,
                    allow_shallow=True,
                )
                and proposed_representation_tokens < best_representation_tokens
            ):
                best_representation_tokens = proposed_representation_tokens
                best_payload = {
                    "thread_summary": candidate_summary,
                    "compacted_message_count": proposed_compacted_count,
                    "detected_context_window": effective_context_window,
                    "manual_compaction_status": "compacted",
                }

        if best_payload is not None:
            return best_payload
        return {
            **no_change_payload,
            "manual_compaction_status": "summary_too_large",
        }

    def _invoke_compaction_model(
        self,
        *,
        model: str,
        prompt: str,
        cancel_check: Callable[[], bool] | None = None,
    ) -> str:
        def abort_request() -> None:
            provider = getattr(getattr(self, "app", None), "llm_provider", None)
            abort_active_requests = getattr(provider, "abort_active_requests", None)
            if callable(abort_active_requests):
                try:
                    abort_active_requests()
                except Exception:
                    pass

        def raise_if_cancelled() -> None:
            if cancel_check is not None and cancel_check():
                abort_request()
                raise RuntimeError("Run stopped by user.")

        raise_if_cancelled()

        def invoke() -> str:
            try:
                response = self.app.llm_provider.chat(model, temperature=0.0, reasoning=False).invoke(
                    [HumanMessage(content=prompt)]
                )
                result_queue.put(("ok", str(response.content).strip()), block=False)
            except Exception as exc:
                result_queue.put(("error", str(exc).strip()), block=False)

        result_queue: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)
        worker = threading.Thread(target=invoke, name="atlas-compaction", daemon=True)
        worker.start()
        deadline = monotonic() + self._compaction_timeout_seconds()
        while True:
            raise_if_cancelled()
            remaining = deadline - monotonic()
            if remaining <= 0:
                abort_request()
                raise _CompactionModelUnavailable(
                    "The compaction model timed out before returning a summary."
                )
            try:
                status, value = result_queue.get(
                    timeout=min(COMPACTION_CANCEL_POLL_SECONDS, remaining)
                )
            except queue.Empty:
                continue
            raise_if_cancelled()
            if status == "ok" and value:
                return value
            if status == "error" and value:
                raise _CompactionModelUnavailable(
                    f"The compaction model could not create a summary: {value}"
                )
            raise _CompactionModelUnavailable(
                "The compaction model returned an empty summary."
            )

    def _summarize_message_batch(
        self,
        *,
        model: str,
        existing_summary: str,
        messages: list[HumanMessage | AIMessage],
        target_words: int | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> str:
        transcript = _render_messages_for_summary(messages)
        if not transcript:
            return existing_summary
        word_limit = _normalized_summary_word_limit(target_words)
        pins = _compaction_exact_pins(existing_summary=existing_summary, transcript=transcript)

        prompt_parts = [
            "Merge the earlier part of this conversation into a compact working summary that preserves exact details needed to continue the thread correctly.",
            "Keep durable user facts, active goals, explicit constraints, exact names and titles, chronology, worldbuilding details, list items, decisions, and unresolved tasks.",
            "Preserve exact code-facing details: repository names, file paths, function names, commands, package names, versions, errors, screenshots referenced by path, release tags, and test results.",
            "When existing summary content and the new chunk overlap, merge them into a single current view instead of duplicating old bullets.",
            "Do not replace specific details with vague phrases like 'details were discussed' or 'a plan was outlined'.",
            "If the conversation contains creative work, preserve concrete canon details such as character names, relationships, settings, episode labels, plot beats, tone, style, and production constraints.",
            "Use short bullets under these headings when relevant: Current objective, Durable facts, Decisions and constraints, Code and file references, Open tasks, Canon details.",
            f"Keep it concise but detail-dense. Stay under {word_limit} words and make the merged summary shorter than the existing summary plus the new chunk it replaces.",
        ]
        if existing_summary.strip():
            prompt_parts.append("\nExisting summary:\n" + existing_summary.strip())
        if pins:
            prompt_parts.append("\nPinned exact details that must survive verbatim:\n" + "\n".join(f"- {pin}" for pin in pins))
        prompt_parts.append("\nNew conversation chunk:\n" + transcript)

        summary = self._invoke_compaction_model(
            model=model,
            prompt="\n".join(prompt_parts),
            cancel_check=cancel_check,
        )
        return _apply_compaction_pins(summary, pins)

    def _tighten_thread_summary(
        self,
        *,
        model: str,
        existing_summary: str,
        target_words: int | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> str:
        summary = existing_summary.strip()
        if not summary:
            return existing_summary
        word_limit = _normalized_summary_word_limit(target_words)
        pins = _compaction_exact_pins(existing_summary=summary, transcript="")
        prompt = "\n".join(
            [
                "Rewrite this existing conversation summary into a much tighter working summary.",
                "Preserve exact active goals, durable user facts, decisions, constraints, code paths, commands, errors, names, and unresolved tasks.",
                "Remove repetition, explanations, completed tangents, and generic background detail.",
                "Do not drop pinned exact details such as file paths, commands, versions, URLs, errors, release tags, test results, or model names.",
                f"Stay under {word_limit} words. Use compact bullets when useful.",
                "\nPinned exact details:\n" + "\n".join(f"- {pin}" for pin in pins) if pins else "",
                "\nExisting summary:\n" + summary,
            ]
        )
        tightened = self._invoke_compaction_model(
            model=model,
            prompt=prompt,
            cancel_check=cancel_check,
        )
        return _apply_compaction_pins(tightened, pins)

    def _resolve_thread_model(
        self,
        *,
        user_id: str,
        thread_id: str,
        requested_chat_model: str | None,
    ) -> str:
        requested = (requested_chat_model or "").strip()
        thread = self.run_store.get_thread(user_id=user_id, thread_id=thread_id)
        locked_model = ""
        if thread:
            locked_model = str(thread.get("chat_model", "") or "").strip()
            if not locked_model and thread.get("last_run_id"):
                locked_model = self._thread_last_run_model(thread)

        if locked_model:
            if requested and requested != locked_model:
                raise RuntimeError(
                    f"Thread '{thread_id}' is locked to chat model '{locked_model}'. Start a new chat to use '{requested}'."
                )
            return locked_model
        resolved = requested
        if not resolved:
            raise RuntimeError("Select a local chat model before starting this chat.")
        return resolved

    def _thread_last_run_model(self, thread: dict[str, Any] | None) -> str:
        if not thread:
            return ""
        last_run_id = str(thread.get("last_run_id", "") or "").strip()
        if not last_run_id:
            return ""
        try:
            artifact = self.run_store.get_run(last_run_id)
        except Exception:
            return ""
        return str(artifact.get("chat_model", "") or "").strip()

    def _resolve_thread_temperature(
        self,
        *,
        user_id: str,
        thread_id: str,
        requested_temperature: float | None,
    ) -> float | None:
        thread = self.run_store.get_thread(user_id=user_id, thread_id=thread_id)
        has_history = bool(thread and thread.get("last_run_id"))
        if has_history:
            locked_temperature = self._record_temperature(thread, fallback_on_missing=self.config.chat_temperature)
            if locked_temperature is None:
                if requested_temperature is not None:
                    raise RuntimeError(
                        f"Thread '{thread_id}' is locked to the selected model's temperature setting. Start a new chat to use '{float(requested_temperature):.1f}'."
                    )
                return None
            if requested_temperature is not None and abs(float(requested_temperature) - locked_temperature) > 1e-9:
                raise RuntimeError(
                    f"Thread '{thread_id}' is locked to temperature '{locked_temperature:.1f}'. Start a new chat to use '{float(requested_temperature):.1f}'."
                )
            return locked_temperature

        if requested_temperature is None:
            return None
        return float(requested_temperature)

    @staticmethod
    def _record_temperature(record: dict[str, Any] | None, *, fallback_on_missing: float | None = None) -> float | None:
        if not record or "temperature" not in record:
            return fallback_on_missing
        raw_value = record.get("temperature")
        if raw_value in (None, ""):
            return None
        return float(raw_value)

    def _resolve_thread_title(
        self,
        *,
        user_id: str,
        thread_id: str,
        prompt: str,
        requested_thread_title: str | None,
    ) -> str:
        requested = (requested_thread_title or "").strip()
        existing_thread = self.run_store.get_thread(user_id=user_id, thread_id=thread_id)
        existing_title = str(existing_thread.get("title", "")).strip() if existing_thread else ""

        if existing_title and existing_title not in {"main", thread_id} and not existing_title.startswith("atlas-"):
            return existing_title
        if requested and requested not in {"main", thread_id} and not requested.startswith("atlas-"):
            return requested
        return _suggest_thread_title(prompt) or thread_id

    def _emit_stage(self, run_id: str, stage: str) -> None:
        self._emit_event(run_id, "stage_changed", {"stage": stage})

    def _emit_trace(
        self,
        run_id: str,
        *,
        stage: str,
        rationale: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        artifacts: dict[str, Any] | None = None,
    ) -> None:
        item = {
            "stage": stage,
            "rationale": rationale,
            "inputs": inputs,
            "outputs": outputs,
            "artifacts": artifacts or {},
        }
        stored = self.run_store.append_trace_item(run_id, item)
        self._emit_event(run_id, "trace_item", stored)

    def _emit_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> RunEvent:
        event = self.run_store.append_event(run_id, event_type, payload)
        self.run_hub.publish(run_id, event)
        return event

    def _complete_run_with_terminal_event(
        self,
        run_id: str,
        *,
        answer: str,
        terminal_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        finish = getattr(self.run_store, "complete_run_with_event", None)
        if callable(finish):
            artifact, event = finish(
                run_id,
                answer=answer,
                terminal_payload=terminal_payload,
            )
            self.run_hub.publish(run_id, event)
            return artifact
        artifact = self.run_store.complete_run(run_id, answer=answer)
        self._emit_event(
            run_id,
            "run_completed",
            {
                "answer": artifact.get("answer", ""),
                **(terminal_payload or {}),
            },
        )
        return artifact

    def _fail_run_with_terminal_event(
        self,
        run_id: str,
        *,
        error: str,
    ) -> dict[str, Any]:
        finish = getattr(self.run_store, "fail_run_with_event", None)
        if callable(finish):
            artifact, event = finish(run_id, error=error)
            self.run_hub.publish(run_id, event)
            return artifact
        artifact = self.run_store.fail_run(run_id, error=error)
        self._emit_event(run_id, "run_failed", {"error": error})
        return artifact

    def _list_checkpoint_threads(self) -> list[tuple[str, str]]:
        config = getattr(self, "config", None)
        checkpoint_db = getattr(config, "langgraph_checkpoint_db", None)
        if checkpoint_db is None or not checkpoint_db.exists():
            return []
        with closing(
            open_application_sqlite(checkpoint_db, data_dir=config.data_dir)
        ) as conn:
            rows = conn.execute(
                """
                SELECT thread_id FROM checkpoints
                UNION
                SELECT thread_id FROM writes
                ORDER BY thread_id ASC
                """
            ).fetchall()
        decoded: list[tuple[str, str]] = []
        for row in rows:
            if not row or not row[0]:
                continue
            parsed = parse_scoped_thread_id(str(row[0]))
            if parsed is not None:
                decoded.append(parsed)
        return decoded

    def _get_snapshot(self, *, user_id: str | None, thread_id: str):
        scoped_id = scoped_thread_id(user_id, thread_id) if user_id else thread_id
        scoped_snapshot = self.app.graph.get_state({"configurable": {"thread_id": scoped_id}})
        if scoped_snapshot.values.get("messages"):
            return scoped_snapshot
        if user_id:
            owned_thread = self.run_store.get_thread(user_id=user_id, thread_id=thread_id)
            if owned_thread:
                legacy_scoped_snapshot = self.app.graph.get_state(
                    {
                        "configurable": {
                            "thread_id": legacy_scoped_thread_id(user_id, thread_id)
                        }
                    }
                )
                if legacy_scoped_snapshot.values.get("messages"):
                    return legacy_scoped_snapshot
                if self._legacy_raw_thread_is_uniquely_owned(
                    user_id=user_id,
                    thread_id=thread_id,
                ):
                    legacy_snapshot = self.app.graph.get_state(
                        {"configurable": {"thread_id": thread_id}}
                    )
                    if legacy_snapshot.values.get("messages"):
                        return legacy_snapshot
        return scoped_snapshot

    def _model_supports_images(self, model_name: str) -> bool:
        for item in self._get_model_catalog().models:
            if item.name == model_name:
                return item.supports_images
        return False

    def _model_supports_reasoning(self, model_name: str) -> bool:
        for item in self._get_model_catalog().models:
            if item.name == model_name:
                return bool(item.supports_reasoning and item.reasoning_mode_strategy != "none")
        return False

    def _get_model_catalog(self, *, ttl_seconds: float = 10.0) -> OllamaCatalogSnapshot:
        cached = self._model_catalog_cache
        now = monotonic()
        if cached and now - cached[0] < ttl_seconds:
            return cached[1]
        catalog = inspect_local_models(self.config)
        self._model_catalog_cache = (now, catalog)
        return catalog


def _estimate_prompt_messages_tokens(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            total += max(1, len(content) // 4)
            continue
        if isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    total += max(1, len(item) // 4)
                elif isinstance(item, dict):
                    item_type = str(item.get("type", "")).strip().lower()
                    if item_type == "text":
                        total += max(1, len(str(item.get("text", ""))) // 4)
                    elif item_type == "image_url":
                        total += 256
            total += 8
            continue
        total += max(1, len(str(content)) // 4)
    return total


def _estimate_history_tokens(messages: list[Any]) -> int:
    return _estimate_prompt_messages_tokens(messages)


def _estimate_thread_representation_tokens(
    *,
    messages: list[Any],
    thread_summary: str,
    compacted_message_count: int,
) -> int:
    clamped_compacted_count = max(0, min(int(compacted_message_count or 0), len(messages)))
    total = _estimate_history_tokens(messages[clamped_compacted_count:])
    summary = thread_summary.strip()
    if summary:
        total += _estimate_prompt_messages_tokens(
            [SystemMessage(content=f"Conversation summary from earlier in this thread:\n{summary}")]
        )
    return total


def _auto_compact_threshold(effective_context_window: int) -> int:
    if effective_context_window <= 0:
        return 0
    return max(256, int(effective_context_window * 0.72))


def _compaction_target_tokens(effective_context_window: int) -> int:
    threshold = _auto_compact_threshold(effective_context_window)
    if threshold <= 0:
        return 0
    return max(512, int(threshold * 0.55))


def _compaction_gain_is_worthwhile(
    *,
    before_tokens: int,
    after_tokens: int,
    effective_context_window: int,
) -> bool:
    if after_tokens >= before_tokens:
        return False
    gain = before_tokens - after_tokens
    budget_basis = min(max(before_tokens, 1), max(effective_context_window, 1024))
    minimum_gain = max(64, int(budget_basis * 0.03))
    return gain >= minimum_gain


def _compaction_gain_is_acceptable(
    *,
    before_tokens: int,
    after_tokens: int,
    effective_context_window: int,
    allow_shallow: bool = False,
) -> bool:
    if _compaction_gain_is_worthwhile(
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        effective_context_window=effective_context_window,
    ):
        return True
    return allow_shallow and after_tokens < before_tokens


def _summary_target_words(
    *,
    existing_summary: str,
    messages: list[HumanMessage | AIMessage],
    replacement_tokens: int,
    effective_context_window: int,
) -> int:
    replacement_word_count = len(_render_messages_for_summary(messages).split()) + len(existing_summary.split())
    replacement_word_cap = max(40, replacement_word_count - 1) if replacement_word_count > 1 else 160
    token_budget_words = max(40, int(max(1, replacement_tokens) * 0.35))
    context_budget_words = max(64, int(max(effective_context_window, 1024) * 0.03))
    return _normalized_summary_word_limit(
        min(replacement_word_cap, token_budget_words, context_budget_words)
    )


def _compaction_summary_candidates(
    *,
    primary_summary: str,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for candidate in (primary_summary,):
        cleaned = str(candidate or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cleaned)
    return candidates


def _normalized_summary_word_limit(target_words: int | None) -> int:
    if target_words is None:
        return 520
    return max(40, min(520, int(target_words or 0)))


def _select_messages_for_compaction(
    messages: list[HumanMessage | AIMessage],
    *,
    max_chars: int,
) -> tuple[list[HumanMessage | AIMessage], int]:
    """Select the largest complete, prefix-preserving batch within ``max_chars``.

    A user message and its immediately following assistant response are treated
    as one unit. Oversized units are never truncated because advancing the
    compacted-message boundary after summarizing only part of a message would
    silently discard the omitted content.
    """

    selected: list[HumanMessage | AIMessage] = []
    consumed = 0
    index = 0
    used_chars = 0
    bounded_max_chars = max(0, int(max_chars))
    while index < len(messages):
        unit_end = index + 1
        if (
            isinstance(messages[index], HumanMessage)
            and unit_end < len(messages)
            and isinstance(messages[unit_end], AIMessage)
        ):
            unit_end += 1
        unit = [
            message
            for message in messages[index:unit_end]
            if _message_text_for_summary(message)
        ]
        unit_transcript = _render_messages_for_summary(unit)
        separator_chars = 1 if selected and unit_transcript else 0
        candidate_chars = used_chars + separator_chars + len(unit_transcript)
        if candidate_chars > bounded_max_chars:
            if not selected:
                return [], 0
            break
        selected.extend(unit)
        consumed = unit_end
        used_chars = candidate_chars
        index = unit_end
    return selected, consumed


def _compaction_prefix_exceeds_limit(
    messages: list[HumanMessage | AIMessage],
    *,
    max_chars: int,
) -> bool:
    if not messages:
        return False
    unit_end = 1
    if (
        isinstance(messages[0], HumanMessage)
        and len(messages) > 1
        and isinstance(messages[1], AIMessage)
    ):
        unit_end = 2
    transcript = _render_messages_for_summary(messages[:unit_end])
    return bool(transcript) and len(transcript) > max(0, int(max_chars))


def _render_messages_for_summary(messages: list[HumanMessage | AIMessage]) -> str:
    rendered: list[str] = []
    for message in messages:
        text = _message_text_for_summary(message)
        if not text:
            continue
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        rendered.append(f"{role}: {text}")
    return "\n".join(rendered)


def _message_text_for_summary(message: HumanMessage | AIMessage) -> str:
    text = _latest_user_text({"messages": [message]}) if isinstance(message, HumanMessage) else _chunk_to_text(message)
    return _clean_transcript_text(str(text))


def _clean_transcript_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    cleaned: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank and cleaned:
                cleaned.append("")
            blank = True
            continue
        cleaned.append(line)
        blank = False
    return "\n".join(cleaned).strip()


def _compaction_exact_pins(*, existing_summary: str, transcript: str) -> list[str]:
    source = "\n".join(part for part in (existing_summary, transcript) if part.strip())
    if not source.strip():
        return []
    patterns = [
        r"`([^`\n]{2,220})`",
        r"https?://[^\s)>\]]+",
        r"(?<!\w)(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.@+:-]+",
        r"(?<!\w)[A-Za-z0-9_.-]+\.(?:py|ts|tsx|js|jsx|json|toml|yaml|yml|md|txt|css|html|sqlite|db|sh|ps1|bat|rs)\b",
        r"\bv?\d+\.\d+(?:\.\d+)?(?:[-+][A-Za-z0-9_.-]+)?\b",
        r"\b[A-Za-z][A-Za-z0-9_.-]*:[A-Za-z0-9_.-]+\b",
        r"\b\d+\s+(?:passed|failed|errors?|subtests passed)\b",
        r"\b(?:RuntimeError|ImportError|AssertionError|Traceback|CUDA error|Ollama request failed)[^\n]{0,180}",
    ]
    pins: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, source):
            value = match.group(1) if match.lastindex else match.group(0)
            pin = _normalize_compaction_pin(value)
            if not pin:
                continue
            key = pin.lower()
            if key in seen:
                continue
            seen.add(key)
            pins.append(pin)
            if len(pins) >= COMPACTION_PIN_LIMIT:
                return pins
    return pins


def _normalize_compaction_pin(value: str) -> str:
    pin = " ".join(str(value).strip().split())
    pin = pin.strip(".,;:!?()[]{}<>\"'")
    if len(pin) < 2 or len(pin) > 220:
        return ""
    return pin


def _apply_compaction_pins(summary: str, pins: list[str]) -> str:
    cleaned_summary = summary.strip()
    if not cleaned_summary or not pins:
        return cleaned_summary
    summary_lookup = cleaned_summary.lower()
    missing = [pin for pin in pins if pin.lower() not in summary_lookup]
    if not missing:
        return cleaned_summary
    lines = [f"- {_format_compaction_pin(pin)}" for pin in missing]
    return f"{cleaned_summary}\n\nPinned exact details:\n" + "\n".join(lines)


def _format_compaction_pin(pin: str) -> str:
    if "`" in pin:
        return pin
    if re.search(r"[\s/.:_-]", pin):
        return f"`{pin}`"
    return pin


def _chunk_to_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return str(content) if content else ""


@dataclass
class _ThinkingStreamParser:
    in_think: bool = False
    pending: str = ""
    open_tag: str = "<think>"
    close_tag: str = "</think>"

    def consume(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        data = f"{self.pending}{text}"
        self.pending = ""
        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        index = 0

        while index < len(data):
            marker = self.close_tag if self.in_think else self.open_tag
            match = data.find(marker, index)
            if match == -1:
                safe_end = _safe_stream_boundary(data, marker)
                segment = data[index:safe_end]
                if self.in_think:
                    thinking_parts.append(segment)
                else:
                    answer_parts.append(segment)
                self.pending = data[safe_end:]
                break

            segment = data[index:match]
            if self.in_think:
                thinking_parts.append(segment)
                index = match + len(self.close_tag)
                self.in_think = False
            else:
                answer_parts.append(segment)
                index = match + len(self.open_tag)
                self.in_think = True

        return "".join(answer_parts), "".join(thinking_parts)

    def flush(self) -> tuple[str, str]:
        if not self.pending:
            return "", ""
        remainder = self.pending
        self.pending = ""
        if self.in_think:
            return "", remainder
        return remainder, ""


def _extract_chunk_stream_parts(chunk: Any, parser: _ThinkingStreamParser) -> tuple[str, str]:
    thinking_parts: list[str] = []
    answer_parts: list[str] = []

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
    if isinstance(additional_kwargs, dict):
        reasoning_text = str(additional_kwargs.get("reasoning_content") or "")
        if reasoning_text:
            thinking_parts.append(reasoning_text)

    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        answer_text, thinking_text = parser.consume(content)
        answer_parts.append(answer_text)
        thinking_parts.append(thinking_text)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                answer_text, thinking_text = parser.consume(item)
                answer_parts.append(answer_text)
                thinking_parts.append(thinking_text)
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type in {"reasoning", "thinking"}:
                thinking_parts.append(str(item.get("reasoning") or item.get("text") or item.get("content") or ""))
                continue
            text = str(item.get("text") or item.get("content") or "")
            answer_text, thinking_text = parser.consume(text)
            answer_parts.append(answer_text)
            thinking_parts.append(thinking_text)
    else:
        answer_text, thinking_text = parser.consume(str(content) if content else "")
        answer_parts.append(answer_text)
        thinking_parts.append(thinking_text)

    return "".join(part for part in answer_parts if part), "".join(part for part in thinking_parts if part)


def _safe_stream_boundary(data: str, marker: str) -> int:
    max_overlap = min(len(marker) - 1, len(data))
    for overlap in range(max_overlap, 0, -1):
        if data.endswith(marker[:overlap]):
            return len(data) - overlap
    return len(data)


def _reasoning_mode_requested(value: str | None) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized not in {"", "off", "false", "none", "default"}


def _validated_attachments(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(attachments) > MAX_ATTACHMENTS:
        raise RuntimeError(f"A chat can include at most {MAX_ATTACHMENTS} attachments.")
    validated: list[dict[str, Any]] = []
    total_byte_size = 0
    for index, item in enumerate(attachments):
        if not isinstance(item, dict):
            raise RuntimeError(f"Attachment {index + 1} is invalid.")
        kind = str(item.get("kind", "")).strip().lower()
        name = str(item.get("name", "")).strip() or "attachment"
        media_type = _canonical_media_type(str(item.get("media_type", "")).strip())
        raw_byte_size = item.get("byte_size")
        byte_size = _safe_attachment_size(raw_byte_size)
        if raw_byte_size is not None and byte_size is None:
            raise RuntimeError(f"Attachment '{name}' has an invalid byte size.")
        data_url = str(item.get("data_url", "")).strip()
        text_content = str(item.get("text_content", "") or "")

        if kind not in {"", "file", "image"}:
            raise RuntimeError(f"Attachment '{name}' has an unsupported kind.")
        if data_url and text_content:
            raise RuntimeError(f"Attachment '{name}' must use one content source.")

        decoded_data_url = _decode_data_url(data_url) if data_url else None
        if data_url and decoded_data_url is None:
            raise RuntimeError(
                f"Attachment '{name}' must contain valid base64 data no larger than 10 MiB."
            )
        data_url_media_type = decoded_data_url[0] if decoded_data_url else ""
        raw_bytes = decoded_data_url[1] if decoded_data_url else None
        is_image = (
            kind == "image"
            or media_type.startswith("image/")
            or data_url_media_type.startswith("image/")
        )

        if is_image:
            if text_content or raw_bytes is None:
                raise RuntimeError(f"Image attachment '{name}' requires a base64 image data URL.")
            resolved_media_type = _validated_image_media_type(
                name=name,
                declared_media_type=media_type,
                data_url_media_type=data_url_media_type,
                raw_bytes=raw_bytes,
            )
            actual_byte_size = len(raw_bytes)
            _validate_declared_attachment_size(
                name=name,
                declared_byte_size=byte_size,
                actual_byte_size=actual_byte_size,
            )
            validated.append(
                {
                    "kind": "image",
                    "name": name,
                    "media_type": resolved_media_type,
                    "data_url": data_url,
                    "byte_size": actual_byte_size,
                }
            )
            total_byte_size += actual_byte_size
            if total_byte_size > MAX_TOTAL_ATTACHMENT_BYTES:
                raise RuntimeError("Attachments exceed the 25 MiB request limit.")
            continue

        if media_type in {"", "application/octet-stream"} and data_url_media_type:
            resolved_media_type = data_url_media_type
        else:
            resolved_media_type = media_type or "text/plain"
        extracted_text = _attachment_text_content(
            name=name,
            media_type=resolved_media_type,
            text_content=text_content,
            data_url=data_url,
            raw_bytes=raw_bytes,
        )
        if not extracted_text:
            if _is_pdf_attachment(name, resolved_media_type):
                raise RuntimeError(
                    f"PDF attachment '{name}' is invalid or does not contain extractable text."
                )
            raise RuntimeError(f"Attachment '{name}' is not a supported text or PDF file.")

        if raw_bytes is not None:
            _validate_file_media_type_match(
                name=name,
                declared_media_type=media_type,
                data_url_media_type=data_url_media_type,
            )
            actual_byte_size = len(raw_bytes)
        else:
            actual_byte_size = len(text_content.encode("utf-8"))
        _validate_declared_attachment_size(
            name=name,
            declared_byte_size=byte_size,
            actual_byte_size=actual_byte_size,
        )
        validated.append(
            {
                "kind": "file",
                "name": name,
                "media_type": resolved_media_type,
                "text_content": extracted_text,
                "byte_size": actual_byte_size,
            }
        )
        total_byte_size += actual_byte_size
        if total_byte_size > MAX_TOTAL_ATTACHMENT_BYTES:
            raise RuntimeError("Attachments exceed the 25 MiB request limit.")
    return validated


def _build_user_message_content(prompt: str, attachments: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    text = prompt.strip()
    file_context = _build_attachment_context_text(attachments)
    if file_context:
        text = "\n\n".join(part for part in [text, file_context] if part).strip()

    image_attachments = [item for item in attachments if item.get("kind") == "image" and item.get("data_url")]
    if not image_attachments:
        return text

    parts: list[dict[str, Any]] = [{"type": "text", "text": text or "Describe the attached images."}]
    for item in image_attachments:
        parts.append({"type": "image_url", "image_url": item["data_url"]})
    return parts


def _message_to_history_parts(message: Any) -> tuple[str, list[dict[str, Any]]]:
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    stored_prompt = str(additional_kwargs.get("atlas_user_prompt", "") or "").strip()
    stored_attachments = additional_kwargs.get("atlas_attachments")
    content = stored_prompt or _message_content_to_history_text(getattr(message, "content", ""))
    attachments = _history_attachment_list(stored_attachments)
    if attachments:
        return content, attachments
    fallback_content, fallback_attachments = _message_content_to_history_parts(getattr(message, "content", ""))
    return content or fallback_content, fallback_attachments


def _messages_to_search_index_items(messages: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = ""
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        if not role:
            continue
        content, _attachments = _message_to_history_parts(message)
        if content.strip():
            items.append({"role": role, "content": content})
    return items


def _message_content_to_history_parts(content: Any) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    attachments: list[dict[str, Any]] = []
    if isinstance(content, str):
        return content, attachments
    if not isinstance(content, list):
        return str(content), attachments

    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text_parts.append(str(item.get("text", "")))
            continue
        if item.get("type") == "image_url":
            raw_value = item.get("image_url")
            if isinstance(raw_value, dict):
                raw_value = raw_value.get("url", "")
            data_url = str(raw_value or "")
            media_type = _data_url_media_type(data_url)
            if not _is_safe_image_media_type(media_type):
                continue
            attachments.append(
                {
                    "kind": "image",
                    "name": "image",
                    "media_type": media_type,
                    "data_url": data_url,
                }
            )
    return "\n".join(part for part in text_parts if part).strip(), attachments


def _message_content_to_history_text(content: Any) -> str:
    text, _ = _message_content_to_history_parts(content)
    return text


def _data_url_media_type(value: str) -> str:
    parsed = _parse_data_url_header(value)
    return parsed[0] if parsed else ""


def _history_attachment_metadata(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for item in attachments:
        kind = str(item.get("kind", "")).strip().lower()
        if kind == "image":
            metadata.append(
                {
                    "kind": "image",
                    "name": str(item.get("name", "") or "image"),
                    "media_type": str(item.get("media_type", "") or "image/png"),
                    "data_url": str(item.get("data_url", "") or ""),
                    "byte_size": _safe_attachment_size(item.get("byte_size")),
                }
            )
        elif kind == "file":
            metadata.append(
                {
                    "kind": "file",
                    "name": str(item.get("name", "") or "file"),
                    "media_type": str(item.get("media_type", "") or "text/plain"),
                    "byte_size": _safe_attachment_size(item.get("byte_size")),
                }
            )
    return metadata


def _history_attachment_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    attachments: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in {"image", "file"}:
            continue
        data_url = str(item.get("data_url", "") or "") if kind == "image" else ""
        default_media_type = "image/png" if kind == "image" else "text/plain"
        declared_media_type = _canonical_media_type(str(item.get("media_type", "") or ""))
        media_type = declared_media_type or default_media_type
        if kind == "image":
            embedded_media_type = _data_url_media_type(data_url)
            media_type = declared_media_type or embedded_media_type
            if (
                not _is_safe_image_media_type(media_type)
                or not _is_safe_image_media_type(embedded_media_type)
                or media_type != embedded_media_type
            ):
                continue
        attachments.append(
            {
                "kind": kind,
                "name": str(item.get("name", "") or ("image" if kind == "image" else "file")),
                "media_type": media_type,
                "data_url": data_url,
                "byte_size": _safe_attachment_size(item.get("byte_size")),
            }
        )
    return attachments


def _history_prompt_text(prompt: str, attachments: list[dict[str, Any]]) -> str:
    text = prompt.strip()
    if text:
        return text
    file_names = [str(item.get("name", "") or "").strip() for item in attachments if str(item.get("name", "")).strip()]
    if not file_names:
        return ""
    if len(file_names) == 1:
        return f"Attached {file_names[0]}"
    return f"Attached {len(file_names)} files"


def _build_attachment_context_text(attachments: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    remaining = MAX_ATTACHMENT_CONTEXT_CHARS
    for item in attachments:
        if item.get("kind") != "file":
            continue
        text_content = str(item.get("text_content", "") or "").strip()
        if not text_content or remaining <= 0:
            continue
        snippet = text_content[: min(remaining, MAX_ATTACHMENT_TEXT_CHARS)].strip()
        if not snippet:
            continue
        sections.append(f"[Attached file: {item.get('name', 'file')}]\n{snippet}")
        remaining -= len(snippet)
    if not sections:
        return ""
    return "Use the attached files as context when answering.\n\n" + "\n\n".join(sections)


def _attachment_text_content(
    *,
    name: str,
    media_type: str,
    text_content: str,
    data_url: str,
    raw_bytes: bytes | None = None,
) -> str:
    direct_text = text_content.strip()
    if direct_text:
        return direct_text[:MAX_ATTACHMENT_TEXT_CHARS]
    lowered_name = name.strip().lower()
    lowered_type = _canonical_media_type(media_type)
    if _is_pdf_attachment(lowered_name, lowered_type):
        return _extract_pdf_text(data_url, raw_bytes=raw_bytes)[:MAX_ATTACHMENT_TEXT_CHARS]
    if _is_text_attachment(lowered_name, lowered_type):
        return _decode_data_url_text(data_url, raw_bytes=raw_bytes)[:MAX_ATTACHMENT_TEXT_CHARS]
    return ""


def _extract_pdf_text(data_url: str, *, raw_bytes: bytes | None = None) -> str:
    if raw_bytes is None:
        raw_bytes = _decode_data_url_bytes(data_url)
    if raw_bytes is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
    except Exception:
        return ""
    parts: list[str] = []
    extracted_chars = 0
    try:
        pages = islice(reader.pages, MAX_PDF_PAGES)
        for page in pages:
            try:
                page_text = (page.extract_text() or "").strip()
            except Exception:
                page_text = ""
            if not page_text:
                continue
            remaining = MAX_PDF_EXTRACTED_CHARS - extracted_chars
            if remaining <= 0:
                break
            part = page_text[:remaining]
            parts.append(part)
            extracted_chars += len(part)
            if extracted_chars >= MAX_PDF_EXTRACTED_CHARS:
                break
    except Exception:
        pass
    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def _decode_data_url_text(data_url: str, *, raw_bytes: bytes | None = None) -> str:
    if raw_bytes is None:
        raw_bytes = _decode_data_url_bytes(data_url)
    if raw_bytes is None:
        return ""
    return raw_bytes.decode("utf-8", errors="replace").strip()


def _decode_data_url_bytes(data_url: str) -> bytes | None:
    decoded = _decode_data_url(data_url)
    return decoded[1] if decoded else None


def _is_valid_data_url(value: str) -> bool:
    return _decode_data_url(value) is not None


def _parse_data_url_header(value: str) -> tuple[str, str] | None:
    header, separator, encoded = value.partition(",")
    if not separator or not header.lower().startswith("data:"):
        return None
    metadata = header[5:].split(";")
    if len(metadata) < 2 or metadata[-1].strip().lower() != "base64":
        return None
    media_type = _canonical_media_type(metadata[0])
    if not media_type or not _MEDIA_TYPE_PATTERN.fullmatch(media_type):
        return None
    return media_type, encoded


def _decode_data_url(value: str) -> tuple[str, bytes] | None:
    parsed = _parse_data_url_header(value)
    if parsed is None:
        return None
    media_type, encoded = parsed
    max_encoded_size = 4 * ((MAX_ATTACHMENT_BYTES + 2) // 3)
    if not encoded or len(encoded) > max_encoded_size:
        return None
    try:
        raw_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        return None
    if not raw_bytes or len(raw_bytes) > MAX_ATTACHMENT_BYTES:
        return None
    return media_type, raw_bytes


def _canonical_media_type(value: str) -> str:
    media_type = str(value or "").strip().lower().partition(";")[0].strip()
    return _IMAGE_MEDIA_TYPE_ALIASES.get(media_type, media_type)


def _is_safe_image_media_type(media_type: str) -> bool:
    normalized = _canonical_media_type(media_type)
    return normalized in _SUPPORTED_IMAGE_MEDIA_TYPES


def _validated_image_media_type(
    *,
    name: str,
    declared_media_type: str,
    data_url_media_type: str,
    raw_bytes: bytes,
) -> str:
    embedded = _canonical_media_type(data_url_media_type)
    declared = _canonical_media_type(declared_media_type)
    if not _is_safe_image_media_type(embedded):
        raise RuntimeError(f"Image attachment '{name}' uses an unsupported image type.")
    if declared not in {"", "application/octet-stream"}:
        if not _is_safe_image_media_type(declared) or declared != embedded:
            raise RuntimeError(
                f"Image attachment '{name}' media type does not match its data URL."
            )
    if not _image_signature_matches(embedded, raw_bytes):
        raise RuntimeError(
            f"Image attachment '{name}' content does not match its media type."
        )
    return embedded


def _image_signature_matches(media_type: str, raw_bytes: bytes) -> bool:
    if media_type == "image/png":
        return raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    if media_type == "image/jpeg":
        return raw_bytes.startswith(b"\xff\xd8\xff")
    if media_type == "image/gif":
        return raw_bytes.startswith((b"GIF87a", b"GIF89a"))
    if media_type == "image/webp":
        return (
            len(raw_bytes) >= 12
            and raw_bytes.startswith(b"RIFF")
            and raw_bytes[8:12] == b"WEBP"
        )
    if media_type in {"image/bmp", "image/x-ms-bmp"}:
        return raw_bytes.startswith(b"BM")
    if media_type in {"image/tiff", "image/x-tiff"}:
        return raw_bytes.startswith((b"II*\x00", b"MM\x00*"))
    if media_type in {"image/vnd.microsoft.icon", "image/x-icon"}:
        return raw_bytes.startswith(b"\x00\x00\x01\x00")
    return False


def _validate_file_media_type_match(
    *,
    name: str,
    declared_media_type: str,
    data_url_media_type: str,
) -> None:
    declared = _canonical_media_type(declared_media_type)
    embedded = _canonical_media_type(data_url_media_type)
    unspecified = {"", "application/octet-stream"}
    if declared not in unspecified and embedded not in unspecified and declared != embedded:
        raise RuntimeError(f"Attachment '{name}' media type does not match its data URL.")


def _validate_declared_attachment_size(
    *,
    name: str,
    declared_byte_size: int | None,
    actual_byte_size: int,
) -> None:
    if actual_byte_size > MAX_ATTACHMENT_BYTES:
        raise RuntimeError(f"Attachment '{name}' exceeds the 10 MiB limit.")
    if declared_byte_size is not None and declared_byte_size != actual_byte_size:
        raise RuntimeError(
            f"Attachment '{name}' reported {declared_byte_size} bytes "
            f"but contains {actual_byte_size} bytes."
        )


def _is_pdf_attachment(name: str, media_type: str) -> bool:
    return name.strip().lower().endswith(".pdf") or _canonical_media_type(media_type) == "application/pdf"


def _safe_attachment_size(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    text = str(value or "").strip()
    return int(text) if text.isdigit() else None


def _is_text_attachment(name: str, media_type: str) -> bool:
    if media_type.startswith("text/"):
        return True
    if media_type in {"application/json", "application/xml", "application/x-yaml", "text/x-python"}:
        return True
    text_extensions = {
        ".txt", ".md", ".markdown", ".json", ".csv", ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
        ".html", ".css", ".scss", ".sass", ".sql", ".yaml", ".yml", ".xml", ".sh", ".ps1", ".java", ".c",
        ".cc", ".cpp", ".h", ".hpp", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".kts", ".toml", ".ini",
    }
    return any(name.endswith(extension) for extension in text_extensions)


def _suggest_thread_title(prompt: str, *, max_words: int = 6, max_chars: int = 56) -> str:
    cleaned = " ".join(prompt.replace("\n", " ").split()).strip(" -:.,")
    if not cleaned:
        return ""
    words = cleaned.split()
    title = " ".join(words[:max_words]).strip()
    if len(title) > max_chars:
        title = title[:max_chars].rstrip(" -:.,")
    return title or cleaned[:max_chars].rstrip(" -:.,")


def _duplicate_thread_title(value: str) -> str:
    base = value.strip() or "New chat"
    if base.lower().endswith(" copy"):
        return base
    return f"{base} copy"


def _branch_thread_title(value: str) -> str:
    base = value.strip() or "New chat"
    if base.lower().endswith(" branch"):
        return base
    return f"{base} branch"


def _temperature_dropdown_values() -> list[float]:
    return [round(step / 10, 1) for step in range(21)]


def _normalize_search_query(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()


def _build_search_snippet(content: str, normalized_query: str, *, radius: int = 88) -> str:
    cleaned = " ".join(content.split()).strip()
    if not cleaned:
        return ""
    haystack = cleaned.casefold()
    index = haystack.find(normalized_query)
    if index < 0:
        return cleaned[: radius * 2].rstrip()
    start = max(0, index - radius)
    end = min(len(cleaned), index + len(normalized_query) + radius)
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(cleaned) else ""
    return f"{prefix}{cleaned[start:end].strip()}{suffix}"


def _build_thread_search_snippet(*, title: str, last_prompt: str, normalized_query: str) -> str:
    if normalized_query in title.casefold():
        return title
    if last_prompt:
        return _build_search_snippet(last_prompt, normalized_query)
    return title


def _sort_search_results(items: list[dict[str, Any]], *, current_thread: bool) -> list[dict[str, Any]]:
    if current_thread:
        return sorted(
            items,
            key=lambda item: (
                0 if str(item.get("match_type", "")) == "thread" else 1,
                -(int(item.get("history_index", -1) or -1)),
                str(item.get("thread_title", "") or ""),
            ),
        )
    return sorted(
        items,
        key=lambda item: (
            str(item.get("updated_at", "") or ""),
            1 if str(item.get("match_type", "")) == "thread" else 0,
            int(item.get("history_index", -1) or -1),
            str(item.get("thread_title", "") or ""),
        ),
        reverse=True,
    )


def _build_run_diagnostics(artifact: dict[str, Any]) -> dict[str, Any]:
    started_at = _parse_iso_timestamp(str(artifact.get("started_at", "") or ""))
    completed_at = _parse_iso_timestamp(str(artifact.get("completed_at", "") or ""))
    first_token_at = None
    total_compaction_gain = 0
    compaction_events = 0

    for event in artifact.get("events", []):
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type", "") or "")
        if event_type == "token" and first_token_at is None:
            first_token_at = _parse_iso_timestamp(str(event.get("timestamp", "") or ""))
        if event_type == "context_compacted":
            payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
            before_tokens = int(payload.get("history_representation_tokens_before_compaction", 0) or 0)
            after_tokens = int(payload.get("history_representation_tokens_after_compaction", 0) or 0)
            if before_tokens > after_tokens > 0:
                total_compaction_gain += before_tokens - after_tokens
            compaction_events += 1

    answer = str(artifact.get("answer", "") or "")
    output_tokens_estimate = _estimate_text_tokens(answer)
    first_token_latency_ms = _duration_ms(started_at, first_token_at)
    total_duration_ms = _duration_ms(started_at, completed_at)
    generation_duration_ms = _duration_ms(first_token_at, completed_at)
    output_tokens_per_second_estimate = None
    if generation_duration_ms and generation_duration_ms > 0 and output_tokens_estimate > 0:
        output_tokens_per_second_estimate = round(output_tokens_estimate / (generation_duration_ms / 1000.0), 2)

    return {
        "first_token_latency_ms": first_token_latency_ms,
        "total_duration_ms": total_duration_ms,
        "generation_duration_ms": generation_duration_ms,
        "output_tokens_estimate": output_tokens_estimate,
        "output_tokens_per_second_estimate": output_tokens_per_second_estimate,
        "compaction_gain_tokens_estimate": total_compaction_gain or None,
        "compaction_events_count": compaction_events,
    }


def _parse_iso_timestamp(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_ms(start: datetime | None, end: datetime | None) -> int | None:
    if start is None or end is None:
        return None
    return max(0, int((end - start).total_seconds() * 1000))


def _estimate_text_tokens(value: str) -> int:
    cleaned = value.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def _looks_like_embedding_model(model_name: str) -> bool:
    normalized = str(model_name or "").casefold()
    return any(marker in normalized for marker in ("embed", "embedding", "bert"))


def _looks_like_ollama_resource_failure(error_message: str) -> bool:
    normalized = str(error_message or "").casefold()
    return any(
        marker in normalized
        for marker in (
            "cuda",
            "cgo execution",
            "out of memory",
            "memory exhaustion",
            "model failed to load",
            "failed to load",
        )
    )


def _persistent_thread_timeline_events(items: list[dict[str, Any]] | Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [
        item
        for item in items
        if isinstance(item, dict) and str(item.get("type", "")).strip() == "context_compacted"
    ]


def _sorted_thread_timeline_events(items: list[dict[str, Any]] | Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    filtered = [
        item
        for item in items
        if isinstance(item, dict) and str(item.get("type", "")).strip()
    ]
    return sorted(
        filtered,
        key=lambda item: (
            int(item.get("after_message_count", 0) or 0),
            str(item.get("timestamp", "") or ""),
        ),
    )


def _timeline_event_to_history_item(item: dict[str, Any]) -> dict[str, Any]:
    item_type = str(item.get("type", "") or "")
    if item_type == "context_compacted":
        compacted_message_count = int(item.get("compacted_message_count", 0) or 0)
        detected_context_window = int(item.get("detected_context_window", 0) or 0)
        representation_before = int(item.get("history_representation_tokens_before_compaction", 0) or 0)
        representation_after = int(item.get("history_representation_tokens_after_compaction", 0) or 0)
        return {
            "role": "system",
            "kind": "context_compacted",
            "content": "Context compacted",
            "attachments": [],
            "run_id": str(item.get("run_id", "") or ""),
            "timestamp": str(item.get("timestamp", "") or ""),
            "thread_summary": str(item.get("thread_summary", "") or ""),
            "compaction_reason": str(item.get("compaction_reason", "") or ""),
            "compacted_message_count": compacted_message_count,
            "newly_compacted_message_count": int(item.get("newly_compacted_message_count", 0) or 0),
            "detected_context_window": detected_context_window,
            "history_representation_tokens_before_compaction": representation_before,
            "history_representation_tokens_after_compaction": representation_after,
        }

    return {
        "role": "system",
        "kind": item_type,
        "content": "System event",
        "attachments": [],
        "run_id": str(item.get("run_id", "") or ""),
        "timestamp": str(item.get("timestamp", "") or ""),
    }
