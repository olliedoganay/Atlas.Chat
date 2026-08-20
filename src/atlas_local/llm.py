from __future__ import annotations

import concurrent.futures
import json
import threading
from dataclasses import asdict, dataclass, field
from time import monotonic
from typing import Any, Callable
from urllib import error, request
from urllib.parse import urljoin

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from .config import (
    DEFAULT_OLLAMA_URL,
    OPENAI_COMPATIBLE_PROVIDER_DEFAULT_URLS,
    AppConfig,
    chat_provider_label,
    is_ollama_chat_provider,
    normalize_chat_provider,
)
from .local_provider import (
    normalize_local_provider_base_url,
    provider_urlopen,
)


OLLAMA_CONTEXT_WINDOW_PRESETS = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
MIN_OLLAMA_CONTEXT_WINDOW = 1024
MAX_OLLAMA_CONTEXT_WINDOW = 262144
MAX_MODEL_INSPECTION_WORKERS = 8
MAX_MODEL_CATALOG_ENTRIES = 256
MAX_MODEL_CATALOG_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_PROVIDER_JSON_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_PROVIDER_ERROR_RESPONSE_BYTES = 64 * 1024
MAX_PROVIDER_STREAM_EVENT_BYTES = 1024 * 1024
MAX_CATALOG_MODEL_NAME_LENGTH = 200
MAX_OLLAMA_MODEL_SIZE_BYTES = 1024**5
OLLAMA_CHAT_REQUEST_TIMEOUT_SECONDS = 300.0


class _OpenAICompatibleRequestAborted(RuntimeError):
    """Raised when Atlas cancels an in-flight OpenAI-compatible request."""


class _OpenAICompatibleStreamUnsupported(RuntimeError):
    """Raised only when a local provider explicitly cannot produce SSE."""


_STREAM_UNSUPPORTED_HTTP_STATUSES = {400, 404, 405, 406, 415, 422, 501}
_STREAM_PROTOCOL_MARKERS = ("stream", "streaming", "sse", "event-stream")
_STREAM_REJECTION_MARKERS = (
    "not supported",
    "unsupported",
    "not implemented",
    "unrecognized",
    "unknown",
    "invalid",
)


def format_runtime_error(config: AppConfig, exc: Exception, *, chat_model: str | None = None) -> RuntimeError:
    resolved_chat_model = (chat_model or "").strip()
    provider = _config_chat_provider(config)
    provider_label = chat_provider_label(provider)
    base_url = _config_chat_base_url(config)
    model_detail = (
        f"Requested chat_model={resolved_chat_model!r}, "
        if resolved_chat_model
        else "No chat model was selected. "
    )
    original_error = str(exc).strip()
    message = (
        f"{provider_label} request failed. "
        f"{model_detail}"
        f"embed_model={config.embed_model!r}, "
        f"base_url={base_url!r}. "
        f"Make sure {provider_label} is running locally and the required models are available."
    )
    if original_error:
        message = f"{message} Original error: {original_error}"
    normalized_error = original_error.lower()
    if "cuda" in normalized_error:
        message = (
            f"{message} This looks like a local GPU/CUDA runtime failure; restart {provider_label}, "
            "then retry or switch to a smaller/local model if it keeps happening."
        )
    if "out of memory" in normalized_error:
        if is_ollama_chat_provider(provider):
            message = (
                f"{message} Ollama also reported GPU memory exhaustion. Lower the Ollama context window "
                "in Settings > Models or switch to a smaller model before retrying."
            )
        else:
            message = (
                f"{message} The local runtime reported memory exhaustion. Lower that runtime's context setting "
                "or switch to a smaller model before retrying."
            )
    if any(marker in normalized_error for marker in ("context length", "maximum context", "num_ctx", "prompt too long", "too many tokens")):
        message = (
            f"{message} The prompt appears too large for this model context. Compact the thread, lower the "
            "requested context window, or switch to a model with a larger stable context."
        )
    return RuntimeError(message)


@dataclass
class LLMProvider:
    config: AppConfig
    _chat_models: dict[tuple[Any, ...], Any] = field(default_factory=dict, init=False, repr=False)
    _json_chat_models: dict[tuple[Any, ...], Any] = field(default_factory=dict, init=False, repr=False)
    _context_windows: dict[str, tuple[float, int]] = field(default_factory=dict, init=False, repr=False)
    _ollama_context_window: int | None = field(default=None, init=False, repr=False)
    _ollama_context_window_loaded: bool = field(default=False, init=False, repr=False)

    def chat(
        self,
        model: str | None = None,
        *,
        temperature: float | None = None,
        reasoning: bool | str | None = None,
    ) -> Any:
        resolved_model = (model or "").strip()
        if not resolved_model:
            raise RuntimeError(f"Select a local {chat_provider_label(_config_chat_provider(self.config))} model before starting this chat.")
        resolved_temperature = None if temperature is None else float(temperature)
        provider = _config_chat_provider(self.config)
        base_url = _config_chat_base_url(self.config)
        if not is_ollama_chat_provider(provider):
            cache_key = (provider, base_url, resolved_model, resolved_temperature, "chat")
            if cache_key not in self._chat_models:
                self._chat_models[cache_key] = OpenAICompatibleChat(
                    model=resolved_model,
                    base_url=base_url,
                    temperature=resolved_temperature,
                    api_key=_config_chat_api_key(self.config),
                )
            return self._chat_models[cache_key]

        resolved_reasoning = _resolve_reasoning_for_model(resolved_model, reasoning)
        context_window = self.ollama_context_window()
        cache_key = (provider, base_url, resolved_model, resolved_temperature, repr(resolved_reasoning), context_window)
        if cache_key not in self._chat_models:
            options: dict[str, Any] = {
                "model": resolved_model,
                "base_url": base_url,
                "validate_model_on_init": True,
                "client_kwargs": {
                    "follow_redirects": False,
                    "timeout": OLLAMA_CHAT_REQUEST_TIMEOUT_SECONDS,
                    "trust_env": False,
                },
            }
            if resolved_temperature is not None:
                options["temperature"] = resolved_temperature
            if resolved_reasoning is not None:
                options["reasoning"] = resolved_reasoning
            if context_window is not None:
                options["num_ctx"] = context_window
            self._chat_models[cache_key] = ChatOllama(**options)
        return self._chat_models[cache_key]

    def json_chat(self, model: str | None = None) -> Any:
        resolved_model = (model or "").strip()
        if not resolved_model:
            raise RuntimeError(f"Select a local {chat_provider_label(_config_chat_provider(self.config))} model before starting this chat.")
        provider = _config_chat_provider(self.config)
        base_url = _config_chat_base_url(self.config)
        if not is_ollama_chat_provider(provider):
            cache_key = (provider, base_url, resolved_model, "json")
            if cache_key not in self._json_chat_models:
                self._json_chat_models[cache_key] = OpenAICompatibleChat(
                    model=resolved_model,
                    base_url=base_url,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    api_key=_config_chat_api_key(self.config),
                )
            return self._json_chat_models[cache_key]

        context_window = self.ollama_context_window()
        cache_key = (provider, base_url, resolved_model, context_window)
        if cache_key not in self._json_chat_models:
            options: dict[str, Any] = {
                "model": resolved_model,
                "base_url": base_url,
                "temperature": 0.0,
                "format": "json",
                "validate_model_on_init": True,
                "client_kwargs": {
                    "follow_redirects": False,
                    "timeout": OLLAMA_CHAT_REQUEST_TIMEOUT_SECONDS,
                    "trust_env": False,
                },
            }
            if context_window is not None:
                options["num_ctx"] = context_window
            self._json_chat_models[cache_key] = ChatOllama(**options)
        return self._json_chat_models[cache_key]

    def count_message_tokens(self, model: str | None, messages: list[Any]) -> int:
        resolved_model = (model or "").strip()
        if not messages:
            return 0
        if not resolved_model:
            return _approximate_message_tokens(messages)
        if not is_ollama_chat_provider(_config_chat_provider(self.config)):
            return _approximate_message_tokens(messages)
        try:
            counter = getattr(self.chat(resolved_model), "get_num_tokens_from_messages", None)
            if callable(counter):
                counted = int(counter(messages))
                if counted >= 0:
                    return counted
        except Exception:
            pass
        return _approximate_message_tokens(messages)

    def effective_context_window(self, model: str | None = None, *, ttl_seconds: float = 15.0) -> int:
        resolved_model = (model or "").strip()
        if not resolved_model:
            return 0
        if not is_ollama_chat_provider(_config_chat_provider(self.config)):
            return 8192
        override = self.ollama_context_window()
        if override is not None:
            return override
        cached = self._context_windows.get(resolved_model)
        now = monotonic()
        if cached and now - cached[0] < ttl_seconds:
            return cached[1]
        value = resolve_effective_context_window(self.config, resolved_model)
        self._context_windows[resolved_model] = (now, value)
        return value

    def ollama_context_window(self) -> int | None:
        self._load_ollama_context_window()
        return self._ollama_context_window

    def set_ollama_context_window(self, context_window: int | None) -> int | None:
        if not is_ollama_chat_provider(_config_chat_provider(self.config)):
            raise RuntimeError("Context window overrides are only available for Ollama runtimes.")
        if context_window is None:
            saved_value = None
        else:
            parsed = int(context_window)
            if parsed < MIN_OLLAMA_CONTEXT_WINDOW or parsed > MAX_OLLAMA_CONTEXT_WINDOW:
                raise RuntimeError(
                    f"Context window must be between {MIN_OLLAMA_CONTEXT_WINDOW:,} and {MAX_OLLAMA_CONTEXT_WINDOW:,} tokens."
                )
            saved_value = parsed
        self._write_ollama_context_window(saved_value)
        self._ollama_context_window = saved_value
        self._ollama_context_window_loaded = True
        self._clear_model_caches()
        return saved_value

    def loaded_models(self) -> tuple[str, ...]:
        if not is_ollama_chat_provider(_config_chat_provider(self.config)):
            return ()
        return tuple(loaded_ollama_models(self.config))

    def unload_model(self, model: str) -> dict[str, Any]:
        if not is_ollama_chat_provider(_config_chat_provider(self.config)):
            raise RuntimeError("This local provider does not expose an Atlas model-unload operation.")
        payload = unload_ollama_model(self.config, model)
        self._clear_model_caches()
        return payload

    def abort_active_requests(self) -> None:
        for chat_model in list(self._chat_models.values()):
            _close_chat_client(chat_model)
        for chat_model in list(self._json_chat_models.values()):
            _close_chat_client(chat_model)
        self._chat_models.clear()
        self._json_chat_models.clear()

    def _load_ollama_context_window(self) -> None:
        if self._ollama_context_window_loaded:
            return
        data_dir = getattr(self.config, "data_dir", None)
        if data_dir is None:
            self._ollama_context_window = None
            self._ollama_context_window_loaded = True
            return
        path = data_dir / "ollama_context_window.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._ollama_context_window = None
            self._ollama_context_window_loaded = True
            return
        if not isinstance(payload, dict):
            self._ollama_context_window = None
            self._ollama_context_window_loaded = True
            return
        parsed = _parse_positive_int(payload.get("context_window"))
        self._ollama_context_window = (
            parsed
            if parsed and MIN_OLLAMA_CONTEXT_WINDOW <= parsed <= MAX_OLLAMA_CONTEXT_WINDOW
            else None
        )
        self._ollama_context_window_loaded = True

    def _write_ollama_context_window(self, context_window: int | None) -> None:
        data_dir = getattr(self.config, "data_dir", None)
        if data_dir is None:
            raise RuntimeError("Atlas cannot persist the Ollama context window without a data directory.")
        path = data_dir / "ollama_context_window.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if context_window is None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return
        path.write_text(
            json.dumps({"context_window": context_window}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _clear_model_caches(self) -> None:
        self._context_windows.clear()
        for chat_model in list(self._chat_models.values()):
            _close_chat_client(chat_model)
        for chat_model in list(self._json_chat_models.values()):
            _close_chat_client(chat_model)
        self._chat_models.clear()
        self._json_chat_models.clear()


@dataclass
class OpenAICompatibleChat:
    model: str
    base_url: str
    temperature: float | None = None
    response_format: dict[str, Any] | None = None
    api_key: str | None = None
    timeout_seconds: float = 120.0
    _response_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _active_responses: list[Any] = field(default_factory=list, init=False, repr=False)
    _abort_epoch: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = normalize_local_provider_base_url(self.base_url)

    def invoke(self, messages: list[Any]) -> AIMessage:
        request_epoch = self._current_abort_epoch()
        return self._invoke_at_epoch(messages, request_epoch=request_epoch)

    def _invoke_at_epoch(
        self,
        messages: list[Any],
        *,
        request_epoch: int,
    ) -> AIMessage:
        self._raise_if_request_aborted(request_epoch)
        payload = self._chat_payload(messages, stream=False)
        response = _openai_compatible_json_request_required(
            self.base_url,
            "chat/completions",
            timeout_seconds=self.timeout_seconds,
            body=payload,
            api_key=self.api_key,
            on_response_opened=lambda active_response: self._register_active_response_at_epoch(
                active_response,
                request_epoch=request_epoch,
            ),
            on_response_closed=self._unregister_active_response,
        )
        self._raise_if_request_aborted(request_epoch)
        return AIMessage(content=_openai_compatible_response_text(response))

    def stream(self, messages: list[Any]):
        request_epoch = self._current_abort_epoch()
        payload = self._chat_payload(messages, stream=True)
        emitted = False
        try:
            for delta in _stream_openai_compatible_chat(
                self.base_url,
                payload,
                timeout_seconds=self.timeout_seconds,
                api_key=self.api_key,
                on_response_opened=lambda active_response: self._register_active_response_at_epoch(
                    active_response,
                    request_epoch=request_epoch,
                ),
                on_response_closed=self._unregister_active_response,
            ):
                self._raise_if_request_aborted(request_epoch)
                if isinstance(delta, tuple):
                    content, reasoning_content = delta
                else:
                    # Preserve compatibility with simple/testing adapters that
                    # yield only answer text.
                    content, reasoning_content = str(delta), ""
                if content or reasoning_content:
                    emitted = True
                    additional_kwargs = (
                        {"reasoning_content": reasoning_content}
                        if reasoning_content
                        else {}
                    )
                    yield AIMessage(
                        content=content,
                        additional_kwargs=additional_kwargs,
                    )
            if emitted:
                return
        except Exception as exc:
            if self._request_was_aborted(request_epoch):
                raise _OpenAICompatibleRequestAborted(
                    "OpenAI-compatible request was aborted."
                ) from exc
            if emitted:
                # A non-streaming retry after partial output would generate a
                # second answer and append it to the already-emitted text.
                # Surface the interrupted stream instead; callers can preserve
                # the partial response and report the transport failure.
                raise
            if not isinstance(exc, _OpenAICompatibleStreamUnsupported):
                raise
            # A small number of local servers explicitly reject SSE while still
            # exposing non-streaming chat completions. Only that protocol-level
            # incompatibility is eligible for a second request.
        self._raise_if_request_aborted(request_epoch)
        response = self._invoke_at_epoch(messages, request_epoch=request_epoch)
        if response.content:
            yield response

    def abort(self) -> None:
        with self._response_lock:
            self._abort_epoch += 1
            responses = list(self._active_responses)
        for response in responses:
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except OSError:
                    pass

    def _register_active_response(self, response: Any) -> None:
        self._register_active_response_at_epoch(
            response,
            request_epoch=self._current_abort_epoch(),
        )

    def _register_active_response_at_epoch(
        self,
        response: Any,
        *,
        request_epoch: int,
    ) -> None:
        with self._response_lock:
            aborted = self._abort_epoch != request_epoch
            if not aborted:
                self._active_responses.append(response)
        if not aborted:
            return
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except OSError:
                pass
        raise _OpenAICompatibleRequestAborted(
            "OpenAI-compatible request was aborted."
        )

    def _unregister_active_response(self, response: Any) -> None:
        with self._response_lock:
            self._active_responses = [
                active_response
                for active_response in self._active_responses
                if active_response is not response
            ]

    def _current_abort_epoch(self) -> int:
        with self._response_lock:
            return self._abort_epoch

    def _request_was_aborted(self, request_epoch: int) -> bool:
        with self._response_lock:
            return self._abort_epoch != request_epoch

    def _raise_if_request_aborted(self, request_epoch: int) -> None:
        if self._request_was_aborted(request_epoch):
            raise _OpenAICompatibleRequestAborted(
                "OpenAI-compatible request was aborted."
            )

    def get_num_tokens_from_messages(self, messages: list[Any]) -> int:
        return _approximate_message_tokens(messages)

    def _chat_payload(self, messages: list[Any], *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_openai_compatible(messages),
            "stream": stream,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.response_format is not None:
            payload["response_format"] = self.response_format
        return payload


@dataclass(frozen=True)
class OllamaModelInfo:
    name: str
    size_bytes: int | None = None
    family: str = ""
    families: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    supports_images: bool = False
    supports_reasoning: bool = False
    reasoning_mode_strategy: str = "none"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["families"] = list(self.families)
        payload["capabilities"] = list(self.capabilities)
        return payload


@dataclass(frozen=True)
class OllamaCatalogSnapshot:
    models: tuple[OllamaModelInfo, ...] = ()
    ollama_online: bool = False
    has_local_models: bool = False
    source: str = "fallback"
    provider: str = "ollama"
    provider_label: str = "Ollama"
    provider_base_url: str = ""
    provider_online: bool = False
    supports_context_window: bool = True
    supports_model_unload: bool = True


def inspect_local_models(config: AppConfig, *, timeout_seconds: float = 3.0) -> OllamaCatalogSnapshot:
    provider = _config_chat_provider(config)
    if is_ollama_chat_provider(provider):
        return inspect_local_ollama_models(config, timeout_seconds=timeout_seconds)
    return inspect_openai_compatible_models(config, timeout_seconds=timeout_seconds)


def list_local_ollama_models(config: AppConfig, *, timeout_seconds: float = 3.0) -> list[str]:
    return [item.name for item in list_local_ollama_model_info(config, timeout_seconds=timeout_seconds)]


def loaded_ollama_models(config: AppConfig, *, timeout_seconds: float = 2.0) -> list[str]:
    payload = _ollama_json_request(config, "api/ps", timeout_seconds=timeout_seconds)
    models = payload.get("models", [])
    if not isinstance(models, list):
        return []
    loaded: list[str] = []
    seen: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if (
            not name
            or len(name) > MAX_CATALOG_MODEL_NAME_LENGTH
            or name in seen
        ):
            continue
        seen.add(name)
        loaded.append(name)
    return loaded


def unload_ollama_model(config: AppConfig, model: str, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
    resolved_model = model.strip()
    if not resolved_model:
        raise RuntimeError("Model name is required before Atlas can stop an Ollama model.")
    payload = _ollama_json_request_required(
        config,
        "api/generate",
        timeout_seconds=timeout_seconds,
        body={
            "model": resolved_model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        },
    )
    return {
        "status": "unloaded",
        "model": resolved_model,
        "ollama": payload,
    }


def list_local_ollama_model_info(config: AppConfig, *, timeout_seconds: float = 3.0) -> list[OllamaModelInfo]:
    return list(inspect_local_ollama_models(config, timeout_seconds=timeout_seconds).models)


def inspect_local_ollama_models(config: AppConfig, *, timeout_seconds: float = 3.0) -> OllamaCatalogSnapshot:
    base_url = normalize_local_provider_base_url(config.ollama_url)
    endpoint = urljoin(f"{base_url}/", "api/tags")
    try:
        with provider_urlopen(endpoint, timeout=timeout_seconds) as response:
            raw_payload = response.read(MAX_MODEL_CATALOG_RESPONSE_BYTES + 1)
            if len(raw_payload) > MAX_MODEL_CATALOG_RESPONSE_BYTES:
                raise ValueError("Ollama model catalog is too large.")
            decoded_payload = json.loads(raw_payload.decode("utf-8"))
            if not isinstance(decoded_payload, dict):
                raise ValueError("Ollama model catalog must be a JSON object.")
            payload: dict[str, Any] = decoded_payload
    except (
        error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ):
        return OllamaCatalogSnapshot(
            provider="ollama",
            provider_label=chat_provider_label("ollama"),
            provider_base_url=base_url,
            supports_context_window=True,
            supports_model_unload=True,
        )

    models = payload.get("models", [])
    if not isinstance(models, list):
        models = []
    model_payloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in models[:MAX_MODEL_CATALOG_ENTRIES]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if (
            not name
            or len(name) > MAX_CATALOG_MODEL_NAME_LENGTH
            or name in seen
        ):
            continue
        seen.add(name)
        model_payloads.append(item)

    entries: list[OllamaModelInfo] = []
    if model_payloads:
        worker_count = min(MAX_MODEL_INSPECTION_WORKERS, len(model_payloads))
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="atlas-model-inspect",
        )
        future_items = {
            executor.submit(
                _inspect_ollama_model_info,
                config,
                item,
                timeout_seconds=min(timeout_seconds, 1.5),
            ): item
            for item in model_payloads
        }
        try:
            done, pending = concurrent.futures.wait(
                future_items,
                timeout=max(0.05, float(timeout_seconds)),
            )
            entries_by_name: dict[str, OllamaModelInfo] = {}
            for future in done:
                try:
                    inspected = future.result()
                except Exception:
                    inspected = None
                if inspected is not None:
                    entries_by_name[inspected.name] = inspected
            for future in pending:
                future.cancel()
            for item in model_payloads:
                name = str(item.get("name", "") or "").strip()
                if name in entries_by_name:
                    continue
                fallback = _ollama_model_info_from_payload(item, {})
                if fallback is not None:
                    entries_by_name[name] = fallback
            entries.extend(entries_by_name.values())
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    entries.sort(key=lambda item: item.name)
    return OllamaCatalogSnapshot(
        models=tuple(entries),
        ollama_online=True,
        has_local_models=bool(entries),
        source="ollama",
        provider="ollama",
        provider_label=chat_provider_label("ollama"),
        provider_base_url=base_url,
        provider_online=True,
        supports_context_window=True,
        supports_model_unload=True,
    )


def _inspect_ollama_model_info(
    config: AppConfig,
    tag_payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> OllamaModelInfo | None:
    name = str(tag_payload.get("name", "")).strip()
    show_payload = _ollama_json_request(
        config,
        "api/show",
        timeout_seconds=timeout_seconds,
        body={"model": name},
    )
    return _ollama_model_info_from_payload(tag_payload, show_payload)


def _ollama_model_info_from_payload(
    tag_payload: dict[str, Any],
    show_payload: dict[str, Any],
) -> OllamaModelInfo | None:
    name = str(tag_payload.get("name", "")).strip()
    model_payload = _merge_ollama_model_payload(tag_payload, show_payload)
    if not _is_chat_capable_model(model_payload):
        return None
    details = (
        model_payload.get("details", {})
        if isinstance(model_payload.get("details"), dict)
        else {}
    )
    family = str(details.get("family", "")).strip()
    raw_families = details.get("families", [])
    if not isinstance(raw_families, list):
        raw_families = []
    families = tuple(
        str(entry).strip()
        for entry in raw_families
        if str(entry).strip()
    )
    capabilities = _payload_capabilities(model_payload)
    return OllamaModelInfo(
        name=name,
        size_bytes=_parse_ollama_model_size(tag_payload.get("size")),
        family=family,
        families=families,
        capabilities=capabilities,
        supports_images=_is_vision_capable_model(model_payload),
        supports_reasoning=_is_reasoning_capable_model(model_payload),
        reasoning_mode_strategy=_reasoning_mode_strategy(model_payload),
    )


def inspect_openai_compatible_models(config: AppConfig, *, timeout_seconds: float = 3.0) -> OllamaCatalogSnapshot:
    provider = _config_chat_provider(config)
    base_url = _config_chat_base_url(config)
    payload = _openai_compatible_json_request(
        base_url,
        "models",
        timeout_seconds=timeout_seconds,
        api_key=_config_chat_api_key(config),
        max_response_bytes=MAX_MODEL_CATALOG_RESPONSE_BYTES,
    )
    if not payload:
        return OllamaCatalogSnapshot(
            source=provider,
            provider=provider,
            provider_label=chat_provider_label(provider),
            provider_base_url=base_url,
            provider_online=False,
            supports_context_window=False,
            supports_model_unload=False,
        )

    raw_models = payload.get("data", payload.get("models", []))
    entries: list[OllamaModelInfo] = []
    seen: set[str] = set()
    if isinstance(raw_models, list):
        for item in raw_models[:MAX_MODEL_CATALOG_ENTRIES]:
            model_payload = item if isinstance(item, dict) else {"id": item}
            name = str(
                model_payload.get("id")
                or model_payload.get("name")
                or model_payload.get("model")
                or ""
            ).strip()
            if (
                not name
                or len(name) > MAX_CATALOG_MODEL_NAME_LENGTH
                or name in seen
            ):
                continue
            seen.add(name)
            inferred_payload = {
                "name": name,
                "model": name,
                "details": {
                    "family": str(model_payload.get("owned_by") or model_payload.get("family") or "").strip(),
                    "families": [],
                },
                "capabilities": model_payload.get("capabilities", []),
            }
            if not _is_chat_capable_model(inferred_payload):
                continue
            entries.append(
                OllamaModelInfo(
                    name=name,
                    family=str(inferred_payload["details"]["family"]),
                    capabilities=_payload_capabilities(inferred_payload),
                    supports_images=_is_vision_capable_model(inferred_payload),
                    supports_reasoning=_is_reasoning_capable_model(inferred_payload),
                    reasoning_mode_strategy=_reasoning_mode_strategy(inferred_payload),
                )
            )
    entries.sort(key=lambda item: item.name)
    return OllamaCatalogSnapshot(
        models=tuple(entries),
        ollama_online=False,
        has_local_models=bool(entries),
        source=provider,
        provider=provider,
        provider_label=chat_provider_label(provider),
        provider_base_url=base_url,
        provider_online=True,
        supports_context_window=False,
        supports_model_unload=False,
    )


def resolve_effective_context_window(
    config: AppConfig,
    model: str,
    *,
    timeout_seconds: float = 2.5,
    fallback: int = 8192,
) -> int:
    resolved_model = model.strip()
    if not resolved_model:
        return 0
    ps_payload = _ollama_json_request(config, "api/ps", timeout_seconds=timeout_seconds)
    context_from_ps = _context_from_ps_payload(ps_payload, resolved_model)
    if context_from_ps:
        return context_from_ps

    show_payload = _ollama_json_request(
        config,
        "api/show",
        timeout_seconds=timeout_seconds,
        body={"model": resolved_model},
    )
    context_from_show = _context_from_show_payload(show_payload)
    if context_from_show:
        return context_from_show
    return fallback


def _ollama_json_request(
    config: AppConfig,
    endpoint: str,
    *,
    timeout_seconds: float,
    body: dict[str, Any] | None = None,
    max_response_bytes: int = MAX_MODEL_CATALOG_RESPONSE_BYTES,
) -> dict[str, Any]:
    base_url = normalize_local_provider_base_url(config.ollama_url)
    url = urljoin(f"{base_url}/", endpoint)
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request_object = request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    try:
        with provider_urlopen(request_object, timeout=timeout_seconds) as response:
            payload = json.loads(
                _read_bounded_response(
                    response,
                    max_bytes=max_response_bytes,
                    label="Ollama JSON response",
                ).decode("utf-8")
            )
            return payload if isinstance(payload, dict) else {}
    except (
        error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        OSError,
        ValueError,
    ):
        return {}


def _ollama_json_request_required(
    config: AppConfig,
    endpoint: str,
    *,
    timeout_seconds: float,
    body: dict[str, Any] | None = None,
    max_response_bytes: int = MAX_MODEL_CATALOG_RESPONSE_BYTES,
) -> dict[str, Any]:
    base_url = normalize_local_provider_base_url(config.ollama_url)
    url = urljoin(f"{base_url}/", endpoint)
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request_object = request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    try:
        with provider_urlopen(request_object, timeout=timeout_seconds) as response:
            payload = json.loads(
                _read_bounded_response(
                    response,
                    max_bytes=max_response_bytes,
                    label="Ollama JSON response",
                ).decode("utf-8")
            )
            return payload if isinstance(payload, dict) else {}
    except error.HTTPError as exc:
        detail = _read_bounded_error_detail(exc)
        reason = detail or str(exc)
        raise RuntimeError(f"Ollama could not stop the model through {endpoint}: {reason}") from exc
    except (
        error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        OSError,
        ValueError,
    ) as exc:
        raise RuntimeError(f"Ollama could not stop the model through {endpoint}: {exc}") from exc


def _openai_compatible_json_request(
    base_url: str,
    endpoint: str,
    *,
    timeout_seconds: float,
    body: dict[str, Any] | None = None,
    api_key: str | None = None,
    max_response_bytes: int = MAX_PROVIDER_JSON_RESPONSE_BYTES,
) -> dict[str, Any]:
    try:
        return _openai_compatible_json_request_required(
            base_url,
            endpoint,
            timeout_seconds=timeout_seconds,
            body=body,
            api_key=api_key,
            max_response_bytes=max_response_bytes,
        )
    except Exception:
        return {}


def _openai_compatible_json_request_required(
    base_url: str,
    endpoint: str,
    *,
    timeout_seconds: float,
    body: dict[str, Any] | None = None,
    api_key: str | None = None,
    max_response_bytes: int = MAX_PROVIDER_JSON_RESPONSE_BYTES,
    on_response_opened: Callable[[Any], None] | None = None,
    on_response_closed: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    resolved_base_url = normalize_local_provider_base_url(base_url)
    url = urljoin(f"{resolved_base_url}/", endpoint.lstrip("/"))
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request_object = request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    try:
        with provider_urlopen(request_object, timeout=timeout_seconds) as response:
            if on_response_opened is not None:
                on_response_opened(response)
            try:
                payload = json.loads(
                    _read_bounded_response(
                        response,
                        max_bytes=max_response_bytes,
                        label="OpenAI-compatible JSON response",
                    ).decode("utf-8")
                )
                return payload if isinstance(payload, dict) else {}
            finally:
                if on_response_closed is not None:
                    on_response_closed(response)
    except error.HTTPError as exc:
        detail = _read_bounded_error_detail(exc)
        reason = detail or str(exc)
        raise RuntimeError(f"OpenAI-compatible local request to {endpoint} failed: {reason}") from exc
    except (
        error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        OSError,
        ValueError,
    ) as exc:
        raise RuntimeError(f"OpenAI-compatible local request to {endpoint} failed: {exc}") from exc


def _stream_openai_compatible_chat(
    base_url: str,
    body: dict[str, Any],
    *,
    timeout_seconds: float,
    api_key: str | None = None,
    on_response_opened: Callable[[Any], None] | None = None,
    on_response_closed: Callable[[Any], None] | None = None,
):
    resolved_base_url = normalize_local_provider_base_url(base_url)
    url = urljoin(f"{resolved_base_url}/", "chat/completions")
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request_object = request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with provider_urlopen(request_object, timeout=timeout_seconds) as response:
            if on_response_opened is not None:
                on_response_opened(response)
            saw_sse_event = False
            saw_non_sse_payload = False
            try:
                for raw_line in _bounded_response_lines(
                    response,
                    max_line_bytes=MAX_PROVIDER_STREAM_EVENT_BYTES,
                    label="OpenAI-compatible stream event",
                ):
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        saw_non_sse_payload = True
                        continue
                    saw_sse_event = True
                    data = line.partition(":")[2].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    content, reasoning_content = _openai_compatible_delta_parts(payload)
                    if content or reasoning_content:
                        yield content, reasoning_content
                if saw_non_sse_payload and not saw_sse_event:
                    raise _OpenAICompatibleStreamUnsupported(
                        "The local provider returned a non-SSE response to a streaming request."
                    )
            finally:
                if on_response_closed is not None:
                    on_response_closed(response)
    except error.HTTPError as exc:
        detail = _read_bounded_error_detail(exc)
        if _is_explicit_stream_rejection(exc.code, detail):
            raise _OpenAICompatibleStreamUnsupported(
                detail or "The local provider explicitly rejected streaming."
            ) from exc
        reason = detail or str(exc)
        raise RuntimeError(
            f"OpenAI-compatible local streaming request failed: {reason}"
        ) from exc


def _is_explicit_stream_rejection(status_code: int, detail: str) -> bool:
    if status_code not in _STREAM_UNSUPPORTED_HTTP_STATUSES:
        return False
    normalized = detail.strip().lower()
    return (
        any(marker in normalized for marker in _STREAM_PROTOCOL_MARKERS)
        and any(marker in normalized for marker in _STREAM_REJECTION_MARKERS)
    )


def _read_bounded_response(
    response: Any,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    resolved_limit = max(1, int(max_bytes))
    headers = getattr(response, "headers", None)
    content_length = None
    if headers is not None:
        try:
            content_length = headers.get("Content-Length")
        except (AttributeError, TypeError):
            content_length = None
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except (TypeError, ValueError):
            declared_length = None
        if declared_length is not None and declared_length > resolved_limit:
            raise ValueError(f"{label} exceeded the {resolved_limit:,}-byte limit.")
    payload = response.read(resolved_limit + 1)
    if len(payload) > resolved_limit:
        raise ValueError(f"{label} exceeded the {resolved_limit:,}-byte limit.")
    return payload


def _read_bounded_error_detail(response: Any) -> str:
    try:
        payload = response.read(MAX_PROVIDER_ERROR_RESPONSE_BYTES + 1)
    except OSError:
        return ""
    if len(payload) > MAX_PROVIDER_ERROR_RESPONSE_BYTES:
        return "The provider returned an oversized error response."
    return payload.decode("utf-8", errors="replace").strip()


def _bounded_response_lines(
    response: Any,
    *,
    max_line_bytes: int,
    label: str,
):
    resolved_limit = max(1, int(max_line_bytes))
    readline = getattr(response, "readline", None)
    if callable(readline):
        while True:
            raw_line = readline(resolved_limit + 1)
            if not raw_line:
                return
            if len(raw_line) > resolved_limit:
                raise ValueError(f"{label} exceeded the {resolved_limit:,}-byte limit.")
            yield raw_line
        return

    for raw_line in response:
        if len(raw_line) > resolved_limit:
            raise ValueError(f"{label} exceeded the {resolved_limit:,}-byte limit.")
        yield raw_line


def _context_from_ps_payload(payload: dict[str, Any], model: str) -> int | None:
    models = payload.get("models", [])
    if not isinstance(models, list):
        return None
    target = model.strip().lower()
    for item in models:
        if not isinstance(item, dict):
            continue
        names = {
            str(item.get("name", "")).strip().lower(),
            str(item.get("model", "")).strip().lower(),
        }
        if target not in names:
            continue
        candidates = (
            item.get("context_length"),
            item.get("num_ctx"),
            item.get("options", {}).get("num_ctx") if isinstance(item.get("options"), dict) else None,
            item.get("details", {}).get("context_length") if isinstance(item.get("details"), dict) else None,
        )
        for value in candidates:
            parsed = _parse_positive_int(value)
            if parsed:
                return parsed
    return None


def _context_from_show_payload(payload: dict[str, Any]) -> int | None:
    parameters = payload.get("parameters")
    if isinstance(parameters, str):
        for line in parameters.splitlines():
            key, _, value = line.partition(" ")
            if key.strip().lower() == "num_ctx":
                parsed = _parse_positive_int(value)
                if parsed:
                    return parsed
    return None


def _parse_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    text = str(value or "").strip()
    if not text:
        return None
    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        return None
    parsed = int(digits)
    return parsed if parsed > 0 else None


def _parse_ollama_model_size(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        return None
    if parsed <= 0 or parsed > MAX_OLLAMA_MODEL_SIZE_BYTES:
        return None
    return parsed


def _merge_ollama_model_payload(tag_payload: dict[str, Any], show_payload: dict[str, Any]) -> dict[str, Any]:
    if not show_payload:
        return dict(tag_payload)
    payload = dict(tag_payload)
    for key in ("capabilities", "details", "model_info", "model", "parameters"):
        if key in show_payload:
            payload[key] = show_payload[key]
    return payload


def _is_chat_capable_model(payload: dict[str, Any]) -> bool:
    if _payload_declares_capabilities(payload):
        capabilities = set(_payload_capabilities(payload))
        return bool(capabilities.intersection({"chat", "completion"}))

    name = str(payload.get("name", "")).lower()
    model = str(payload.get("model", "")).lower()
    details = payload.get("details", {}) if isinstance(payload.get("details"), dict) else {}
    family = str(details.get("family", "")).lower()
    families = [str(item).lower() for item in details.get("families", []) if str(item).strip()]
    combined = " ".join([name, model, family, " ".join(families)])
    blocked_markers = ("embed", "embedding", "bert")
    return not any(marker in combined for marker in blocked_markers)


def _is_vision_capable_model(payload: dict[str, Any]) -> bool:
    if _payload_declares_capabilities(payload):
        return _payload_has_capability(payload, "vision")

    if _payload_has_capability(payload, "vision") or _model_info_indicates_vision(payload):
        return True

    name = str(payload.get("name", "")).lower()
    model = str(payload.get("model", "")).lower()
    details = payload.get("details", {}) if isinstance(payload.get("details"), dict) else {}
    family = str(details.get("family", "")).lower()
    families = [str(item).lower() for item in details.get("families", []) if str(item).strip()]
    combined = " ".join([name, model, family, " ".join(families)])
    if _is_qwen36_multimodal_fallback(combined):
        return True
    vision_markers = (
        "vision",
        "llava",
        "vl",
        "minicpm-v",
        "bakllava",
        "moondream",
        "gemma3",
        "gemma4",
        "qwen2.5vl",
        "qwen2-vl",
        "qwen-vl",
    )
    return any(marker in combined for marker in vision_markers)


def _payload_capabilities(payload: dict[str, Any]) -> tuple[str, ...]:
    capabilities = payload.get("capabilities", [])
    if not isinstance(capabilities, list):
        return ()
    return tuple(str(item).strip().lower() for item in capabilities if str(item).strip())


def _payload_declares_capabilities(payload: dict[str, Any]) -> bool:
    capabilities = payload.get("capabilities")
    return isinstance(capabilities, list) and bool(capabilities)


def _payload_has_capability(payload: dict[str, Any], capability: str) -> bool:
    normalized = capability.strip().lower()
    return any(item == normalized for item in _payload_capabilities(payload))


def _model_info_indicates_vision(payload: dict[str, Any]) -> bool:
    model_info = payload.get("model_info", {})
    if not isinstance(model_info, dict):
        return False
    descriptor = " ".join(str(key).lower() for key in model_info)
    markers = (".vision.", ".vision_", ".mm.", ".image_", ".image.")
    return any(marker in descriptor for marker in markers)


def _is_qwen36_multimodal_fallback(descriptor: str) -> bool:
    qwen36_markers = ("qwen3.6", "qwen 3.6", "qwen-3.6")
    text_only_markers = ("coding", "coder", "mlx")
    return any(marker in descriptor for marker in qwen36_markers) and not any(
        marker in descriptor for marker in text_only_markers
    )


def _reasoning_mode_strategy(payload: dict[str, Any]) -> str:
    descriptor = _normalized_model_descriptor(payload)
    if _descriptor_explicitly_disables_reasoning(descriptor):
        return "none"
    if _payload_declares_capabilities(payload):
        if not _payload_has_reasoning_capability(payload):
            return "none"
        return "levels" if "gpt-oss" in descriptor else "boolean"
    if _payload_has_reasoning_capability(payload):
        return "levels" if "gpt-oss" in descriptor else "boolean"
    return _reasoning_mode_strategy_from_descriptor(descriptor)


def _is_reasoning_capable_model(payload: dict[str, Any]) -> bool:
    return _reasoning_mode_strategy(payload) != "none"


def _normalized_model_descriptor(payload: dict[str, Any]) -> str:
    name = str(payload.get("name", "")).lower()
    model = str(payload.get("model", "")).lower()
    details = payload.get("details", {}) if isinstance(payload.get("details"), dict) else {}
    family = str(details.get("family", "")).lower()
    families = [str(item).lower() for item in details.get("families", []) if str(item).strip()]
    return " ".join([name, model, family, " ".join(families)])


def _normalize_reasoning_value(value: bool | str | None) -> bool | str | None:
    if isinstance(value, bool) or value is None:
        return value
    normalized = str(value).strip().lower()
    if normalized in {"", "default", "none"}:
        return None
    if normalized in {"true", "on"}:
        return True
    if normalized in {"false", "off"}:
        return False
    if normalized in {"low", "medium", "high"}:
        return normalized
    return None


def _resolve_reasoning_for_model(model: str, value: bool | str | None) -> bool | str | None:
    strategy = _reasoning_mode_strategy_from_descriptor(model.strip().lower())
    normalized = _normalize_reasoning_value(value)
    if strategy == "levels":
        if normalized in {"low", "medium", "high"}:
            return normalized
        if normalized is True:
            return "medium"
        return None
    if strategy == "boolean":
        if normalized in {"low", "medium", "high"}:
            return True
        return normalized
    return None


def _reasoning_mode_strategy_from_descriptor(descriptor: str) -> str:
    if _descriptor_explicitly_disables_reasoning(descriptor):
        return "none"
    reasoning_markers = (
        "qwen3",
        "qwen 3",
        "qwen-3",
        "qwen3.5",
        "gpt-oss",
        "deepseek-r1",
        "deepseek r1",
        "deepseek-v3.1",
        "deepseek v3.1",
        "deepseek-v31",
        "deepseek v31",
    )
    if "gpt-oss" in descriptor:
        return "levels"
    if any(marker in descriptor for marker in reasoning_markers):
        return "boolean"
    return "none"


def _payload_has_reasoning_capability(payload: dict[str, Any]) -> bool:
    reasoning_capabilities = {"thinking", "reasoning"}
    return any(capability in reasoning_capabilities for capability in _payload_capabilities(payload))


def _descriptor_explicitly_disables_reasoning(descriptor: str) -> bool:
    if any(marker in descriptor for marker in ("qwen3-coder", "qwen3 coder", "qwen3coder")):
        return True
    qwen36_markers = ("qwen3.6", "qwen 3.6", "qwen-3.6")
    qwen36_text_only_markers = ("coding", "coder", "mlx")
    return any(marker in descriptor for marker in qwen36_markers) and any(
        marker in descriptor for marker in qwen36_text_only_markers
    )


def _config_chat_provider(config: AppConfig) -> str:
    return normalize_chat_provider(getattr(config, "chat_provider", "ollama"))


def _config_chat_base_url(config: AppConfig) -> str:
    provider = _config_chat_provider(config)
    value = str(getattr(config, "chat_base_url", "") or "").strip()
    if value:
        return normalize_local_provider_base_url(value)
    if is_ollama_chat_provider(provider):
        return normalize_local_provider_base_url(
            str(
                getattr(config, "ollama_url", DEFAULT_OLLAMA_URL)
                or DEFAULT_OLLAMA_URL
            ).strip()
        )
    return normalize_local_provider_base_url(
        OPENAI_COMPATIBLE_PROVIDER_DEFAULT_URLS.get(
            provider,
            OPENAI_COMPATIBLE_PROVIDER_DEFAULT_URLS["openai-compatible"],
        )
    )


def _config_chat_api_key(config: AppConfig) -> str | None:
    value = getattr(config, "chat_api_key", None)
    text = str(value or "").strip()
    return text or None


def _messages_to_openai_compatible(messages: list[Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = _openai_role_for_message(message)
        converted.append({"role": role, "content": _openai_content_for_message(getattr(message, "content", message))})
    return converted


def _openai_role_for_message(message: Any) -> str:
    message_type = str(getattr(message, "type", "") or "").lower()
    if message_type in {"system", "human", "ai"}:
        return {"system": "system", "human": "user", "ai": "assistant"}[message_type]
    class_name = type(message).__name__.lower()
    if "system" in class_name:
        return "system"
    if "human" in class_name or "user" in class_name:
        return "user"
    if "ai" in class_name or "assistant" in class_name:
        return "assistant"
    return "user"


def _openai_content_for_message(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[Any] = []
    for item in content:
        if isinstance(item, str):
            parts.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            parts.append({"type": "text", "text": str(item)})
            continue
        item_type = str(item.get("type", "")).strip().lower()
        if item_type == "text":
            parts.append({"type": "text", "text": str(item.get("text", ""))})
        elif item_type == "image_url":
            parts.append({"type": "image_url", "image_url": item.get("image_url", {})})
        else:
            parts.append(item)
    return parts


def _openai_compatible_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message", {})
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return _text_from_openai_content_parts(content)
    text = first.get("text", "")
    return text if isinstance(text, str) else ""


def _openai_compatible_delta_text(payload: dict[str, Any]) -> str:
    return _openai_compatible_delta_parts(payload)[0]


def _openai_compatible_delta_parts(payload: dict[str, Any]) -> tuple[str, str]:
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return "", ""
    first = choices[0]
    if not isinstance(first, dict):
        return "", ""
    delta = first.get("delta", {})
    if isinstance(delta, dict):
        content = delta.get("content", "")
        reasoning_content = next(
            (
                str(delta[key])
                for key in ("reasoning_content", "reasoning", "thinking")
                if isinstance(delta.get(key), str) and delta.get(key)
            ),
            "",
        )
        if isinstance(content, str):
            return content, reasoning_content
        if isinstance(content, list):
            return _text_from_openai_content_parts(content), reasoning_content
    message = first.get("message", {})
    if isinstance(message, dict):
        content = message.get("content", "")
        reasoning_content = next(
            (
                str(message[key])
                for key in ("reasoning_content", "reasoning", "thinking")
                if isinstance(message.get(key), str) and message.get(key)
            ),
            "",
        )
        if isinstance(content, str):
            return content, reasoning_content
    return "", ""


def _text_from_openai_content_parts(parts: list[Any]) -> str:
    texts: list[str] = []
    for item in parts:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "".join(texts)


def _close_chat_client(chat_model: Any) -> None:
    abort = getattr(chat_model, "abort", None)
    if callable(abort):
        try:
            abort()
        except Exception:
            pass
    client_wrapper = getattr(chat_model, "_client", None)
    transport_client = getattr(client_wrapper, "_client", None)
    close = getattr(transport_client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _approximate_message_tokens(messages: list[Any]) -> int:
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
