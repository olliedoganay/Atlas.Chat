from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from time import monotonic
from typing import Any
from urllib import error, request
from urllib.parse import urljoin

from langchain_ollama import ChatOllama

from .config import AppConfig


OLLAMA_CONTEXT_WINDOW_PRESETS = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
MIN_OLLAMA_CONTEXT_WINDOW = 1024
MAX_OLLAMA_CONTEXT_WINDOW = 262144


def format_runtime_error(config: AppConfig, exc: Exception, *, chat_model: str | None = None) -> RuntimeError:
    resolved_chat_model = (chat_model or "").strip()
    model_detail = (
        f"Requested chat_model={resolved_chat_model!r}, "
        if resolved_chat_model
        else "No chat model was selected. "
    )
    original_error = str(exc).strip()
    message = (
        "Ollama request failed. "
        f"{model_detail}"
        f"embed_model={config.embed_model!r}, "
        f"base_url={config.ollama_url!r}. "
        "Make sure Ollama is running and the required models are pulled."
    )
    if original_error:
        message = f"{message} Original error: {original_error}"
    if "cuda" in original_error.lower():
        message = (
            f"{message} This looks like an Ollama GPU/CUDA runtime failure; restart Ollama, "
            "then retry or switch to a smaller/local model if it keeps happening."
        )
    return RuntimeError(message)


@dataclass
class LLMProvider:
    config: AppConfig
    _chat_models: dict[tuple[str, float | None, str, int | None], ChatOllama] = field(default_factory=dict, init=False, repr=False)
    _json_chat_models: dict[tuple[str, int | None], ChatOllama] = field(default_factory=dict, init=False, repr=False)
    _context_windows: dict[str, tuple[float, int]] = field(default_factory=dict, init=False, repr=False)
    _ollama_context_window: int | None = field(default=None, init=False, repr=False)
    _ollama_context_window_loaded: bool = field(default=False, init=False, repr=False)

    def chat(
        self,
        model: str | None = None,
        *,
        temperature: float | None = None,
        reasoning: bool | str | None = None,
    ) -> ChatOllama:
        resolved_model = (model or "").strip()
        if not resolved_model:
            raise RuntimeError("Select a local Ollama model before starting this chat.")
        resolved_temperature = None if temperature is None else float(temperature)
        resolved_reasoning = _resolve_reasoning_for_model(resolved_model, reasoning)
        context_window = self.ollama_context_window()
        cache_key = (resolved_model, resolved_temperature, repr(resolved_reasoning), context_window)
        if cache_key not in self._chat_models:
            options: dict[str, Any] = {
                "model": resolved_model,
                "base_url": self.config.ollama_url,
                "validate_model_on_init": True,
            }
            if resolved_temperature is not None:
                options["temperature"] = resolved_temperature
            if resolved_reasoning is not None:
                options["reasoning"] = resolved_reasoning
            if context_window is not None:
                options["num_ctx"] = context_window
            self._chat_models[cache_key] = ChatOllama(**options)
        return self._chat_models[cache_key]

    def json_chat(self, model: str | None = None) -> ChatOllama:
        resolved_model = (model or "").strip()
        if not resolved_model:
            raise RuntimeError("Select a local Ollama model before starting this chat.")
        context_window = self.ollama_context_window()
        cache_key = (resolved_model, context_window)
        if cache_key not in self._json_chat_models:
            options: dict[str, Any] = {
                "model": resolved_model,
                "base_url": self.config.ollama_url,
                "temperature": 0.0,
                "format": "json",
                "validate_model_on_init": True,
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


@dataclass(frozen=True)
class OllamaModelInfo:
    name: str
    family: str = ""
    families: tuple[str, ...] = ()
    supports_images: bool = False
    supports_reasoning: bool = False
    reasoning_mode_strategy: str = "none"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["families"] = list(self.families)
        return payload


@dataclass(frozen=True)
class OllamaCatalogSnapshot:
    models: tuple[OllamaModelInfo, ...] = ()
    ollama_online: bool = False
    has_local_models: bool = False
    source: str = "fallback"


def list_local_ollama_models(config: AppConfig, *, timeout_seconds: float = 3.0) -> list[str]:
    return [item.name for item in list_local_ollama_model_info(config, timeout_seconds=timeout_seconds)]


def list_local_ollama_model_info(config: AppConfig, *, timeout_seconds: float = 3.0) -> list[OllamaModelInfo]:
    return list(inspect_local_ollama_models(config, timeout_seconds=timeout_seconds).models)


def inspect_local_ollama_models(config: AppConfig, *, timeout_seconds: float = 3.0) -> OllamaCatalogSnapshot:
    endpoint = urljoin(f"{config.ollama_url.rstrip('/')}/", "api/tags")
    try:
        with request.urlopen(endpoint, timeout=timeout_seconds) as response:
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return OllamaCatalogSnapshot()

    models = payload.get("models", [])
    entries: list[OllamaModelInfo] = []
    seen: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name in seen or not _is_chat_capable_model(item):
            continue
        seen.add(name)
        details = item.get("details", {}) if isinstance(item.get("details"), dict) else {}
        family = str(details.get("family", "")).strip()
        families = tuple(str(entry).strip() for entry in details.get("families", []) if str(entry).strip())
        entries.append(
            OllamaModelInfo(
                name=name,
                family=family,
                families=families,
                supports_images=_is_vision_capable_model(item),
                supports_reasoning=_is_reasoning_capable_model(item),
                reasoning_mode_strategy=_reasoning_mode_strategy(item),
            )
        )

    entries.sort(key=lambda item: item.name)
    return OllamaCatalogSnapshot(
        models=tuple(entries),
        ollama_online=True,
        has_local_models=bool(entries),
        source="ollama",
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
) -> dict[str, Any]:
    url = urljoin(f"{config.ollama_url.rstrip('/')}/", endpoint)
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request_object = request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    try:
        with request.urlopen(request_object, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
    except (error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return {}


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


def _is_chat_capable_model(payload: dict[str, Any]) -> bool:
    name = str(payload.get("name", "")).lower()
    model = str(payload.get("model", "")).lower()
    details = payload.get("details", {}) if isinstance(payload.get("details"), dict) else {}
    family = str(details.get("family", "")).lower()
    families = [str(item).lower() for item in details.get("families", []) if str(item).strip()]
    combined = " ".join([name, model, family, " ".join(families)])
    blocked_markers = ("embed", "embedding", "bert")
    return not any(marker in combined for marker in blocked_markers)


def _is_vision_capable_model(payload: dict[str, Any]) -> bool:
    name = str(payload.get("name", "")).lower()
    model = str(payload.get("model", "")).lower()
    details = payload.get("details", {}) if isinstance(payload.get("details"), dict) else {}
    family = str(details.get("family", "")).lower()
    families = [str(item).lower() for item in details.get("families", []) if str(item).strip()]
    combined = " ".join([name, model, family, " ".join(families)])
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


def _reasoning_mode_strategy(payload: dict[str, Any]) -> str:
    return _reasoning_mode_strategy_from_descriptor(_normalized_model_descriptor(payload))


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
    reasoning_markers = (
        "qwen3",
        "qwen 3",
        "qwen-3",
        "qwen3.5",
        "qwen3-coder",
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


def _close_chat_client(chat_model: ChatOllama) -> None:
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
