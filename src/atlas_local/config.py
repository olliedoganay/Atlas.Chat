from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from .local_provider import normalize_local_provider_base_url
from .provider_settings import load_provider_settings


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_CHAT_PROVIDER = "ollama"
OPENAI_COMPATIBLE_PROVIDER_DEFAULT_URLS = {
    "lmstudio": "http://127.0.0.1:1234/v1",
    "llamacpp": "http://127.0.0.1:8080/v1",
    "vllm": "http://127.0.0.1:8000/v1",
    "localai": "http://127.0.0.1:8080/v1",
    "openai-compatible": "http://127.0.0.1:8000/v1",
}
CHAT_PROVIDER_LABELS = {
    "ollama": "Ollama",
    "lmstudio": "LM Studio",
    "llamacpp": "llama.cpp server",
    "vllm": "vLLM",
    "localai": "LocalAI",
    "openai-compatible": "OpenAI-compatible local runtime",
}
DEFAULT_CHAT_TEMPERATURE: float | None = None
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
DEFAULT_MEM0_COLLECTION = "atlas_local_memory"
DEFAULT_EMBED_DIM = 768
DEFAULT_MEMORY_TOP_K = 5
DEFAULT_COMPACTION_TIMEOUT_SECONDS = 25.0
MAX_COMPACTION_TIMEOUT_SECONDS = 180.0
MAX_EMBED_DIM = 65_536
MAX_MEMORY_TOP_K = 100


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    prompt_dir: Path
    data_dir: Path
    qdrant_path: Path
    langgraph_checkpoint_db: Path
    mem0_history_db: Path
    ollama_url: str
    chat_provider: str
    chat_base_url: str
    chat_api_key: str | None
    chat_temperature: float | None
    embed_model: str
    mem0_collection: str
    embed_dim: int
    memory_top_k: int
    compaction_timeout_seconds: float
    allow_legacy_plaintext_migration: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_value(env: Mapping[str, str], key: str, default: Path, *, base: Path | None = None) -> Path:
    raw = env.get(key)
    if not raw or not str(raw).strip():
        return default
    path = Path(str(raw).strip())
    if path.is_absolute() or base is None:
        return path
    return base / path


def _storage_path_value(
    env: Mapping[str, str],
    key: str,
    *,
    default: Path,
    data_dir: Path,
    project_root: Path,
    legacy_default: str,
) -> Path:
    raw = str(env.get(key, "") or "").strip()
    if not raw:
        return default
    if raw.replace("\\", "/") == legacy_default and data_dir != project_root / ".data":
        # `.env.example` historically pinned storage back to the repository.
        # Treat those exact legacy defaults as defaults when ATLAS_DATA_DIR is
        # intentionally redirected, while preserving existing source installs.
        return default
    path = Path(raw)
    if path.is_absolute():
        return path
    if raw.replace("\\", "/") == legacy_default:
        return project_root / path
    return data_dir / path


def _value(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key, default)
    return value.strip() if isinstance(value, str) else default


def normalize_chat_provider(value: str | None) -> str:
    normalized = str(value or DEFAULT_CHAT_PROVIDER).strip().lower()
    aliases = {
        "lm-studio": "lmstudio",
        "lm_studio": "lmstudio",
        "llama.cpp": "llamacpp",
        "llama-cpp": "llamacpp",
        "llama_cpp": "llamacpp",
        "llama": "llamacpp",
        "openai": "openai-compatible",
        "openai_compatible": "openai-compatible",
        "openai-compatible": "openai-compatible",
        "local-ai": "localai",
        "local_ai": "localai",
    }
    return aliases.get(normalized, normalized) if normalized in CHAT_PROVIDER_LABELS or normalized in aliases else DEFAULT_CHAT_PROVIDER


def chat_provider_label(provider: str | None) -> str:
    return CHAT_PROVIDER_LABELS.get(normalize_chat_provider(provider), CHAT_PROVIDER_LABELS[DEFAULT_CHAT_PROVIDER])


def is_ollama_chat_provider(provider: str | None) -> bool:
    return normalize_chat_provider(provider) == "ollama"


def _default_chat_base_url(provider: str, ollama_url: str) -> str:
    if is_ollama_chat_provider(provider):
        return ollama_url
    return OPENAI_COMPATIBLE_PROVIDER_DEFAULT_URLS.get(provider, OPENAI_COMPATIBLE_PROVIDER_DEFAULT_URLS["openai-compatible"])


def _optional_secret_value(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key)
    if value is None:
        return None
    text = value.strip() if isinstance(value, str) else str(value).strip()
    return text or None


def _optional_float_value(env: Mapping[str, str], key: str, default: float | None) -> float | None:
    raw = env.get(key)
    if raw is None:
        return default
    text = raw.strip() if isinstance(raw, str) else str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _bounded_float_value(
    env: Mapping[str, str],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = _optional_float_value(env, key, default)
    if value is None:
        return default
    return min(maximum, max(minimum, float(value)))


def _bounded_int_value(
    env: Mapping[str, str],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, value))


def _bool_value(
    env: Mapping[str, str],
    key: str,
    default: bool = False,
) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _restrict_private_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        path.chmod(0o700)
    except OSError:
        pass


def load_config(
    *,
    env: Mapping[str, str] | None = None,
    project_root: Path | None = None,
) -> AppConfig:
    fallback_root = project_root or _repo_root()
    root = fallback_root
    if env is None:
        load_dotenv(root / ".env")
        source = dict(os.environ)
    else:
        source = dict(env)

    root = _path_value(source, "ATLAS_PROJECT_ROOT", fallback_root, base=fallback_root)
    prompt_dir = _path_value(source, "ATLAS_PROMPT_DIR", root / "prompts", base=root)
    data_dir = _path_value(source, "ATLAS_DATA_DIR", root / ".data", base=root)
    saved_provider_settings = load_provider_settings(data_dir)
    qdrant_path = _storage_path_value(
        source,
        "QDRANT_PATH",
        default=data_dir / "qdrant",
        data_dir=data_dir,
        project_root=root,
        legacy_default=".data/qdrant",
    )
    checkpoint_db = _storage_path_value(
        source,
        "LANGGRAPH_CHECKPOINT_DB",
        default=data_dir / "langgraph" / "checkpoints.sqlite",
        data_dir=data_dir,
        project_root=root,
        legacy_default=".data/langgraph/checkpoints.sqlite",
    )
    mem0_history_db = _storage_path_value(
        source,
        "MEM0_HISTORY_DB",
        default=data_dir / "mem0_history.sqlite",
        data_dir=data_dir,
        project_root=root,
        legacy_default=".data/mem0_history.sqlite",
    )

    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    qdrant_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    mem0_history_db.parent.mkdir(parents=True, exist_ok=True)
    _restrict_private_directory(data_dir)
    _restrict_private_directory(qdrant_path)
    for storage_parent in (checkpoint_db.parent, mem0_history_db.parent):
        try:
            if storage_parent.resolve().is_relative_to(data_dir.resolve()):
                _restrict_private_directory(storage_parent)
        except (OSError, ValueError):
            pass

    ollama_url = normalize_local_provider_base_url(
        _value(source, "OLLAMA_URL", DEFAULT_OLLAMA_URL)
    )
    configured_provider = saved_provider_settings.get("provider") or _value(
        source,
        "ATLAS_CHAT_PROVIDER",
        DEFAULT_CHAT_PROVIDER,
    )
    chat_provider = normalize_chat_provider(configured_provider)
    if is_ollama_chat_provider(chat_provider) and saved_provider_settings.get("base_url"):
        ollama_url = normalize_local_provider_base_url(
            saved_provider_settings["base_url"]
        )
    default_chat_base_url = _default_chat_base_url(chat_provider, ollama_url)
    chat_base_url = normalize_local_provider_base_url(
        saved_provider_settings.get("base_url")
        or _value(source, "ATLAS_CHAT_BASE_URL", default_chat_base_url)
        or default_chat_base_url
    )
    saved_api_key = saved_provider_settings.get("api_key")

    return AppConfig(
        project_root=root,
        prompt_dir=prompt_dir,
        data_dir=data_dir,
        qdrant_path=qdrant_path,
        langgraph_checkpoint_db=checkpoint_db,
        mem0_history_db=mem0_history_db,
        ollama_url=ollama_url,
        chat_provider=chat_provider,
        chat_base_url=chat_base_url,
        chat_api_key=saved_api_key or _optional_secret_value(source, "ATLAS_CHAT_API_KEY"),
        chat_temperature=(
            None
            if _optional_float_value(source, "CHAT_TEMPERATURE", DEFAULT_CHAT_TEMPERATURE) is None
            else _bounded_float_value(
                source,
                "CHAT_TEMPERATURE",
                0.0,
                minimum=0.0,
                maximum=2.0,
            )
        ),
        embed_model=_value(source, "EMBED_MODEL", DEFAULT_EMBED_MODEL),
        mem0_collection=_value(source, "MEM0_COLLECTION", DEFAULT_MEM0_COLLECTION),
        embed_dim=_bounded_int_value(
            source,
            "EMBED_DIM",
            DEFAULT_EMBED_DIM,
            minimum=1,
            maximum=MAX_EMBED_DIM,
        ),
        memory_top_k=_bounded_int_value(
            source,
            "MEMORY_TOP_K",
            DEFAULT_MEMORY_TOP_K,
            minimum=1,
            maximum=MAX_MEMORY_TOP_K,
        ),
        compaction_timeout_seconds=_bounded_float_value(
            source,
            "ATLAS_COMPACTION_TIMEOUT_SECONDS",
            DEFAULT_COMPACTION_TIMEOUT_SECONDS,
            minimum=1.0,
            maximum=MAX_COMPACTION_TIMEOUT_SECONDS,
        ),
        allow_legacy_plaintext_migration=_bool_value(
            source,
            "ATLAS_ALLOW_LEGACY_PLAINTEXT_MIGRATION",
        ),
    )
