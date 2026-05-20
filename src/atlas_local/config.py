from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_CHAT_TEMPERATURE: float | None = None
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
DEFAULT_MEM0_COLLECTION = "atlas_local_memory"
DEFAULT_EMBED_DIM = 768
DEFAULT_MEMORY_TOP_K = 5
DEFAULT_COMPACTION_TIMEOUT_SECONDS = 25.0


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    prompt_dir: Path
    data_dir: Path
    qdrant_path: Path
    langgraph_checkpoint_db: Path
    mem0_history_db: Path
    ollama_url: str
    chat_temperature: float | None
    embed_model: str
    mem0_collection: str
    embed_dim: int
    memory_top_k: int
    compaction_timeout_seconds: float


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


def _value(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key, default)
    return value.strip() if isinstance(value, str) else default


def _optional_float_value(env: Mapping[str, str], key: str, default: float | None) -> float | None:
    raw = env.get(key)
    if raw is None:
        return default
    text = raw.strip() if isinstance(raw, str) else str(raw).strip()
    if not text:
        return None
    return float(text)


def _positive_float_value(env: Mapping[str, str], key: str, default: float) -> float:
    value = _optional_float_value(env, key, default)
    if value is None:
        return default
    return max(1.0, float(value))


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
    qdrant_path = _path_value(source, "QDRANT_PATH", root / ".data" / "qdrant", base=root)
    checkpoint_db = _path_value(
        source,
        "LANGGRAPH_CHECKPOINT_DB",
        root / ".data" / "langgraph" / "checkpoints.sqlite",
        base=root,
    )
    mem0_history_db = _path_value(
        source,
        "MEM0_HISTORY_DB",
        data_dir / "mem0_history.sqlite",
        base=data_dir,
    )

    qdrant_path.mkdir(parents=True, exist_ok=True)
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    mem0_history_db.parent.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        project_root=root,
        prompt_dir=prompt_dir,
        data_dir=data_dir,
        qdrant_path=qdrant_path,
        langgraph_checkpoint_db=checkpoint_db,
        mem0_history_db=mem0_history_db,
        ollama_url=_value(source, "OLLAMA_URL", DEFAULT_OLLAMA_URL),
        chat_temperature=_optional_float_value(source, "CHAT_TEMPERATURE", DEFAULT_CHAT_TEMPERATURE),
        embed_model=_value(source, "EMBED_MODEL", DEFAULT_EMBED_MODEL),
        mem0_collection=_value(source, "MEM0_COLLECTION", DEFAULT_MEM0_COLLECTION),
        embed_dim=int(_value(source, "EMBED_DIM", str(DEFAULT_EMBED_DIM))),
        memory_top_k=int(_value(source, "MEMORY_TOP_K", str(DEFAULT_MEMORY_TOP_K))),
        compaction_timeout_seconds=_positive_float_value(
            source,
            "ATLAS_COMPACTION_TIMEOUT_SECONDS",
            DEFAULT_COMPACTION_TIMEOUT_SECONDS,
        ),
    )
