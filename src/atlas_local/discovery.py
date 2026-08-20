from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error
from urllib.parse import urljoin

from .config import AppConfig, is_ollama_chat_provider
from .llm import OllamaCatalogSnapshot
from .local_provider import (
    normalize_local_provider_base_url,
    provider_urlopen,
)


@dataclass(frozen=True)
class RecommendedModel:
    name: str
    title: str
    use_case: str
    atlas_role: str
    min_ram_gb: float
    good_ram_gb: float
    min_vram_gb: float | None = None
    good_vram_gb: float | None = None
    supports_images: bool = False
    supports_reasoning: bool = False
    model_size_gb: float | None = None


DISCOVERY_MANIFEST_VERSION = 1
DEFAULT_DISCOVERY_MANIFEST = Path(__file__).with_name("discovery_models.json")
MAX_OLLAMA_TAGS_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_DISCOVERY_MANIFEST_BYTES = 1024 * 1024
MAX_DISCOVERY_MODELS = 256
MAX_HARDWARE_COMMAND_OUTPUT_CHARS = 1024 * 1024


FALLBACK_DISCOVERY_MODELS: tuple[RecommendedModel, ...] = (
    RecommendedModel(
        name="llama3.1:8b",
        title="Popular general chat",
        use_case="chat",
        atlas_role="chat",
        min_ram_gb=12.0,
        good_ram_gb=16.0,
        min_vram_gb=8.0,
        good_vram_gb=10.0,
    ),
    RecommendedModel(
        name="qwen3:8b",
        title="Current Qwen all-rounder",
        use_case="reasoning",
        atlas_role="chat",
        min_ram_gb=12.0,
        good_ram_gb=18.0,
        min_vram_gb=8.0,
        good_vram_gb=12.0,
    ),
    RecommendedModel(
        name="qwen2.5-coder:7b",
        title="Popular coding model",
        use_case="coding",
        atlas_role="chat",
        min_ram_gb=12.0,
        good_ram_gb=16.0,
        min_vram_gb=8.0,
        good_vram_gb=10.0,
    ),
    RecommendedModel(
        name="gemma3:4b",
        title="Lightweight vision starter",
        use_case="vision",
        atlas_role="chat",
        min_ram_gb=8.0,
        good_ram_gb=12.0,
        min_vram_gb=6.0,
        good_vram_gb=8.0,
        supports_images=True,
    ),
    RecommendedModel(
        name="deepseek-r1:8b",
        title="Reasoning-focused model",
        use_case="reasoning",
        atlas_role="chat",
        min_ram_gb=16.0,
        good_ram_gb=24.0,
        min_vram_gb=10.0,
        good_vram_gb=14.0,
    ),
    RecommendedModel(
        name="gpt-oss:20b",
        title="High-end reasoning",
        use_case="reasoning",
        atlas_role="chat",
        min_ram_gb=24.0,
        good_ram_gb=32.0,
        min_vram_gb=16.0,
        good_vram_gb=20.0,
    ),
    RecommendedModel(
        name="nomic-embed-text",
        title="Memory retrieval embeddings",
        use_case="embedding",
        atlas_role="embedding",
        min_ram_gb=4.0,
        good_ram_gb=8.0,
        min_vram_gb=2.0,
        good_vram_gb=4.0,
    ),
)


def build_discovery_report(config: AppConfig, catalog: OllamaCatalogSnapshot) -> dict[str, Any]:
    system = detect_local_hardware()
    provider_is_ollama = is_ollama_chat_provider(catalog.provider)
    provider_online = catalog.provider_online or catalog.ollama_online
    installed_model_names = (
        list_installed_ollama_model_names(config)
        if provider_is_ollama
        else [model.name for model in catalog.models]
    )
    installed_lookup = {_normalize_model_name(name): name for name in installed_model_names}
    chat_models = list(catalog.models)
    chat_lookup = {_normalize_model_name(item.name): item for item in chat_models}

    chat_available = provider_online and catalog.has_local_models
    configured_embed_installed = _is_model_installed(config.embed_model, installed_lookup)

    atlas_status = "ready"
    atlas_summary = "Atlas can start chats and memory retrieval is fully configured."
    atlas_notes: list[str] = []
    if not provider_online:
        atlas_status = "runtime-unavailable"
        atlas_summary = f"Atlas cannot validate local models until {catalog.provider_label} is running."
    elif not catalog.has_local_models:
        atlas_status = "chat-blocked"
        atlas_summary = "Atlas cannot start chats until at least one local chat model is installed."
    elif not configured_embed_installed:
        atlas_status = "memory-degraded"
        atlas_summary = "Atlas can start chats, but memory retrieval is degraded until the embed model is installed."

    if chat_available:
        atlas_notes.append("Choose any installed chat model in Workspace before starting a new thread.")
    if chat_available and configured_embed_installed:
        atlas_notes.append(f"Persistent memory is using '{config.embed_model}'.")

    installed_models: list[dict[str, Any]] = []
    for normalized_name, raw_name in sorted(installed_lookup.items(), key=lambda item: item[1].lower()):
        chat_info = chat_lookup.get(normalized_name)
        installed_models.append(
            {
                "name": raw_name,
                "atlas_role": _atlas_role_for_installed_model(
                    raw_name,
                    configured_embed_model=config.embed_model,
                    chat_info=chat_info,
                ),
                "configured_embed_model": _matches_model_name(raw_name, config.embed_model),
                "supports_images": bool(chat_info.supports_images) if chat_info else False,
                "supports_reasoning": bool(chat_info.supports_reasoning) if chat_info else False,
            }
        )

    recommendations = _build_recommendations(
        configured_embed_model=config.embed_model,
        installed_lookup=installed_lookup,
        chat_lookup=chat_lookup,
        discovery_models=load_discovery_models(config),
        system=system,
    )

    return {
        "system": system,
        "atlas": {
            "status": atlas_status,
            "summary": atlas_summary,
            "notes": atlas_notes,
            "ollama_url": config.ollama_url,
            "ollama_online": catalog.ollama_online,
            "provider": catalog.provider,
            "provider_label": catalog.provider_label,
            "provider_base_url": catalog.provider_base_url,
            "provider_online": provider_online,
            "has_local_chat_models": catalog.has_local_models,
            "configured_embed_model": config.embed_model,
            "configured_embed_model_installed": configured_embed_installed,
        },
        "installed_models": installed_models,
        "recommended_models": recommendations,
        "recommendation_manifest": {
            "version": DISCOVERY_MANIFEST_VERSION,
            "source": str(_resolve_discovery_manifest_path(config) or "built-in"),
        },
    }


def detect_local_hardware() -> dict[str, Any]:
    system_name = platform.system().strip() or "Unknown"
    release = platform.release().strip()
    system_label = f"{system_name} {release}".strip()
    snapshot = {
        "os": system_label,
        "platform": sys.platform,
        "cpu": {
            "model": None,
            "logical_cores": os.cpu_count(),
        },
        "memory": {
            "total_gb": None,
        },
        "gpus": [],
        "detection": {
            "confidence": "minimal",
            "notes": [],
        },
    }

    if sys.platform == "win32":
        _populate_windows_snapshot(snapshot)
    elif sys.platform == "darwin":
        _populate_macos_snapshot(snapshot)
    else:
        _populate_linux_snapshot(snapshot)

    confidence_score = 0
    if snapshot["cpu"]["model"]:
        confidence_score += 1
    if snapshot["memory"]["total_gb"] is not None:
        confidence_score += 1
    if snapshot["gpus"]:
        confidence_score += 1
    snapshot["detection"]["confidence"] = {
        0: "minimal",
        1: "partial",
        2: "good",
        3: "full",
    }[confidence_score]
    if not snapshot["gpus"]:
        snapshot["detection"]["notes"].append("GPU details were not detected. Atlas will estimate fit from system RAM only.")
    return snapshot


def list_installed_ollama_model_names(config: AppConfig, *, timeout_seconds: float = 3.0) -> list[str]:
    base_url = normalize_local_provider_base_url(config.ollama_url)
    endpoint = urljoin(f"{base_url}/", "api/tags")
    try:
        with provider_urlopen(endpoint, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_OLLAMA_TAGS_RESPONSE_BYTES:
                return []
            raw_payload = response.read(MAX_OLLAMA_TAGS_RESPONSE_BYTES + 1)
            if len(raw_payload) > MAX_OLLAMA_TAGS_RESPONSE_BYTES:
                return []
            payload = json.loads(raw_payload.decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []

    models = payload.get("models", [])
    if not isinstance(models, list):
        return []

    seen: set[str] = set()
    names: list[str] = []
    for item in models[:MAX_DISCOVERY_MODELS]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        normalized = _normalize_model_name(name)
        if not name or normalized in seen:
            continue
        seen.add(normalized)
        names.append(name)
    return names


def _build_recommendations(
    *,
    configured_embed_model: str,
    installed_lookup: dict[str, str],
    chat_lookup: dict[str, Any],
    discovery_models: tuple[RecommendedModel, ...],
    system: dict[str, Any],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    candidates = list(discovery_models)
    seen_candidates = {_normalize_model_name(item.name) for item in candidates}
    for model_info in chat_lookup.values():
        normalized_name = _normalize_model_name(str(model_info.name))
        if normalized_name in seen_candidates:
            continue
        candidates.append(_recommended_model_from_ollama_metadata(model_info))
        seen_candidates.add(normalized_name)
    for catalog_rank, candidate in enumerate(candidates):
        installed = _is_model_installed(candidate.name, installed_lookup)
        fit, runtime, reason = _estimate_model_fit(candidate, system)
        chat_info = chat_lookup.get(_normalize_model_name(candidate.name))
        catalog_model_size_gb = _model_size_gb_from_ollama_info(chat_info)
        recommendations.append(
            {
                "name": candidate.name,
                "title": candidate.title,
                "use_case": candidate.use_case,
                "atlas_role": candidate.atlas_role,
                "installed": installed,
                "supports_images": (candidate.supports_images or bool(chat_info.supports_images))
                if chat_info
                else candidate.supports_images,
                "supports_reasoning": (
                    candidate.supports_reasoning or bool(chat_info.supports_reasoning)
                )
                if chat_info
                else candidate.supports_reasoning,
                "model_size_gb": candidate.model_size_gb or catalog_model_size_gb,
                "fit": fit,
                "runtime": runtime,
                "reason": reason,
                "pull_command": f"ollama pull {candidate.name}",
                "source": "ollama" if _normalize_model_name(candidate.name) in chat_lookup and catalog_rank >= len(discovery_models) else "manifest",
                "_catalog_rank": catalog_rank,
            }
        )
    recommendations.sort(key=_recommendation_sort_key)
    for item in recommendations:
        item.pop("_catalog_rank", None)
    return recommendations


def load_discovery_models(config: AppConfig) -> tuple[RecommendedModel, ...]:
    manifest_path = _resolve_discovery_manifest_path(config)
    if manifest_path is None:
        return FALLBACK_DISCOVERY_MODELS
    try:
        if manifest_path.stat().st_size > MAX_DISCOVERY_MANIFEST_BYTES:
            return FALLBACK_DISCOVERY_MODELS
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return FALLBACK_DISCOVERY_MODELS
    if not isinstance(payload, dict):
        return FALLBACK_DISCOVERY_MODELS
    try:
        version = int(payload.get("version", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return FALLBACK_DISCOVERY_MODELS
    if version != DISCOVERY_MANIFEST_VERSION:
        return FALLBACK_DISCOVERY_MODELS
    raw_models = payload.get("models", [])
    if not isinstance(raw_models, list):
        return FALLBACK_DISCOVERY_MODELS
    models: list[RecommendedModel] = []
    for item in raw_models[:MAX_DISCOVERY_MODELS]:
        parsed = _parse_manifest_model(item)
        if parsed:
            models.append(parsed)
    return tuple(models) or FALLBACK_DISCOVERY_MODELS


def _resolve_discovery_manifest_path(config: AppConfig) -> Path | None:
    override = os.environ.get("ATLAS_DISCOVERY_MANIFEST", "").strip()
    if override:
        path = Path(override)
        return path if path.exists() else None
    project_manifest = config.project_root / "discovery_models.json"
    if project_manifest.exists():
        return project_manifest
    return DEFAULT_DISCOVERY_MANIFEST if DEFAULT_DISCOVERY_MANIFEST.exists() else None


def _parse_manifest_model(item: Any) -> RecommendedModel | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name", "") or "").strip()
    title = str(item.get("title", "") or "").strip()
    if (
        not name
        or len(name) > 200
        or not title
        or len(title) > 200
        or any(ord(character) < 0x20 for character in name + title)
    ):
        return None
    min_ram = _safe_float(item.get("min_ram_gb"))
    good_ram = _safe_float(item.get("good_ram_gb"))
    if (
        min_ram is None
        or good_ram is None
        or min_ram <= 0
        or good_ram < min_ram
    ):
        return None
    min_vram = _safe_float(item.get("min_vram_gb"))
    good_vram = _safe_float(item.get("good_vram_gb"))
    if min_vram is not None and min_vram <= 0:
        return None
    if good_vram is not None and (
        good_vram <= 0 or (min_vram is not None and good_vram < min_vram)
    ):
        return None
    return RecommendedModel(
        name=name,
        title=title,
        use_case=str(item.get("use_case", "chat") or "chat").strip() or "chat",
        atlas_role=str(item.get("atlas_role", "chat") or "chat").strip() or "chat",
        min_ram_gb=min_ram,
        good_ram_gb=good_ram,
        min_vram_gb=min_vram,
        good_vram_gb=good_vram,
        supports_images=item.get("supports_images") is True,
    )


def _recommended_model_from_ollama_metadata(model_info: Any) -> RecommendedModel:
    name = str(model_info.name)
    normalized = name.casefold()
    supports_images = bool(getattr(model_info, "supports_images", False))
    supports_reasoning = bool(getattr(model_info, "supports_reasoning", False))
    use_case = "vision" if supports_images else "reasoning" if supports_reasoning or "r1" in normalized else "chat"
    if supports_images and supports_reasoning:
        title = "Installed vision and reasoning model"
    elif use_case == "vision":
        title = "Installed vision model"
    elif use_case == "reasoning":
        title = "Installed reasoning model"
    else:
        title = "Installed chat model"
    model_size_gb = _model_size_gb_from_ollama_info(model_info)
    default_min_vram = 6.0 if supports_images else 4.0
    if model_size_gb is None:
        min_ram_gb = 8.0
        good_ram_gb = 16.0
        min_vram_gb = default_min_vram
        good_vram_gb = 8.0
    else:
        # Ollama's on-disk weight size is a materially better lower bound than
        # the model name. Leave room for the OS, runtime, and KV cache as well.
        min_ram_gb = float(max(8, math.ceil((model_size_gb * 1.15) + 4)))
        good_ram_gb = float(max(16, math.ceil((model_size_gb * 1.35) + 8)))
        min_vram_gb = float(max(default_min_vram, math.ceil(model_size_gb * 1.05)))
        good_vram_gb = float(max(8, math.ceil(model_size_gb * 1.2)))
    return RecommendedModel(
        name=name,
        title=title,
        use_case=use_case,
        atlas_role="chat",
        min_ram_gb=min_ram_gb,
        good_ram_gb=good_ram_gb,
        min_vram_gb=min_vram_gb,
        good_vram_gb=good_vram_gb,
        supports_images=supports_images,
        supports_reasoning=supports_reasoning,
        model_size_gb=model_size_gb,
    )


def _model_size_gb_from_ollama_info(model_info: Any | None) -> float | None:
    size_bytes = getattr(model_info, "size_bytes", None)
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        return None
    return round(size_bytes / (1024**3), 1)


def _recommendation_sort_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
    fit_priority = {
        "good": 0,
        "tight": 1,
        "cpu-only": 2,
        "unavailable": 3,
        "too-large": 4,
    }.get(str(item.get("fit")), 5)
    installed_priority = 0 if item.get("installed") else 1
    catalog_rank = int(item.get("_catalog_rank", 999))
    return (fit_priority, installed_priority, catalog_rank, str(item.get("name", "")))


def _estimate_model_fit(candidate: RecommendedModel, system: dict[str, Any]) -> tuple[str, str, str]:
    total_ram = _safe_float(system.get("memory", {}).get("total_gb"))
    dedicated_gpu_memories = [
        value
        for value in (
            _safe_float(item.get("memory_gb"))
            for item in system.get("gpus", [])
            if str(item.get("kind", "")).strip().lower() != "integrated"
        )
        if value is not None
    ]
    best_gpu_vram = max(dedicated_gpu_memories, default=None)

    if candidate.good_vram_gb is not None and best_gpu_vram is not None and best_gpu_vram >= candidate.good_vram_gb:
        return ("good", "GPU", f"Detected GPU memory should handle {candidate.name} with headroom.")
    if candidate.min_vram_gb is not None and best_gpu_vram is not None and best_gpu_vram >= candidate.min_vram_gb:
        return ("tight", "GPU", f"Detected GPU memory should run {candidate.name}, but headroom will be limited.")
    if total_ram is not None and total_ram >= candidate.good_ram_gb:
        runtime = "Hybrid" if best_gpu_vram is not None else "CPU"
        detail = "mixed CPU/GPU offload" if runtime == "Hybrid" else "CPU execution"
        return ("tight", runtime, f"System RAM should support {candidate.name} with {detail}; first load can be slower.")
    if total_ram is not None and total_ram >= candidate.min_ram_gb:
        if best_gpu_vram is not None:
            return (
                "tight",
                "Hybrid",
                f"{candidate.name} should fit with mixed CPU/GPU offload, but GPU memory is below the full-model estimate.",
            )
        return ("cpu-only", "CPU", f"{candidate.name} should fit in system RAM, but expect slower CPU-first performance.")
    if (
        total_ram is not None
        and best_gpu_vram is not None
        and total_ram + best_gpu_vram >= candidate.min_ram_gb
    ):
        return (
            "tight",
            "Hybrid",
            f"Combined system and GPU memory may run {candidate.name}, but available headroom will be limited.",
        )
    if total_ram is None and best_gpu_vram is None:
        return ("unavailable", "Unknown", f"Atlas could not detect enough hardware detail to estimate {candidate.name}.")
    return ("too-large", "Unknown", f"Detected RAM or VRAM looks too small for a comfortable {candidate.name} setup.")


def _atlas_role_for_installed_model(
    model_name: str,
    *,
    configured_embed_model: str,
    chat_info: Any | None,
) -> str:
    if _matches_model_name(model_name, configured_embed_model):
        return "embedding"
    if chat_info and chat_info.supports_images:
        return "vision"
    if chat_info:
        return "chat"
    if "embed" in model_name.lower():
        return "embedding"
    return "other"


def _matches_model_name(left: str, right: str) -> bool:
    if not left.strip() or not right.strip():
        return False
    return _normalize_model_name(left) == _normalize_model_name(right)


def _is_model_installed(target: str, installed_lookup: dict[str, str]) -> bool:
    return _normalize_model_name(target) in installed_lookup


def _normalize_model_name(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return ""
    if ":" not in normalized:
        return f"{normalized}:latest"
    return normalized


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _bytes_to_gb(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        numeric = float(value)
        if numeric <= 0:
            return None
        return round(numeric / (1024 ** 3), 1)
    except (TypeError, ValueError):
        return None


def _populate_windows_snapshot(snapshot: dict[str, Any]) -> None:
    snapshot["os"] = _resolve_windows_system_label()
    payload = _run_json_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name; "
                "$ram = Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory; "
                "$gpus = @(Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM); "
                "@{ cpu = $cpu; ram = $ram; gpus = $gpus } | ConvertTo-Json -Compress"
            ),
        ]
    )
    if not isinstance(payload, dict):
        snapshot["detection"]["notes"].append("Windows hardware detection fell back to generic platform values.")
        return

    snapshot["cpu"]["model"] = str(payload.get("cpu", "")).strip() or None
    snapshot["memory"]["total_gb"] = _bytes_to_gb(payload.get("ram"))
    snapshot["gpus"] = _normalize_windows_gpu_entries(payload.get("gpus"), nvidia_gpus=_detect_nvidia_smi_gpus())


def _populate_macos_snapshot(snapshot: dict[str, Any]) -> None:
    cpu_model = _run_text_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    memory_bytes = _run_text_command(["sysctl", "-n", "hw.memsize"])
    display_payload = _run_json_command(["system_profiler", "SPDisplaysDataType", "-json"])

    snapshot["cpu"]["model"] = cpu_model or None
    snapshot["memory"]["total_gb"] = _bytes_to_gb(memory_bytes)

    gpus: list[dict[str, Any]] = []
    if isinstance(display_payload, dict):
        for item in display_payload.get("SPDisplaysDataType", []):
            if not isinstance(item, dict):
                continue
            gpu_name = str(item.get("sppci_model") or item.get("_name") or "").strip()
            vram = _parse_macos_vram(item.get("spdisplays_vram") or item.get("spdisplays_vram_shared"))
            if gpu_name:
                gpus.append({"name": gpu_name, "memory_gb": vram})
    snapshot["gpus"] = gpus


def _populate_linux_snapshot(snapshot: dict[str, Any]) -> None:
    snapshot["cpu"]["model"] = _read_linux_cpu_model()
    snapshot["memory"]["total_gb"] = _read_linux_total_ram_gb()
    snapshot["gpus"] = _detect_linux_gpus()


def _detect_linux_gpus() -> list[dict[str, Any]]:
    nvidia_gpus = _detect_nvidia_smi_gpus()
    if nvidia_gpus:
        return nvidia_gpus

    if shutil.which("lspci"):
        output = _run_text_command(["lspci"])
        gpus = []
        for line in output.splitlines():
            normalized = line.lower()
            if "vga compatible controller" not in normalized and "3d controller" not in normalized:
                continue
            _, _, name = line.partition(":")
            rendered = name.strip() or line.strip()
            if rendered:
                gpus.append({"name": rendered, "memory_gb": None})
        if gpus:
            return gpus
    return []


def _read_linux_cpu_model() -> str | None:
    cpuinfo = _safe_read_text("/proc/cpuinfo")
    if not cpuinfo:
        return None
    for line in cpuinfo.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        if key.strip().lower() == "model name":
            rendered = value.strip()
            return rendered or None
    return None


def _read_linux_total_ram_gb() -> float | None:
    meminfo = _safe_read_text("/proc/meminfo")
    if not meminfo:
        return None
    for line in meminfo.splitlines():
        if not line.lower().startswith("memtotal:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return round(float(parts[1]) / (1024 ** 2), 1)
        except ValueError:
            return None
    return None


def _safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(MAX_HARDWARE_COMMAND_OUTPUT_CHARS)
    except OSError:
        return ""


def _normalize_gpu_entries(value: Any) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else [value]
    gpus: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name", "")).strip()
        if not name:
            continue
        gpus.append(
            {
                "name": name,
                "memory_gb": _bytes_to_gb(item.get("AdapterRAM")),
            }
        )
    return gpus


def _normalize_windows_gpu_entries(value: Any, *, nvidia_gpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else [value]
    gpus: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name", "")).strip()
        if not name:
            continue

        normalized_name = name.casefold()
        if normalized_name in seen:
            continue
        seen.add(normalized_name)

        kind = _classify_gpu_kind(name)
        matched_nvidia = _match_gpu_by_name(name, nvidia_gpus)
        adapter_ram_gb = _bytes_to_gb(item.get("AdapterRAM"))

        memory_gb = matched_nvidia.get("memory_gb") if matched_nvidia else adapter_ram_gb
        memory_source = matched_nvidia.get("memory_source", "unknown") if matched_nvidia else ("adapterram" if adapter_ram_gb is not None else "unknown")
        if kind == "integrated":
            memory_gb = None
            memory_source = "shared"

        gpus.append(
            {
                "name": name,
                "memory_gb": memory_gb,
                "kind": kind,
                "memory_source": memory_source,
            }
        )

    gpus.sort(
        key=lambda item: (
            0 if item.get("kind") == "dedicated" else 1 if item.get("kind") == "unknown" else 2,
            -(item.get("memory_gb") or 0),
            str(item.get("name", "")).lower(),
        )
    )
    return gpus


def _detect_nvidia_smi_gpus() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []

    output = _run_text_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []

    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        memory_gb = None
        if len(parts) > 1:
            try:
                memory_gb = round(float(parts[1]) / 1024.0, 1)
            except ValueError:
                memory_gb = None
        gpus.append(
            {
                "name": parts[0],
                "memory_gb": memory_gb,
                "kind": "dedicated",
                "memory_source": "nvidia-smi",
            }
        )
    return gpus


def _classify_gpu_kind(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        return "unknown"
    integrated_markers = (
        "intel(",
        "intel ",
        "uhd graphics",
        "iris",
        "integrated",
    )
    dedicated_markers = (
        "nvidia",
        "geforce",
        "quadro",
        "rtx",
        "gtx",
        "radeon rx",
        "radeon pro",
    )
    if any(marker in normalized for marker in integrated_markers):
        return "integrated"
    if any(marker in normalized for marker in dedicated_markers):
        return "dedicated"
    return "unknown"


def _match_gpu_by_name(target_name: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized_target = target_name.casefold()
    for item in candidates:
        candidate_name = str(item.get("name", "")).casefold()
        if not candidate_name:
            continue
        if candidate_name == normalized_target or candidate_name in normalized_target or normalized_target in candidate_name:
            return item
    return None


def _resolve_windows_system_label() -> str:
    build_number = _windows_build_number(platform.version())
    if build_number is not None and build_number >= 22000:
        return "Windows 11"
    release = platform.release().strip()
    return f"Windows {release}".strip() or "Windows"


def _windows_build_number(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = [part for part in text.split(".") if part]
    if not parts:
        return None
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return None


def _parse_macos_vram(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    multiplier = 1.0
    if "mb" in text:
        multiplier = 1.0 / 1024.0
    number_chars = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if not number_chars:
        return None
    try:
        return round(float(number_chars) * multiplier, 1)
    except ValueError:
        return None


def _run_text_command(command: list[str], *, timeout_seconds: float = 3.0) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout[:MAX_HARDWARE_COMMAND_OUTPUT_CHARS].strip()


def _run_json_command(command: list[str], *, timeout_seconds: float = 3.0) -> Any:
    output = _run_text_command(command, timeout_seconds=timeout_seconds)
    if not output:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None
