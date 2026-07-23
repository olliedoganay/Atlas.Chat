from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib


def atlas_version() -> str:
    # Editable installs retain the package metadata that existed when the
    # environment was created. Prefer the checked-out manifest during source
    # development so the running backend cannot report a stale release.
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = repo_root / "pyproject.toml"
    if pyproject.exists():
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return str(payload["project"]["version"]).strip()
    try:
        return version("atlas-local")
    except PackageNotFoundError:
        pass
    return "0.0.0"
