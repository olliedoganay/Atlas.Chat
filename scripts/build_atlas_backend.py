from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    resources_dir = repo_root / "apps" / "atlas" / "src-tauri" / "resources"
    backend_resource_dir = resources_dir / "backend"
    prompt_resource_dir = resources_dir / "prompts"
    build_root = repo_root / "output" / "pyinstaller"
    build_root.mkdir(parents=True, exist_ok=True)
    session_root = Path(tempfile.mkdtemp(prefix="session-", dir=build_root))
    dist_root = session_root / "dist"
    work_root = session_root / "build"
    spec_root = session_root / "spec"

    for path in (backend_resource_dir, prompt_resource_dir):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    _cleanup_old_sessions(build_root, keep=session_root)

    for path in (resources_dir, dist_root, work_root, spec_root):
        path.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "atlas-backend",
        "--distpath",
        str(dist_root),
        "--workpath",
        str(work_root),
        "--specpath",
        str(spec_root),
        "--paths",
        str(repo_root / "src"),
        "--collect-submodules",
        "atlas_local",
        "--collect-submodules",
        "uvicorn",
        "--hidden-import",
        "mem0.configs.vector_stores.qdrant",
        "--hidden-import",
        "mem0.embeddings.ollama",
        "--hidden-import",
        "mem0.llms.ollama",
        "--hidden-import",
        "mem0.vector_stores.qdrant",
        "--exclude-module",
        "pytest",
        "--exclude-module",
        "_pytest",
        "--copy-metadata",
        "mem0ai",
        "--copy-metadata",
        "langgraph",
        "--copy-metadata",
        "langchain-ollama",
        "--copy-metadata",
        "fastapi",
        "--copy-metadata",
        "uvicorn",
        str(repo_root / "src" / "atlas_local" / "backend_entry.py"),
    ]

    try:
        subprocess.run(command, check=True, cwd=repo_root)

        backend_binary_name = "atlas-backend.exe" if sys.platform == "win32" else "atlas-backend"
        built_backend = dist_root / backend_binary_name
        if not built_backend.exists():
            raise RuntimeError(f"Backend build output was not created: {built_backend}")

        backend_resource_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_backend, backend_resource_dir / backend_binary_name)
        shutil.copytree(repo_root / "prompts", prompt_resource_dir, dirs_exist_ok=True)
        _wait_for_tree_ready(backend_resource_dir)
        # Windows can briefly hold newly copied native extension files while
        # Defender or the filesystem finishes post-copy work. Tauri's bundle
        # step touches these resources immediately after this script exits.
        time.sleep(5.0)
    finally:
        shutil.rmtree(session_root, ignore_errors=True)
    return 0


def _cleanup_old_sessions(build_root: Path, *, keep: Path) -> None:
    for path in build_root.glob("session-*"):
        if path == keep:
            continue
        shutil.rmtree(path, ignore_errors=True)


def _wait_for_tree_ready(root: Path) -> None:
    pending = [path for path in root.rglob("*") if path.is_file()]
    for attempt in range(20):
        locked = False
        for path in pending:
            try:
                with path.open("rb"):
                    pass
            except PermissionError:
                locked = True
                break
        if not locked:
            return
        time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"Backend resources stayed locked after copy: {root}")


if __name__ == "__main__":
    raise SystemExit(main())
