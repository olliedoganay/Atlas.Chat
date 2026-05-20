from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _current_version(repo_root: Path) -> str:
    payload = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(payload["project"]["version"]).strip()


def _next_patch(version: str) -> str:
    major, minor, patch = (int(part) for part in version.split("."))
    return f"{major}.{minor}.{patch + 1}"


def _replace(path: Path, pattern: str, replacement: str, *, flags: int = 0) -> None:
    content = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, content, count=1, flags=flags)
    if count != 1:
        raise SystemExit(f"Expected one version replacement in {path}.")
    path.write_text(updated, encoding="utf-8")


def _replace_manifest_versions(repo_root: Path, old_version: str, new_version: str) -> None:
    escaped_old = re.escape(old_version)
    version_json = rf'("version":\s*)"{escaped_old}"'
    version_toml = rf'(?m)^version = "{escaped_old}"$'

    _replace(repo_root / "pyproject.toml", version_toml, f'version = "{new_version}"')
    _replace(repo_root / "apps" / "atlas" / "package.json", version_json, rf'\1"{new_version}"')
    _replace(repo_root / "apps" / "atlas" / "src-tauri" / "tauri.conf.json", version_json, rf'\1"{new_version}"')
    _replace(repo_root / "apps" / "atlas" / "src-tauri" / "Cargo.toml", version_toml, f'version = "{new_version}"')
    _replace(repo_root / "README.md", rf"(?m)^Current version: `{escaped_old}`$", f"Current version: `{new_version}`")
    _replace(repo_root / "AI.md", rf"(?m)^- Current version: `{escaped_old}`$", f"- Current version: `{new_version}`")
    _replace(repo_root / "AI.md", rf"(?m)^- Current release tag: `v{escaped_old}`$", f"- Current release tag: `v{new_version}`")

    package_lock = repo_root / "apps" / "atlas" / "package-lock.json"
    _replace(package_lock, version_json, rf'\1"{new_version}"')
    _replace(
        package_lock,
        rf'(?ms)("packages":\s*\{{\s*"":\s*\{{.*?"version":\s*)"{escaped_old}"',
        rf'\1"{new_version}"',
    )

    _replace(
        repo_root / "apps" / "atlas" / "src-tauri" / "Cargo.lock",
        rf'(?ms)(\[\[package\]\]\s+name = "atlas-desktop"\s+version = )"{escaped_old}"',
        rf'\1"{new_version}"',
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bump Atlas version across all release manifests.")
    parser.add_argument("version", nargs="?", help="New X.Y.Z version. Defaults to the next patch version.")
    args = parser.parse_args()

    repo_root = _repo_root()
    old_version = _current_version(repo_root)
    new_version = args.version.strip() if args.version else _next_patch(old_version)
    if not SEMVER_RE.fullmatch(new_version):
        raise SystemExit("Atlas versions must use X.Y.Z format.")
    if new_version == old_version:
        raise SystemExit(f"Atlas version is already {new_version}.")

    _replace_manifest_versions(repo_root, old_version, new_version)
    print(new_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
