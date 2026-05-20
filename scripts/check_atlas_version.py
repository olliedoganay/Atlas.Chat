from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_tag(tag: str) -> str:
    value = tag.strip()
    if value.startswith("refs/tags/"):
        value = value.removeprefix("refs/tags/")
    return value


def _load_versions(repo_root: Path) -> dict[str, str]:
    package_json = json.loads((repo_root / "apps" / "atlas" / "package.json").read_text(encoding="utf-8"))
    package_lock = json.loads((repo_root / "apps" / "atlas" / "package-lock.json").read_text(encoding="utf-8"))
    tauri_conf = json.loads(
        (repo_root / "apps" / "atlas" / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8")
    )
    cargo_toml = tomllib.loads((repo_root / "apps" / "atlas" / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8"))
    cargo_lock = (repo_root / "apps" / "atlas" / "src-tauri" / "Cargo.lock").read_text(encoding="utf-8")
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    ai_note = (repo_root / "AI.md").read_text(encoding="utf-8")
    cargo_lock_version = _cargo_lock_package_version(cargo_lock, package_name="atlas-desktop")
    readme_version = _readme_current_version(readme)
    ai_version = _ai_current_version(ai_note)
    ai_release_tag_version = _ai_current_release_tag_version(ai_note)
    return {
        "package.json": str(package_json["version"]).strip(),
        "package-lock.json": str(package_lock["version"]).strip(),
        "package-lock.json packages[\"\"]": str(package_lock["packages"][""]["version"]).strip(),
        "tauri.conf.json": str(tauri_conf["version"]).strip(),
        "Cargo.toml": str(cargo_toml["package"]["version"]).strip(),
        "Cargo.lock atlas-desktop": cargo_lock_version,
        "pyproject.toml": str(pyproject["project"]["version"]).strip(),
        "README.md": readme_version,
        "AI.md current version": ai_version,
        "AI.md current release tag": ai_release_tag_version,
    }


def _cargo_lock_package_version(content: str, *, package_name: str) -> str:
    match = re.search(
        rf'(?ms)^\[\[package\]\]\s+name = "{re.escape(package_name)}"\s+version = "([^"]+)"',
        content,
    )
    if not match:
        raise SystemExit(f"Could not find {package_name!r} in Cargo.lock.")
    return match.group(1).strip()


def _readme_current_version(content: str) -> str:
    match = re.search(r"(?m)^Current version: `([^`]+)`$", content)
    if not match:
        raise SystemExit("Could not find README current version line.")
    return match.group(1).strip()


def _ai_current_version(content: str) -> str:
    match = re.search(r"(?m)^- Current version: `([^`]+)`$", content)
    if not match:
        raise SystemExit("Could not find AI.md current version line.")
    return match.group(1).strip()


def _ai_current_release_tag_version(content: str) -> str:
    match = re.search(r"(?m)^- Current release tag: `v([^`]+)`$", content)
    if not match:
        raise SystemExit("Could not find AI.md current release tag line.")
    return match.group(1).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Atlas version consistency across manifests.")
    parser.add_argument("--tag", help="Optional git tag to validate, expected format vX.Y.Z")
    args = parser.parse_args()

    versions = _load_versions(_repo_root())
    unique_versions = sorted(set(versions.values()))
    if len(unique_versions) != 1:
        details = ", ".join(f"{path}={value}" for path, value in versions.items())
        raise SystemExit(f"Atlas version mismatch across manifests: {details}")

    version = unique_versions[0]
    if args.tag:
        normalized_tag = _normalize_tag(args.tag)
        expected_tag = f"v{version}"
        if normalized_tag != expected_tag:
            raise SystemExit(f"Git tag {normalized_tag!r} does not match manifest version {expected_tag!r}.")

    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
