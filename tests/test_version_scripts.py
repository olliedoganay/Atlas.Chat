import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import bump_atlas_version, check_atlas_version


def _write_version_fixture(repo_root: Path, version: str) -> None:
    (repo_root / "apps" / "atlas" / "src-tauri").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text(
        f'[project]\nname = "atlas-local"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (repo_root / "README.md").write_text(f"# Atlas Chat\n\nCurrent version: `{version}`\n", encoding="utf-8")
    (repo_root / "AI.md").write_text(
        f"# Atlas Repo Note\n\n- Current version: `{version}`\n\n## Releases\n\n- Current release tag: `v{version}`\n",
        encoding="utf-8",
    )
    (repo_root / "apps" / "atlas" / "package.json").write_text(
        f'{{"name":"atlas-desktop","version":"{version}"}}\n',
        encoding="utf-8",
    )
    (repo_root / "apps" / "atlas" / "package-lock.json").write_text(
        (
            '{"name":"atlas-desktop",'
            f'"version":"{version}",'
            '"lockfileVersion":3,'
            '"packages":{"":{'
            '"name":"atlas-desktop",'
            f'"version":"{version}"'
            "}}}\n"
        ),
        encoding="utf-8",
    )
    (repo_root / "apps" / "atlas" / "src-tauri" / "tauri.conf.json").write_text(
        f'{{"version":"{version}"}}\n',
        encoding="utf-8",
    )
    (repo_root / "apps" / "atlas" / "src-tauri" / "Cargo.toml").write_text(
        f'[package]\nname = "atlas-desktop"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (repo_root / "apps" / "atlas" / "src-tauri" / "Cargo.lock").write_text(
        f'[[package]]\nname = "atlas-desktop"\nversion = "{version}"\n',
        encoding="utf-8",
    )


class VersionScriptTests(unittest.TestCase):
    def test_version_check_includes_ai_note(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_version_fixture(repo_root, "1.2.3")

            versions = check_atlas_version._load_versions(repo_root)

        self.assertEqual(versions["AI.md current version"], "1.2.3")
        self.assertEqual(versions["AI.md current release tag"], "1.2.3")

    def test_version_check_fails_when_ai_note_drifts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_version_fixture(repo_root, "1.2.3")
            (repo_root / "AI.md").write_text(
                "# Atlas Repo Note\n\n- Current version: `1.2.3`\n\n## Releases\n\n- Current release tag: `v1.2.2`\n",
                encoding="utf-8",
            )

            with (
                patch("scripts.check_atlas_version._repo_root", return_value=repo_root),
                patch.object(sys, "argv", ["check_atlas_version.py"]),
            ):
                with self.assertRaises(SystemExit) as context:
                    check_atlas_version.main()

        self.assertIn("Atlas version mismatch", str(context.exception))
        self.assertIn("AI.md current release tag=1.2.2", str(context.exception))

    def test_version_bump_updates_ai_note(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            _write_version_fixture(repo_root, "1.2.3")

            bump_atlas_version._replace_manifest_versions(repo_root, "1.2.3", "1.2.4")

            ai_note = (repo_root / "AI.md").read_text(encoding="utf-8")

        self.assertIn("- Current version: `1.2.4`", ai_note)
        self.assertIn("- Current release tag: `v1.2.4`", ai_note)


if __name__ == "__main__":
    unittest.main()
