import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atlas_local.config import load_config
from atlas_local.provider_settings import load_provider_settings, save_provider_settings


class ConfigTests(unittest.TestCase):
    def test_defaults_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(project_root=root, env={})
            self.assertIsNone(config.chat_temperature)
            self.assertEqual(config.chat_provider, "ollama")
            self.assertEqual(config.chat_base_url, "http://127.0.0.1:11434")
            self.assertEqual(config.embed_model, "nomic-embed-text:latest")
            self.assertEqual(config.embed_dim, 768)
            self.assertTrue(config.qdrant_path.exists())
            self.assertTrue(config.langgraph_checkpoint_db.parent.exists())
            self.assertTrue(config.mem0_history_db.parent.exists())

    def test_local_openai_compatible_provider_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(project_root=root, env={"ATLAS_CHAT_PROVIDER": "lm-studio"})

            self.assertEqual(config.chat_provider, "lmstudio")
            self.assertEqual(config.chat_base_url, "http://127.0.0.1:1234/v1")

    def test_local_provider_base_url_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(
                project_root=root,
                env={
                    "ATLAS_CHAT_PROVIDER": "vllm",
                    "ATLAS_CHAT_BASE_URL": "http://127.0.0.1:9000/v1",
                },
            )

            self.assertEqual(config.chat_provider, "vllm")
            self.assertEqual(config.chat_base_url, "http://127.0.0.1:9000/v1")

    def test_local_provider_urls_canonicalize_localhost_and_preserve_ipv6(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            localhost = load_config(
                project_root=root,
                env={
                    "OLLAMA_URL": "http://localhost:11434/",
                },
            )
            ipv6 = load_config(
                project_root=root,
                env={
                    "OLLAMA_URL": "http://[::1]:11434",
                },
            )

            self.assertEqual(localhost.ollama_url, "http://127.0.0.1:11434")
            self.assertEqual(ipv6.ollama_url, "http://[::1]:11434")

    def test_remote_provider_urls_from_environment_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for env in (
                {"OLLAMA_URL": "https://example.com"},
                {
                    "ATLAS_CHAT_PROVIDER": "vllm",
                    "ATLAS_CHAT_BASE_URL": "http://192.168.1.8:8000/v1",
                },
            ):
                with self.subTest(env=env):
                    with self.assertRaisesRegex(ValueError, "loopback"):
                        load_config(project_root=root, env=env)

    def test_blank_local_provider_base_url_uses_provider_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(
                project_root=root,
                env={
                    "ATLAS_CHAT_PROVIDER": "localai",
                    "ATLAS_CHAT_BASE_URL": "",
                },
            )

            self.assertEqual(config.chat_provider, "localai")
            self.assertEqual(config.chat_base_url, "http://127.0.0.1:8080/v1")

    def test_saved_provider_settings_override_source_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "runtime-data"
            save_provider_settings(
                data_dir,
                provider="lmstudio",
                base_url="http://127.0.0.1:4321/v1",
                api_key=None,
            )

            config = load_config(
                project_root=root,
                env={
                    "ATLAS_DATA_DIR": str(data_dir),
                    "ATLAS_CHAT_PROVIDER": "ollama",
                },
            )

            self.assertEqual(config.chat_provider, "lmstudio")
            self.assertEqual(config.chat_base_url, "http://127.0.0.1:4321/v1")

    def test_saved_ollama_url_applies_to_chat_discovery_and_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "runtime-data"
            save_provider_settings(
                data_dir,
                provider="ollama",
                base_url="http://127.0.0.1:22434",
                api_key=None,
            )

            config = load_config(
                project_root=root,
                env={"ATLAS_DATA_DIR": str(data_dir)},
            )

            self.assertEqual(config.chat_base_url, "http://127.0.0.1:22434")
            self.assertEqual(config.ollama_url, "http://127.0.0.1:22434")

    def test_provider_settings_never_expose_protected_key_material(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            result = save_provider_settings(
                data_dir,
                provider="ollama",
                base_url="http://127.0.0.1:11434",
                api_key=None,
            )

            self.assertFalse(result["has_api_key"])
            self.assertNotIn("api_key", result)
            self.assertEqual(load_provider_settings(data_dir)["provider"], "ollama")

    def test_provider_key_is_preserved_only_for_the_same_provider_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            with patch(
                "atlas_local.provider_settings.protect_bytes",
                return_value=b"protected-key",
            ):
                save_provider_settings(
                    data_dir,
                    provider="openai-compatible",
                    base_url="http://127.0.0.1:8000/v1",
                    api_key="local-secret",
                )

            unchanged = save_provider_settings(
                data_dir,
                provider="openai-compatible",
                base_url="http://127.0.0.1:8000/v1",
                api_key=None,
                preserve_existing_key=True,
            )
            changed = save_provider_settings(
                data_dir,
                provider="lmstudio",
                base_url="http://127.0.0.1:1234/v1",
                api_key=None,
                preserve_existing_key=True,
            )

            self.assertTrue(unchanged["has_api_key"])
            self.assertFalse(changed["has_api_key"])
            raw = json.loads(
                (data_dir / "provider-settings.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("protected_api_key", raw)

    def test_provider_settings_reject_remote_urls_before_persisting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "loopback"):
                save_provider_settings(
                    Path(tmp),
                    provider="openai-compatible",
                    base_url="https://example.com/v1",
                    api_key=None,
                )

    def test_legacy_remote_saved_url_falls_back_to_safe_provider_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "runtime-data"
            data_dir.mkdir()
            (data_dir / "provider-settings.json").write_text(
                json.dumps(
                    {
                        "format": "atlas-provider-settings-v1",
                        "provider": "lmstudio",
                        "base_url": "https://example.com/v1",
                    }
                ),
                encoding="utf-8",
            )

            saved = load_provider_settings(data_dir)
            config = load_config(
                project_root=root,
                env={"ATLAS_DATA_DIR": str(data_dir)},
            )

            self.assertEqual(saved["base_url"], "")
            self.assertEqual(saved["base_url_invalid"], "true")
            self.assertEqual(config.chat_provider, "lmstudio")
            self.assertEqual(config.chat_base_url, "http://127.0.0.1:1234/v1")

    def test_data_dir_override_redirects_legacy_example_storage_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            data_dir = Path(tmp) / "isolated"
            root.mkdir()

            config = load_config(
                project_root=root,
                env={
                    "ATLAS_DATA_DIR": str(data_dir),
                    "QDRANT_PATH": ".data/qdrant",
                    "LANGGRAPH_CHECKPOINT_DB": ".data/langgraph/checkpoints.sqlite",
                    "MEM0_HISTORY_DB": ".data/mem0_history.sqlite",
                },
            )

            self.assertEqual(config.qdrant_path, data_dir / "qdrant")
            self.assertEqual(config.langgraph_checkpoint_db, data_dir / "langgraph" / "checkpoints.sqlite")
            self.assertEqual(config.mem0_history_db, data_dir / "mem0_history.sqlite")

    def test_compaction_timeout_is_clamped_to_the_service_effective_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            too_high = load_config(
                project_root=root,
                env={"ATLAS_COMPACTION_TIMEOUT_SECONDS": "999"},
            )
            too_low = load_config(
                project_root=root,
                env={"ATLAS_COMPACTION_TIMEOUT_SECONDS": "0"},
            )

            self.assertEqual(too_high.compaction_timeout_seconds, 180.0)
            self.assertEqual(too_low.compaction_timeout_seconds, 1.0)

    def test_legacy_plaintext_migration_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default = load_config(project_root=root, env={})
            enabled = load_config(
                project_root=root,
                env={"ATLAS_ALLOW_LEGACY_PLAINTEXT_MIGRATION": "yes"},
            )

            self.assertFalse(default.allow_legacy_plaintext_migration)
            self.assertTrue(enabled.allow_legacy_plaintext_migration)


if __name__ == "__main__":
    unittest.main()
