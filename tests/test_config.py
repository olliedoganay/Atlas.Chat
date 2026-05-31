import tempfile
import unittest
from pathlib import Path

from atlas_local.config import load_config


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


if __name__ == "__main__":
    unittest.main()
