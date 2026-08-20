from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from atlas_local.config import load_config
from atlas_local.memory.mem0_service import _load_mem0_runtime


class LocalOllamaClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        config = load_config(project_root=Path(self.temp_dir.name), env={})
        _load_mem0_runtime(config.data_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_secure_client_is_loopback_proxy_free_and_redirect_free(self) -> None:
        from atlas_local.memory.local_ollama import secure_ollama_client

        with patch("atlas_local.memory.local_ollama.Client") as client_mock:
            secure_ollama_client("http://localhost:11434/")

        client_mock.assert_called_once_with(
            host="http://127.0.0.1:11434",
            follow_redirects=False,
            trust_env=False,
            timeout=120.0,
        )

    def test_secure_client_rejects_remote_host_before_construction(self) -> None:
        from atlas_local.memory.local_ollama import secure_ollama_client

        with (
            patch("atlas_local.memory.local_ollama.Client") as client_mock,
            self.assertRaisesRegex(ValueError, "loopback"),
        ):
            secure_ollama_client("https://example.com")

        client_mock.assert_not_called()

    def test_embedder_uses_secure_client_before_first_model_probe(self) -> None:
        from atlas_local.memory.local_ollama import AtlasOllamaEmbedding

        client = Mock()
        client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}],
        }
        config = SimpleNamespace(
            model="nomic-embed-text",
            embedding_dims=768,
            ollama_base_url="http://127.0.0.1:11434",
        )

        with patch(
            "atlas_local.memory.local_ollama.secure_ollama_client",
            return_value=client,
        ) as client_factory:
            embedder = AtlasOllamaEmbedding(config)

        self.assertIs(embedder.client, client)
        client_factory.assert_called_once_with("http://127.0.0.1:11434")
        client.list.assert_called_once_with()
        client.pull.assert_not_called()

    def test_embedder_runs_single_embeddings_on_cpu_and_unloads_immediately(
        self,
    ) -> None:
        from atlas_local.memory.local_ollama import AtlasOllamaEmbedding

        client = Mock()
        client.list.return_value = {
            "models": [{"name": "nomic-embed-text:v1.5"}],
        }
        client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        config = SimpleNamespace(
            model="nomic-embed-text:v1.5",
            embedding_dims=768,
            ollama_base_url="http://127.0.0.1:11434",
        )

        with patch(
            "atlas_local.memory.local_ollama.secure_ollama_client",
            return_value=client,
        ):
            embedder = AtlasOllamaEmbedding(config)

        self.assertEqual(embedder.embed("local memory", "search"), [0.1, 0.2, 0.3])
        client.embed.assert_called_once_with(
            model="nomic-embed-text:v1.5",
            input="local memory",
            options={"num_gpu": 0},
            keep_alive=0,
        )

    def test_embedder_runs_batches_on_cpu_and_preserves_result_count(self) -> None:
        from atlas_local.memory.local_ollama import AtlasOllamaEmbedding

        client = Mock()
        client.list.return_value = {
            "models": [{"name": "nomic-embed-text:v1.5"}],
        }
        client.embed.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }
        config = SimpleNamespace(
            model="nomic-embed-text:v1.5",
            embedding_dims=768,
            ollama_base_url="http://127.0.0.1:11434",
        )

        with patch(
            "atlas_local.memory.local_ollama.secure_ollama_client",
            return_value=client,
        ):
            embedder = AtlasOllamaEmbedding(config)

        self.assertEqual(
            embedder.embed_batch(["first", "second"]),
            [[0.1, 0.2], [0.3, 0.4]],
        )
        client.embed.assert_called_once_with(
            model="nomic-embed-text:v1.5",
            input=["first", "second"],
            options={"num_gpu": 0},
            keep_alive=0,
        )

    def test_llm_replaces_inherited_client_before_use(self) -> None:
        from atlas_local.memory.local_ollama import AtlasOllamaLLM

        inherited_client = Mock()
        secured_client = Mock()
        config = SimpleNamespace(
            model="nomic-embed-text",
            temperature=0.0,
            max_tokens=600,
            top_p=0.1,
            top_k=1,
            enable_vision=False,
            vision_details="auto",
            http_client_proxies=None,
            ollama_base_url="http://127.0.0.1:11434",
        )

        with (
            patch(
                "mem0.llms.ollama.Client",
                return_value=inherited_client,
            ),
            patch(
                "atlas_local.memory.local_ollama.secure_ollama_client",
                return_value=secured_client,
            ) as client_factory,
        ):
            llm = AtlasOllamaLLM(config)

        inherited_client.close.assert_called_once_with()
        client_factory.assert_called_once_with("http://127.0.0.1:11434")
        self.assertIs(llm.client, secured_client)

    def test_mem0_factories_are_rebound_to_secure_atlas_adapters(self) -> None:
        from mem0.utils.factory import EmbedderFactory, LlmFactory

        from atlas_local.memory.mem0_service import _secure_mem0_ollama_factories

        with (
            patch.dict(LlmFactory.provider_to_class, {}, clear=False),
            patch.dict(EmbedderFactory.provider_to_class, {}, clear=False),
        ):
            _secure_mem0_ollama_factories()

            self.assertEqual(
                LlmFactory.provider_to_class["ollama"][0],
                "atlas_local.memory.local_ollama.AtlasOllamaLLM",
            )
            self.assertEqual(
                EmbedderFactory.provider_to_class["ollama"],
                "atlas_local.memory.local_ollama.AtlasOllamaEmbedding",
            )


if __name__ == "__main__":
    unittest.main()
