import unittest
from types import SimpleNamespace
from unittest.mock import patch

from atlas_local.llm import LLMProvider, format_runtime_error


class LLMProviderTemperatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimpleNamespace(
            chat_model="gpt-oss:20b",
            embed_model="nomic-embed-text:v1.5",
            chat_temperature=0.2,
            ollama_url="http://127.0.0.1:11434",
        )

    @patch("atlas_local.llm.ChatOllama")
    def test_chat_omits_temperature_when_using_model_default(self, chat_ollama_mock) -> None:
        provider = LLMProvider(self.config)

        provider.chat("gpt-oss:20b", temperature=None, reasoning=True)

        _, kwargs = chat_ollama_mock.call_args
        self.assertTrue(kwargs["reasoning"])
        self.assertNotIn("temperature", kwargs)

    @patch("atlas_local.llm.ChatOllama")
    def test_chat_passes_explicit_temperature_override(self, chat_ollama_mock) -> None:
        provider = LLMProvider(self.config)

        provider.chat("gpt-oss:20b", temperature=0.7, reasoning=True)

        _, kwargs = chat_ollama_mock.call_args
        self.assertTrue(kwargs["reasoning"])
        self.assertEqual(kwargs["temperature"], 0.7)

    def test_format_runtime_error_preserves_cuda_detail(self) -> None:
        error = format_runtime_error(
            self.config,
            RuntimeError("CUDA error: operation not permitted"),
            chat_model="gpt-oss:20b",
        )

        message = str(error)
        self.assertIn("Original error: CUDA error: operation not permitted", message)
        self.assertIn("GPU/CUDA runtime failure", message)


if __name__ == "__main__":
    unittest.main()
