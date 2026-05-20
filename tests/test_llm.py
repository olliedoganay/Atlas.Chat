import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from atlas_local.llm import LLMProvider, format_runtime_error, inspect_local_ollama_models


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


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

    @patch("atlas_local.llm.ChatOllama")
    def test_chat_omits_reasoning_for_qwen3_coder(self, chat_ollama_mock) -> None:
        provider = LLMProvider(self.config)

        provider.chat("qwen3-coder:30b", temperature=0.2, reasoning=True)

        _, kwargs = chat_ollama_mock.call_args
        self.assertNotIn("reasoning", kwargs)

    def test_format_runtime_error_preserves_cuda_detail(self) -> None:
        error = format_runtime_error(
            self.config,
            RuntimeError("CUDA error: operation not permitted"),
            chat_model="gpt-oss:20b",
        )

        message = str(error)
        self.assertIn("Original error: CUDA error: operation not permitted", message)
        self.assertIn("GPU/CUDA runtime failure", message)

    def test_format_runtime_error_explains_cuda_out_of_memory(self) -> None:
        error = format_runtime_error(
            self.config,
            RuntimeError("CUDA error: out of memory"),
            chat_model="qwen3.6:27b",
        )

        message = str(error)
        self.assertIn("GPU memory exhaustion", message)
        self.assertIn("Ollama context window", message)

    def test_format_runtime_error_explains_context_limit_failures(self) -> None:
        error = format_runtime_error(
            self.config,
            RuntimeError("prompt too long for context length"),
            chat_model="qwen3-coder:30b",
        )

        message = str(error)
        self.assertIn("prompt appears too large", message)
        self.assertIn("Compact the thread", message)

    def test_inspect_models_reads_ollama_show_vision_capability(self) -> None:
        responses = {
            "api/tags": {
                "models": [
                    {
                        "name": "qwen3.6:27b",
                        "model": "qwen3.6:27b",
                        "details": {"family": "qwen3", "families": ["qwen3"]},
                    }
                ]
            },
            "api/show": {"capabilities": ["completion", "vision"]},
        }

        def fake_urlopen(request_object, timeout=0):
            url = getattr(request_object, "full_url", str(request_object))
            key = "api/show" if url.endswith("/api/show") else "api/tags"
            return _FakeResponse(json.dumps(responses[key]).encode("utf-8"))

        with patch("atlas_local.llm.request.urlopen", side_effect=fake_urlopen):
            catalog = inspect_local_ollama_models(self.config)

        self.assertEqual(catalog.models[0].name, "qwen3.6:27b")
        self.assertEqual(catalog.models[0].capabilities, ("completion", "vision"))
        self.assertTrue(catalog.models[0].supports_images)

    def test_inspect_models_trusts_declared_capabilities_over_name_vision_markers(self) -> None:
        responses = {
            "api/tags": {
                "models": [
                    {
                        "name": "llava-custom:latest",
                        "model": "llava-custom:latest",
                        "details": {"family": "llama", "families": ["llama"]},
                    }
                ]
            },
            "api/show": {"capabilities": ["completion"]},
        }

        def fake_urlopen(request_object, timeout=0):
            url = getattr(request_object, "full_url", str(request_object))
            key = "api/show" if url.endswith("/api/show") else "api/tags"
            return _FakeResponse(json.dumps(responses[key]).encode("utf-8"))

        with patch("atlas_local.llm.request.urlopen", side_effect=fake_urlopen):
            catalog = inspect_local_ollama_models(self.config)

        self.assertEqual(catalog.models[0].name, "llava-custom:latest")
        self.assertFalse(catalog.models[0].supports_images)

    def test_inspect_models_keeps_qwen36_coding_variant_text_only(self) -> None:
        responses = {
            "api/tags": {
                "models": [
                    {
                        "name": "qwen3.6:27b-coding-nvfp4",
                        "model": "qwen3.6:27b-coding-nvfp4",
                        "details": {"family": "qwen3", "families": ["qwen3"]},
                    }
                ]
            },
            "api/show": {"capabilities": ["completion"]},
        }

        def fake_urlopen(request_object, timeout=0):
            url = getattr(request_object, "full_url", str(request_object))
            key = "api/show" if url.endswith("/api/show") else "api/tags"
            return _FakeResponse(json.dumps(responses[key]).encode("utf-8"))

        with patch("atlas_local.llm.request.urlopen", side_effect=fake_urlopen):
            catalog = inspect_local_ollama_models(self.config)

        self.assertEqual(catalog.models[0].name, "qwen3.6:27b-coding-nvfp4")
        self.assertFalse(catalog.models[0].supports_images)

    def test_inspect_models_disables_reasoning_for_qwen3_coder_without_thinking_capability(self) -> None:
        responses = {
            "api/tags": {
                "models": [
                    {
                        "name": "qwen3-coder:30b",
                        "model": "qwen3-coder:30b",
                        "details": {"family": "qwen3moe", "families": ["qwen3moe"]},
                    }
                ]
            },
            "api/show": {"capabilities": ["completion", "tools"]},
        }

        def fake_urlopen(request_object, timeout=0):
            url = getattr(request_object, "full_url", str(request_object))
            key = "api/show" if url.endswith("/api/show") else "api/tags"
            return _FakeResponse(json.dumps(responses[key]).encode("utf-8"))

        with patch("atlas_local.llm.request.urlopen", side_effect=fake_urlopen):
            catalog = inspect_local_ollama_models(self.config)

        self.assertEqual(catalog.models[0].name, "qwen3-coder:30b")
        self.assertFalse(catalog.models[0].supports_reasoning)
        self.assertEqual(catalog.models[0].reasoning_mode_strategy, "none")

    def test_inspect_models_trusts_declared_capabilities_over_name_reasoning_markers(self) -> None:
        responses = {
            "api/tags": {
                "models": [
                    {
                        "name": "qwen3:8b",
                        "model": "qwen3:8b",
                        "details": {"family": "qwen3", "families": ["qwen3"]},
                    }
                ]
            },
            "api/show": {"capabilities": ["completion"]},
        }

        def fake_urlopen(request_object, timeout=0):
            url = getattr(request_object, "full_url", str(request_object))
            key = "api/show" if url.endswith("/api/show") else "api/tags"
            return _FakeResponse(json.dumps(responses[key]).encode("utf-8"))

        with patch("atlas_local.llm.request.urlopen", side_effect=fake_urlopen):
            catalog = inspect_local_ollama_models(self.config)

        self.assertEqual(catalog.models[0].name, "qwen3:8b")
        self.assertFalse(catalog.models[0].supports_reasoning)
        self.assertEqual(catalog.models[0].reasoning_mode_strategy, "none")

    def test_inspect_models_uses_show_thinking_capability_for_reasoning(self) -> None:
        responses = {
            "api/tags": {
                "models": [
                    {
                        "name": "qwen3.6:27b",
                        "model": "qwen3.6:27b",
                        "details": {"family": "qwen3", "families": ["qwen3"]},
                    }
                ]
            },
            "api/show": {"capabilities": ["completion", "vision", "thinking"]},
        }

        def fake_urlopen(request_object, timeout=0):
            url = getattr(request_object, "full_url", str(request_object))
            key = "api/show" if url.endswith("/api/show") else "api/tags"
            return _FakeResponse(json.dumps(responses[key]).encode("utf-8"))

        with patch("atlas_local.llm.request.urlopen", side_effect=fake_urlopen):
            catalog = inspect_local_ollama_models(self.config)

        self.assertEqual(catalog.models[0].name, "qwen3.6:27b")
        self.assertTrue(catalog.models[0].supports_reasoning)
        self.assertEqual(catalog.models[0].reasoning_mode_strategy, "boolean")

    def test_inspect_models_uses_thinking_capability_for_unknown_model(self) -> None:
        responses = {
            "api/tags": {
                "models": [
                    {
                        "name": "local-thinking:latest",
                        "model": "local-thinking:latest",
                        "details": {"family": "custom", "families": ["custom"]},
                    }
                ]
            },
            "api/show": {"capabilities": ["completion", "thinking"]},
        }

        def fake_urlopen(request_object, timeout=0):
            url = getattr(request_object, "full_url", str(request_object))
            key = "api/show" if url.endswith("/api/show") else "api/tags"
            return _FakeResponse(json.dumps(responses[key]).encode("utf-8"))

        with patch("atlas_local.llm.request.urlopen", side_effect=fake_urlopen):
            catalog = inspect_local_ollama_models(self.config)

        self.assertEqual(catalog.models[0].name, "local-thinking:latest")
        self.assertTrue(catalog.models[0].supports_reasoning)
        self.assertEqual(catalog.models[0].reasoning_mode_strategy, "boolean")

    def test_inspect_models_uses_name_fallback_when_capabilities_are_absent(self) -> None:
        responses = {
            "api/tags": {
                "models": [
                    {
                        "name": "gemma3:4b",
                        "model": "gemma3:4b",
                        "details": {"family": "gemma3", "families": ["gemma3"]},
                    }
                ]
            },
            "api/show": {"details": {"family": "gemma3", "families": ["gemma3"]}},
        }

        def fake_urlopen(request_object, timeout=0):
            url = getattr(request_object, "full_url", str(request_object))
            key = "api/show" if url.endswith("/api/show") else "api/tags"
            return _FakeResponse(json.dumps(responses[key]).encode("utf-8"))

        with patch("atlas_local.llm.request.urlopen", side_effect=fake_urlopen):
            catalog = inspect_local_ollama_models(self.config)

        self.assertEqual(catalog.models[0].name, "gemma3:4b")
        self.assertTrue(catalog.models[0].supports_images)


if __name__ == "__main__":
    unittest.main()
