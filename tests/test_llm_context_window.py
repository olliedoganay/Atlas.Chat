import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atlas_local.config import load_config
from atlas_local.llm import LLMProvider, resolve_effective_context_window


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class LlmContextWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(env={"OLLAMA_URL": "http://127.0.0.1:11434"})

    def test_resolve_context_prefers_running_model_ps_value(self) -> None:
        responses = [
            {"models": [{"name": "gpt-oss:20b", "context_length": 16384}]},
        ]

        def fake_urlopen(request_object, timeout=0):
            payload = responses.pop(0)
            return _FakeResponse(json.dumps(payload).encode("utf-8"))

        with patch("atlas_local.llm.request.urlopen", side_effect=fake_urlopen):
            value = resolve_effective_context_window(self.config, "gpt-oss:20b")

        self.assertEqual(value, 16384)

    def test_resolve_context_uses_explicit_show_num_ctx_parameter(self) -> None:
        responses = [
            {"models": []},
            {
                "parameters": "temperature 0.7\nnum_ctx 24576\n",
                "model_info": {"gptoss.context_length": 131072},
            },
        ]

        def fake_urlopen(request_object, timeout=0):
            payload = responses.pop(0)
            return _FakeResponse(json.dumps(payload).encode("utf-8"))

        with patch("atlas_local.llm.request.urlopen", side_effect=fake_urlopen):
            value = resolve_effective_context_window(self.config, "gpt-oss:20b")

        self.assertEqual(value, 24576)

    def test_resolve_context_does_not_treat_show_model_info_as_effective_window(self) -> None:
        responses = [
            {"models": []},
            {"model_info": {"gptoss.context_length": 131072}},
        ]

        def fake_urlopen(request_object, timeout=0):
            payload = responses.pop(0)
            return _FakeResponse(json.dumps(payload).encode("utf-8"))

        with patch("atlas_local.llm.request.urlopen", side_effect=fake_urlopen):
            value = resolve_effective_context_window(self.config, "gpt-oss:20b")

        self.assertEqual(value, 8192)

    def test_provider_passes_configured_num_ctx_to_chat_models(self) -> None:
        captured: list[dict[str, object]] = []

        class _FakeChatOllama:
            def __init__(self, **kwargs):
                captured.append(kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            provider = LLMProvider(config)
            saved = provider.set_ollama_context_window(16384)

            with patch("atlas_local.llm.ChatOllama", _FakeChatOllama):
                provider.chat("gpt-oss:20b", temperature=0.2, reasoning=False)
                provider.json_chat("gpt-oss:20b")

        self.assertEqual(saved, 16384)
        self.assertEqual(captured[0]["num_ctx"], 16384)
        self.assertEqual(captured[0]["model"], "gpt-oss:20b")
        self.assertEqual(captured[1]["num_ctx"], 16384)
        self.assertEqual(captured[1]["format"], "json")

    def test_provider_persists_and_clears_ollama_context_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            provider = LLMProvider(config)

            provider.set_ollama_context_window(32768)
            reloaded = LLMProvider(config)
            loaded_value = reloaded.ollama_context_window()
            cleared = reloaded.set_ollama_context_window(None)

            final = LLMProvider(config)

        self.assertEqual(loaded_value, 32768)
        self.assertIsNone(cleared)
        self.assertIsNone(final.ollama_context_window())


if __name__ == "__main__":
    unittest.main()
