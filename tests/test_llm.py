import io
import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from atlas_local.llm import (
    LLMProvider,
    MAX_MODEL_CATALOG_ENTRIES,
    MAX_MODEL_CATALOG_RESPONSE_BYTES,
    OpenAICompatibleChat,
    _ollama_json_request,
    _openai_compatible_json_request_required,
    _stream_openai_compatible_chat,
    format_runtime_error,
    inspect_openai_compatible_models,
    inspect_local_ollama_models,
    loaded_ollama_models,
    unload_ollama_model,
)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class _BlockingResponse:
    def __init__(self) -> None:
        self.read_started = threading.Event()
        self.closed = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def readline(self, _limit: int = -1) -> bytes:
        self.read_started.set()
        if not self.closed.wait(5.0):
            raise AssertionError("Timed out waiting for the response to be closed.")
        raise OSError("response closed")

    def close(self) -> None:
        self.closed.set()


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
        self.assertEqual(
            kwargs["client_kwargs"],
            {
                "follow_redirects": False,
                "timeout": 300.0,
                "trust_env": False,
            },
        )

    @patch("atlas_local.llm.ChatOllama")
    def test_json_chat_disables_redirects_and_environment_proxies(
        self,
        chat_ollama_mock,
    ) -> None:
        provider = LLMProvider(self.config)

        provider.json_chat("gpt-oss:20b")

        _, kwargs = chat_ollama_mock.call_args
        self.assertEqual(
            kwargs["client_kwargs"],
            {
                "follow_redirects": False,
                "timeout": 300.0,
                "trust_env": False,
            },
        )

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

    def test_openai_compatible_provider_invokes_chat_completions(self) -> None:
        config = SimpleNamespace(
            chat_provider="lmstudio",
            chat_base_url="http://127.0.0.1:1234/v1",
            chat_api_key=None,
            embed_model="nomic-embed-text:v1.5",
            ollama_url="http://127.0.0.1:11434",
        )
        captured: dict[str, object] = {}

        def fake_urlopen(request_object, timeout=0):
            captured["url"] = getattr(request_object, "full_url", str(request_object))
            captured["method"] = request_object.get_method()
            captured["body"] = json.loads((request_object.data or b"{}").decode("utf-8"))
            return _FakeResponse(
                json.dumps({"choices": [{"message": {"content": "hello from lm studio"}}]}).encode("utf-8")
            )

        with patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen):
            response = LLMProvider(config).chat("local-model").invoke([HumanMessage(content="hi")])

        self.assertEqual(str(response.content), "hello from lm studio")
        self.assertEqual(captured["url"], "http://127.0.0.1:1234/v1/chat/completions")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(
            captured["body"],
            {
                "model": "local-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

    def test_openai_compatible_chat_rejects_remote_base_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            OpenAICompatibleChat(
                model="local-model",
                base_url="https://example.com/v1",
                api_key="must-not-leak",
            )

    def test_openai_compatible_stream_does_not_retry_after_partial_output(self) -> None:
        chat = OpenAICompatibleChat(
            model="local-model",
            base_url="http://127.0.0.1:1234/v1",
        )

        def interrupted_stream(*_args, **_kwargs):
            yield "partial answer"
            raise RuntimeError("connection dropped")

        with (
            patch("atlas_local.llm._stream_openai_compatible_chat", side_effect=interrupted_stream),
            patch.object(chat, "invoke") as invoke_mock,
        ):
            chunks = chat.stream([HumanMessage(content="hi")])
            self.assertEqual(str(next(chunks).content), "partial answer")
            with self.assertRaisesRegex(RuntimeError, "connection dropped"):
                next(chunks)

        invoke_mock.assert_not_called()

    def test_openai_compatible_stream_falls_back_before_any_output(self) -> None:
        chat = OpenAICompatibleChat(
            model="local-model",
            base_url="http://127.0.0.1:1234/v1",
        )
        responses = [
            _FakeResponse(b'{"choices":[{"message":{"content":"ignored JSON stream"}}]}'),
            _FakeResponse(
                b'{"choices":[{"message":{"content":"non-streaming fallback"}}]}'
            ),
        ]

        with patch("atlas_local.llm.provider_urlopen", side_effect=responses) as urlopen_mock:
            chunks = list(chat.stream([HumanMessage(content="hi")]))

        self.assertEqual([str(chunk.content) for chunk in chunks], ["non-streaming fallback"])
        self.assertEqual(urlopen_mock.call_count, 2)

    def test_openai_compatible_stream_does_not_fallback_for_generic_pre_token_failure(
        self,
    ) -> None:
        chat = OpenAICompatibleChat(
            model="local-model",
            base_url="http://127.0.0.1:1234/v1",
        )

        def failed_stream(*_args, **_kwargs):
            raise RuntimeError("connection dropped")
            yield  # pragma: no cover - keeps this helper as a generator

        with (
            patch(
                "atlas_local.llm._stream_openai_compatible_chat",
                side_effect=failed_stream,
            ),
            patch.object(chat, "_invoke_at_epoch") as invoke_mock,
            self.assertRaisesRegex(RuntimeError, "connection dropped"),
        ):
            list(chat.stream([HumanMessage(content="hi")]))

        invoke_mock.assert_not_called()

    def test_openai_compatible_abort_before_first_token_never_starts_fallback(
        self,
    ) -> None:
        chat = OpenAICompatibleChat(
            model="local-model",
            base_url="http://127.0.0.1:1234/v1",
        )
        response = _BlockingResponse()
        failures: list[BaseException] = []

        def consume_stream() -> None:
            try:
                list(chat.stream([HumanMessage(content="hi")]))
            except BaseException as exc:
                failures.append(exc)

        with (
            patch("atlas_local.llm.provider_urlopen", return_value=response),
            patch.object(chat, "_invoke_at_epoch") as invoke_mock,
        ):
            worker = threading.Thread(target=consume_stream)
            worker.start()
            self.assertTrue(response.read_started.wait(2.0))
            chat.abort()
            worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIn("aborted", str(failures[0]).lower())
        invoke_mock.assert_not_called()

    def test_openai_compatible_stream_preserves_reasoning_deltas(self) -> None:
        chat = OpenAICompatibleChat(
            model="local-model",
            base_url="http://127.0.0.1:1234/v1",
        )
        response = _FakeResponse(
            b'data: {"choices":[{"delta":{"reasoning_content":"scratch"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"final"}}]}\n\n'
            b"data: [DONE]\n\n"
        )

        with patch("atlas_local.llm.provider_urlopen", return_value=response):
            chunks = list(chat.stream([HumanMessage(content="hi")]))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(str(chunks[0].content), "")
        self.assertEqual(chunks[0].additional_kwargs["reasoning_content"], "scratch")
        self.assertEqual(str(chunks[1].content), "final")

    def test_openai_compatible_json_reader_rejects_oversized_response(self) -> None:
        with (
            patch(
                "atlas_local.llm.provider_urlopen",
                return_value=_FakeResponse(b'{"answer":"' + (b"x" * 64) + b'"}'),
            ),
            self.assertRaisesRegex(RuntimeError, "exceeded the 32-byte limit"),
        ):
            _openai_compatible_json_request_required(
                "http://127.0.0.1:1234/v1",
                "chat/completions",
                timeout_seconds=1.0,
                max_response_bytes=32,
            )

    def test_openai_compatible_stream_rejects_oversized_event(self) -> None:
        response = _FakeResponse(b"data: " + (b"x" * 64) + b"\n")

        with (
            patch("atlas_local.llm.provider_urlopen", return_value=response),
            patch("atlas_local.llm.MAX_PROVIDER_STREAM_EVENT_BYTES", 32),
            self.assertRaisesRegex(ValueError, "exceeded the 32-byte limit"),
        ):
            list(
                _stream_openai_compatible_chat(
                    "http://127.0.0.1:1234/v1",
                    {"model": "local-model", "messages": [], "stream": True},
                    timeout_seconds=1.0,
                )
            )

    def test_openai_compatible_abort_closes_active_transport_response(self) -> None:
        chat = OpenAICompatibleChat(
            model="local-model",
            base_url="http://127.0.0.1:1234/v1",
        )
        response = _FakeResponse(b"")
        chat._register_active_response(response)

        chat.abort()

        self.assertTrue(response.closed)
        chat._unregister_active_response(response)
        self.assertEqual(chat._active_responses, [])

    def test_provider_abort_propagates_to_openai_compatible_chat(self) -> None:
        config = SimpleNamespace(
            chat_provider="lmstudio",
            chat_base_url="http://127.0.0.1:1234/v1",
            chat_api_key=None,
            embed_model="nomic-embed-text:v1.5",
            ollama_url="http://127.0.0.1:11434",
        )
        provider = LLMProvider(config)
        chat = provider.chat("local-model")

        with patch.object(chat, "abort") as abort_mock:
            provider.abort_active_requests()

        abort_mock.assert_called_once_with()

    def test_inspect_openai_compatible_models_reads_v1_models(self) -> None:
        config = SimpleNamespace(
            chat_provider="llamacpp",
            chat_base_url="http://127.0.0.1:8080/v1",
            chat_api_key=None,
            embed_model="nomic-embed-text:v1.5",
            ollama_url="http://127.0.0.1:11434",
        )

        def fake_urlopen(request_object, timeout=0):
            self.assertEqual(getattr(request_object, "full_url", str(request_object)), "http://127.0.0.1:8080/v1/models")
            return _FakeResponse(json.dumps({"data": [{"id": "qwen3:8b"}, {"id": "nomic-embed-text"}]}).encode("utf-8"))

        with patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen):
            catalog = inspect_openai_compatible_models(config)

        self.assertEqual(catalog.provider, "llamacpp")
        self.assertEqual(catalog.provider_label, "llama.cpp server")
        self.assertTrue(catalog.provider_online)
        self.assertFalse(catalog.ollama_online)
        self.assertEqual([model.name for model in catalog.models], ["qwen3:8b"])

    def test_inspect_models_reads_ollama_show_vision_capability(self) -> None:
        responses = {
            "api/tags": {
                "models": [
                    {
                        "name": "qwen3.6:27b",
                        "model": "qwen3.6:27b",
                        "size": 17_741_872_154,
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

        with patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen):
            catalog = inspect_local_ollama_models(self.config)

        self.assertEqual(catalog.models[0].name, "qwen3.6:27b")
        self.assertEqual(catalog.models[0].size_bytes, 17_741_872_154)
        self.assertEqual(catalog.models[0].capabilities, ("completion", "vision"))
        self.assertTrue(catalog.models[0].supports_images)

    def test_inspect_models_rejects_invalid_catalog_size(self) -> None:
        payload = {
            "models": [
                {
                    "name": "local-chat:latest",
                    "size": -1,
                    "details": {"family": "test"},
                }
            ]
        }
        with (
            patch(
                "atlas_local.llm.provider_urlopen",
                return_value=_FakeResponse(json.dumps(payload).encode("utf-8")),
            ),
            patch(
                "atlas_local.llm._ollama_json_request",
                return_value={"capabilities": ["completion"]},
            ),
        ):
            catalog = inspect_local_ollama_models(self.config)

        self.assertIsNone(catalog.models[0].size_bytes)

    def test_inspect_models_probes_catalog_entries_concurrently(self) -> None:
        responses = {
            "api/tags": {
                "models": [
                    {"name": f"model-{index}", "details": {"family": "test"}}
                    for index in range(4)
                ]
            },
        }
        active = 0
        max_active = 0
        active_lock = threading.Lock()

        def fake_urlopen(request_object, timeout=0):
            return _FakeResponse(json.dumps(responses["api/tags"]).encode("utf-8"))

        def fake_show(*_args, **_kwargs):
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.03)
                return {"capabilities": ["completion"]}
            finally:
                with active_lock:
                    active -= 1

        with (
            patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen),
            patch("atlas_local.llm._ollama_json_request", side_effect=fake_show),
        ):
            catalog = inspect_local_ollama_models(self.config)

        self.assertEqual([item.name for item in catalog.models], [f"model-{index}" for index in range(4)])
        self.assertGreaterEqual(max_active, 2)

    def test_inspect_models_rejects_oversized_catalog_response(self) -> None:
        oversized = b"{" + (b" " * MAX_MODEL_CATALOG_RESPONSE_BYTES) + b"}"

        with patch(
            "atlas_local.llm.provider_urlopen",
            return_value=_FakeResponse(oversized),
        ):
            catalog = inspect_local_ollama_models(self.config)

        self.assertFalse(catalog.provider_online)
        self.assertEqual(catalog.models, ())

    def test_inspect_models_handles_non_list_catalog(self) -> None:
        with patch(
            "atlas_local.llm.provider_urlopen",
            return_value=_FakeResponse(b'{"models": {}}'),
        ):
            catalog = inspect_local_ollama_models(self.config)

        self.assertTrue(catalog.provider_online)
        self.assertEqual(catalog.models, ())

    def test_inspect_models_skips_oversized_names(self) -> None:
        payload = {
            "models": [
                {
                    "name": "x" * 201,
                    "details": {"family": "test"},
                },
                {
                    "name": "valid-model",
                    "details": {"family": "test"},
                },
            ]
        }
        with (
            patch(
                "atlas_local.llm.provider_urlopen",
                return_value=_FakeResponse(json.dumps(payload).encode("utf-8")),
            ),
            patch(
                "atlas_local.llm._ollama_json_request",
                return_value={"capabilities": ["completion"]},
            ),
        ):
            catalog = inspect_local_ollama_models(self.config)

        self.assertEqual([model.name for model in catalog.models], ["valid-model"])

    def test_inspect_models_has_a_global_probe_deadline_and_uses_tag_fallbacks(
        self,
    ) -> None:
        payload = {
            "models": [
                {
                    "name": f"model-{index}",
                    "details": {"family": "test"},
                }
                for index in range(16)
            ]
        }

        def slow_show(*_args, **_kwargs):
            time.sleep(0.2)
            return {"capabilities": ["completion"]}

        started_at = time.monotonic()
        with (
            patch(
                "atlas_local.llm.provider_urlopen",
                return_value=_FakeResponse(json.dumps(payload).encode("utf-8")),
            ),
            patch(
                "atlas_local.llm._ollama_json_request",
                side_effect=slow_show,
            ),
        ):
            catalog = inspect_local_ollama_models(
                self.config,
                timeout_seconds=0.05,
            )
        elapsed = time.monotonic() - started_at

        self.assertLess(elapsed, 0.15)
        self.assertEqual(len(catalog.models), 16)

    def test_ollama_json_reader_rejects_oversized_response(self) -> None:
        with patch(
            "atlas_local.llm.provider_urlopen",
            return_value=_FakeResponse(b'{"value":"' + (b"x" * 64) + b'"}'),
        ):
            payload = _ollama_json_request(
                self.config,
                "api/ps",
                timeout_seconds=1.0,
                max_response_bytes=32,
            )

        self.assertEqual(payload, {})

    def test_openai_compatible_catalog_caps_entries(self) -> None:
        config = SimpleNamespace(
            chat_provider="llamacpp",
            chat_base_url="http://127.0.0.1:8080/v1",
            chat_api_key=None,
            embed_model="nomic-embed-text:v1.5",
            ollama_url="http://127.0.0.1:11434",
        )
        payload = {
            "data": [
                {"id": f"model-{index}", "capabilities": ["completion"]}
                for index in range(MAX_MODEL_CATALOG_ENTRIES + 10)
            ]
        }

        with patch(
            "atlas_local.llm.provider_urlopen",
            return_value=_FakeResponse(json.dumps(payload).encode("utf-8")),
        ):
            catalog = inspect_openai_compatible_models(config)

        self.assertEqual(len(catalog.models), MAX_MODEL_CATALOG_ENTRIES)

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

        with patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen):
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

        with patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen):
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

        with patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen):
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

        with patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen):
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

        with patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen):
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

        with patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen):
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

        with patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen):
            catalog = inspect_local_ollama_models(self.config)

        self.assertEqual(catalog.models[0].name, "gemma3:4b")
        self.assertTrue(catalog.models[0].supports_images)

    def test_loaded_ollama_models_reads_runtime_inventory(self) -> None:
        responses = {
            "api/ps": {
                "models": [
                    {"name": "qwen3-coder:30b"},
                    {"model": "gpt-oss:20b"},
                    {"name": "qwen3-coder:30b"},
                    {"name": ""},
                ]
            }
        }

        def fake_urlopen(request_object, timeout=0):
            url = getattr(request_object, "full_url", str(request_object))
            self.assertTrue(url.endswith("/api/ps"))
            return _FakeResponse(json.dumps(responses["api/ps"]).encode("utf-8"))

        with patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen):
            loaded = loaded_ollama_models(self.config)

        self.assertEqual(loaded, ["qwen3-coder:30b", "gpt-oss:20b"])

    def test_unload_ollama_model_requests_keep_alive_zero(self) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request_object, timeout=0):
            captured["url"] = getattr(request_object, "full_url", str(request_object))
            captured["method"] = request_object.get_method()
            captured["body"] = json.loads((request_object.data or b"{}").decode("utf-8"))
            return _FakeResponse(json.dumps({"done": True, "done_reason": "unload"}).encode("utf-8"))

        with patch("atlas_local.llm.provider_urlopen", side_effect=fake_urlopen):
            payload = unload_ollama_model(self.config, "qwen3-coder:30b")

        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/generate")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(
            captured["body"],
            {"model": "qwen3-coder:30b", "prompt": "", "stream": False, "keep_alive": 0},
        )
        self.assertEqual(payload["status"], "unloaded")
        self.assertEqual(payload["model"], "qwen3-coder:30b")


if __name__ == "__main__":
    unittest.main()
