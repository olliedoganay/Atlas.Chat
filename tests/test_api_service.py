import unittest
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from atlas_local.api_service import AtlasBackendService, _estimate_thread_representation_tokens
from atlas_local.config import load_config
from atlas_local.graph.builder import execution_node_sequence, post_synthesis_node_sequence, pre_synthesis_node_sequence
from atlas_local.graph.context import GraphContext
from atlas_local.llm import OllamaCatalogSnapshot, OllamaModelInfo
from atlas_local.run_contract import RunHub
from atlas_local.run_store import RunStore
from atlas_local.security import application_secret_protection_available, local_secret_storage_label, sqlcipher_enabled


class ApiServiceCreateTests(unittest.TestCase):
    @patch("atlas_local.api_service.RunStore")
    @patch("atlas_local.api_service.build_chat_application")
    def test_create_uses_chat_only_builder(self, build_chat_application, run_store_cls) -> None:
        fake_config = object()
        fake_app = object()
        fake_store = object()
        build_chat_application.return_value = fake_app
        run_store_cls.return_value = fake_store

        service = AtlasBackendService.create(config=fake_config)

        build_chat_application.assert_called_once_with(fake_config)
        run_store_cls.assert_called_once_with(fake_config)
        self.assertIs(service.config, fake_config)
        self.assertIs(service.app, fake_app)
        self.assertIs(service.run_store, fake_store)

    def test_list_users_returns_store_entries_without_synthetic_default_user(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.run_store = SimpleNamespace(
            list_users=lambda: [{"user_id": "other_user", "updated_at": "2026-04-11T00:00:00Z"}]
        )

        users = AtlasBackendService.list_users(service)

        self.assertEqual(
            users,
            [
                {
                    "user_id": "other_user",
                    "updated_at": "2026-04-11T00:00:00Z",
                    "protection": "passwordless",
                    "locked": False,
                }
            ],
        )

    def test_status_reports_current_storage_protection_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(close=lambda: None),
                run_store=RunStore(config),
                run_hub=RunHub(),
            )

            payload = service.status()

            self.assertIn("security", payload)
            self.assertEqual(payload["security"]["profile_key_protection"], local_secret_storage_label())
            self.assertEqual(payload["security"]["run_artifacts_encrypted_at_rest"], application_secret_protection_available())
            self.assertEqual(payload["security"]["run_index_encrypted_at_rest"], application_secret_protection_available())
            self.assertEqual(payload["security"]["sqlite_encrypted_at_rest"], sqlcipher_enabled())
            self.assertEqual(payload["security"]["vector_store"], "local-qdrant")
            self.assertEqual(payload["security"]["vector_store_encrypted_at_rest"], sqlcipher_enabled())


class UserProtectionTests(unittest.TestCase):
    def test_list_users_marks_password_profiles_locked_until_unlocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("protected_user", password="atlas-secret")
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(close=lambda: None),
                run_store=store,
                run_hub=RunHub(),
            )

            users = service.list_users()

            self.assertEqual(users[0]["protection"], "password")
            self.assertTrue(users[0]["locked"])

    def test_unlock_user_requires_matching_password(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("protected_user", password="atlas-secret")
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(close=lambda: None),
                run_store=store,
                run_hub=RunHub(),
            )

            with self.assertRaisesRegex(RuntimeError, "Password did not match"):
                service.unlock_user(user_id="protected_user", password="wrong-secret")

            unlocked = service.unlock_user(user_id="protected_user", password="atlas-secret")

            self.assertFalse(unlocked["locked"])

    def test_locked_user_cannot_list_threads_until_unlocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("protected_user", password="atlas-secret")
            store.upsert_thread(
                user_id="protected_user",
                thread_id="main",
                title="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
            )
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(close=lambda: None),
                run_store=store,
                run_hub=RunHub(),
            )

            with self.assertRaisesRegex(RuntimeError, "Unlock this user"):
                service.list_threads(user_id="protected_user")

            service.unlock_user(user_id="protected_user", password="atlas-secret")
            threads = service.list_threads(user_id="protected_user")

            self.assertEqual(threads[0]["thread_id"], "main")


class GraphExecutionSequenceTests(unittest.TestCase):
    def test_sequences_match_chat_only_runtime(self) -> None:
        self.assertEqual(
            pre_synthesis_node_sequence(),
            ("retrieve_memories",),
        )
        self.assertEqual(
            post_synthesis_node_sequence(),
            ("extract_updates", "persist"),
        )
        self.assertEqual(
            execution_node_sequence(),
            (
                "retrieve_memories",
                "synthesize_answer",
                "extract_updates",
                "persist",
            ),
        )


class ModelCatalogCachingTests(unittest.TestCase):
    @patch("atlas_local.api_service.inspect_local_ollama_models")
    @patch("atlas_local.api_service.monotonic")
    def test_model_catalog_is_cached_across_calls(self, monotonic_mock, model_info_mock) -> None:
        monotonic_mock.side_effect = [10.0, 11.0]
        model_info_mock.return_value = OllamaCatalogSnapshot(
            models=(OllamaModelInfo(name="qwen", supports_images=True),),
            ollama_online=True,
            has_local_models=True,
            source="ollama",
        )

        service = AtlasBackendService.__new__(AtlasBackendService)
        service.config = SimpleNamespace(chat_model="qwen", chat_temperature=0.2)
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(loaded_models=lambda: ("qwen", "nomic-embed-text:v1.5"))
        )
        service._model_catalog_cache = None

        payload = AtlasBackendService.list_models(service)
        supports_images = AtlasBackendService._model_supports_images(service, "qwen")

        self.assertEqual(payload["models"], ["qwen"])
        self.assertEqual(payload["loaded_models"], ["qwen"])
        self.assertTrue(payload["ollama_online"])
        self.assertTrue(payload["has_local_models"])
        self.assertEqual(payload["catalog_source"], "ollama")
        self.assertTrue(supports_images)
        self.assertEqual(model_info_mock.call_count, 1)

    @patch("atlas_local.api_service.inspect_local_ollama_models")
    def test_model_catalog_reports_reasoning_support_from_catalog(self, model_info_mock) -> None:
        model_info_mock.return_value = OllamaCatalogSnapshot(
            models=(
                OllamaModelInfo(name="plain-code:latest", supports_reasoning=False, reasoning_mode_strategy="none"),
                OllamaModelInfo(name="thinker:latest", supports_reasoning=True, reasoning_mode_strategy="boolean"),
            ),
            ollama_online=True,
            has_local_models=True,
            source="ollama",
        )

        service = AtlasBackendService.__new__(AtlasBackendService)
        service.config = SimpleNamespace(chat_model="thinker:latest", chat_temperature=0.2)
        service._model_catalog_cache = None

        self.assertFalse(AtlasBackendService._model_supports_reasoning(service, "plain-code:latest"))
        self.assertTrue(AtlasBackendService._model_supports_reasoning(service, "thinker:latest"))
        self.assertFalse(AtlasBackendService._model_supports_reasoning(service, "missing:latest"))

    @patch("atlas_local.api_service.inspect_local_ollama_models")
    def test_model_catalog_reports_ollama_context_window_override(self, model_info_mock) -> None:
        model_info_mock.return_value = OllamaCatalogSnapshot(
            models=(
                OllamaModelInfo(name="qwen"),
                OllamaModelInfo(name="gemma"),
            ),
            ollama_online=True,
            has_local_models=True,
            source="ollama",
        )

        service = AtlasBackendService.__new__(AtlasBackendService)
        service.config = SimpleNamespace(chat_model="qwen", chat_temperature=0.2)
        service._model_catalog_cache = None
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                ollama_context_window=lambda: 16384,
                effective_context_window=lambda model: 8192 if model == "gemma" else 16384,
            )
        )

        payload = AtlasBackendService.list_models(service)

        self.assertEqual(payload["ollama_context_window"]["source"], "configured")
        self.assertEqual(payload["ollama_context_window"]["configured_context_window"], 16384)
        self.assertEqual(payload["ollama_context_window"]["effective_context_window"], 16384)

    @patch("atlas_local.api_service.inspect_local_ollama_models")
    def test_model_catalog_reports_ollama_offline_without_local_models(self, model_info_mock) -> None:
        model_info_mock.return_value = OllamaCatalogSnapshot()

        service = AtlasBackendService.__new__(AtlasBackendService)
        service.config = SimpleNamespace(chat_model="qwen", chat_temperature=0.2)
        service._model_catalog_cache = None

        payload = AtlasBackendService.list_models(service)

        self.assertFalse(payload["ollama_online"])
        self.assertFalse(payload["has_local_models"])
        self.assertEqual(payload["catalog_source"], "fallback")
        self.assertEqual(payload["models"], [])

    def test_prepare_ollama_model_switch_unloads_different_chat_models(self) -> None:
        unloaded: list[str] = []
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                loaded_models=lambda: (
                    "qwen3-coder:30b",
                    "nomic-embed-text:v1.5",
                    "qwen3.6:27b",
                    "qwen3-coder:30b",
                ),
                unload_model=lambda model: unloaded.append(model) or {"status": "unloaded", "model": model},
            )
        )
        service._last_chat_model = "gpt-oss:20b"
        service._known_chat_models = {"llama3.2:3b", "qwen3.6:27b", "gpt-oss:20b"}

        result = AtlasBackendService._prepare_ollama_model_switch(service, "qwen3.6:27b")

        self.assertEqual(result, ["qwen3-coder:30b", "gpt-oss:20b", "llama3.2:3b"])
        self.assertEqual(unloaded, result)

    def test_prepare_ollama_model_switch_ignores_unload_failures(self) -> None:
        unloaded: list[str] = []

        def unload_model(model: str) -> dict[str, str]:
            if model == "stuck-model:latest":
                raise RuntimeError("Ollama refused unload")
            unloaded.append(model)
            return {"status": "unloaded", "model": model}

        service = AtlasBackendService.__new__(AtlasBackendService)
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                loaded_models=lambda: ("stuck-model:latest", "old-model:latest"),
                unload_model=unload_model,
            )
        )
        service._last_chat_model = ""
        service._known_chat_models = set()

        result = AtlasBackendService._prepare_ollama_model_switch(service, "target-model:latest")

        self.assertEqual(result, ["old-model:latest"])
        self.assertEqual(unloaded, ["old-model:latest"])

    def test_remember_ollama_chat_model_tracks_last_and_known_models(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)

        AtlasBackendService._remember_ollama_chat_model(service, " qwen3.6:27b ")
        AtlasBackendService._remember_ollama_chat_model(service, "gpt-oss:20b")

        self.assertEqual(service._last_chat_model, "gpt-oss:20b")
        self.assertEqual(service._known_chat_models, {"qwen3.6:27b", "gpt-oss:20b"})

    def test_cleanup_ollama_model_after_resource_failure_unloads_failed_model(self) -> None:
        unloaded: list[str] = []
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                unload_model=lambda model: unloaded.append(model) or {"status": "unloaded", "model": model}
            )
        )

        cleaned = AtlasBackendService._cleanup_ollama_model_after_resource_failure(
            service,
            "qwen3.6:27b",
            "Ollama request failed. Original error: CUDA error: out of memory",
        )

        self.assertTrue(cleaned)
        self.assertEqual(unloaded, ["qwen3.6:27b"])

    def test_cleanup_ollama_model_after_resource_failure_ignores_unrelated_errors(self) -> None:
        unloaded: list[str] = []
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                unload_model=lambda model: unloaded.append(model) or {"status": "unloaded", "model": model}
            )
        )

        cleaned = AtlasBackendService._cleanup_ollama_model_after_resource_failure(
            service,
            "qwen3.6:27b",
            "Model returned an empty response.",
        )

        self.assertFalse(cleaned)
        self.assertEqual(unloaded, [])


class ThreadTemperatureResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AtlasBackendService.__new__(AtlasBackendService)
        self.service.config = SimpleNamespace(chat_model="qwen", chat_temperature=None)
        self.service.run_store = SimpleNamespace(get_thread=lambda **_: None)

    def test_new_thread_without_requested_temperature_uses_model_default(self) -> None:
        self.assertIsNone(
            AtlasBackendService._resolve_thread_temperature(
                self.service,
                user_id="research_user",
                thread_id="main",
                requested_temperature=None,
            )
        )

    def test_existing_thread_can_lock_to_model_default(self) -> None:
        self.service.run_store = SimpleNamespace(
            get_thread=lambda **_: {"thread_id": "main", "last_run_id": "run-1", "temperature": None}
        )

        self.assertIsNone(
            AtlasBackendService._resolve_thread_temperature(
                self.service,
                user_id="research_user",
                thread_id="main",
                requested_temperature=None,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "locked to the selected model's temperature setting"):
            AtlasBackendService._resolve_thread_temperature(
                self.service,
                user_id="research_user",
                thread_id="main",
                requested_temperature=0.6,
            )

    def test_existing_thread_preserves_numeric_temperature(self) -> None:
        self.service.run_store = SimpleNamespace(
            get_thread=lambda **_: {"thread_id": "main", "last_run_id": "run-1", "temperature": 0.6}
        )

        self.assertEqual(
            AtlasBackendService._resolve_thread_temperature(
                self.service,
                user_id="research_user",
                thread_id="main",
                requested_temperature=None,
            ),
            0.6,
        )

    def test_legacy_thread_without_temperature_field_falls_back_to_config_default(self) -> None:
        self.service.run_store = SimpleNamespace(get_thread=lambda **_: {"thread_id": "main", "last_run_id": "run-1"})

        self.assertIsNone(
            AtlasBackendService._resolve_thread_temperature(
                self.service,
                user_id="research_user",
                thread_id="main",
                requested_temperature=None,
            )
        )


class ResetUserTests(unittest.TestCase):
    def test_reset_user_clears_memories_and_resets_threads(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.app = SimpleNamespace(memory_service=SimpleNamespace(delete_all=lambda **_: None))
        service.run_store = SimpleNamespace(
            list_threads=lambda **_: [{"thread_id": "main"}, {"thread_id": "notes"}],
            delete_user=lambda user_id: None,
        )
        reset_thread_calls: list[tuple[str, str | None]] = []
        delete_all_calls: list[str] = []
        deleted_users: list[str] = []

        def delete_all(*, user_id: str) -> None:
            delete_all_calls.append(user_id)

        def delete_user(user_id: str) -> None:
            deleted_users.append(user_id)

        def reset_thread(*, thread_id: str, user_id: str | None = None) -> dict[str, object]:
            reset_thread_calls.append((thread_id, user_id))
            return {"status": "ok"}

        service.app.memory_service = SimpleNamespace(delete_all=delete_all)
        service.run_store = SimpleNamespace(
            list_threads=lambda **_: [{"thread_id": "main"}, {"thread_id": "notes"}],
            delete_user=delete_user,
        )
        service.reset_thread = reset_thread

        result = AtlasBackendService.reset_user(
            service,
            user_id="research_user",
            confirmation_user_id="research_user",
        )

        self.assertEqual(result, {"status": "ok", "user_id": "research_user"})
        self.assertEqual(delete_all_calls, ["research_user"])
        self.assertEqual(deleted_users, ["research_user"])
        self.assertCountEqual(
            reset_thread_calls,
            [("main", "research_user"), ("notes", "research_user")],
        )

    def test_reset_user_reports_memory_delete_failure(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        deleted_users: list[str] = []
        reset_thread_calls: list[str] = []

        def delete_all(*, user_id: str) -> None:
            raise RuntimeError("memory delete failed")

        def delete_user(user_id: str) -> None:
            deleted_users.append(user_id)

        def reset_thread(*, thread_id: str, user_id: str | None = None) -> dict[str, object]:
            reset_thread_calls.append(thread_id)
            return {"status": "ok"}

        service.app = SimpleNamespace(memory_service=SimpleNamespace(delete_all=delete_all))
        service.run_store = SimpleNamespace(
            list_threads=lambda **_: [{"thread_id": "main"}],
            delete_user=delete_user,
        )
        service.reset_thread = reset_thread

        with self.assertRaisesRegex(RuntimeError, "memory delete failed"):
            AtlasBackendService.reset_user(
                service,
                user_id="research_user",
                confirmation_user_id="research_user",
            )

        self.assertEqual(deleted_users, [])
        self.assertEqual(reset_thread_calls, [])


class ContextCompactionTests(unittest.TestCase):
    def test_compaction_uses_full_uncompacted_history_budget(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        summarized_batches: list[list[HumanMessage | AIMessage]] = []

        def summarize_message_batch(
            *,
            model: str,
            existing_summary: str,
            messages: list[HumanMessage | AIMessage],
            target_words: int | None = None,
        ) -> str:
            summarized_batches.append(messages)
            return "summary"

        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                count_message_tokens=lambda _model, messages: len(messages) * 700,
            )
        )
        service._summarize_message_batch = summarize_message_batch
        state = {
            "messages": [
                HumanMessage(content="u1"),
                AIMessage(content="a1"),
                HumanMessage(content="u2"),
                AIMessage(content="a2"),
            ],
            "thread_summary": "",
            "compacted_message_count": 0,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                auto_compact_long_chats=True,
                effective_context_window=1024,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._maybe_compact_context(service, state=state, runtime=runtime)

        self.assertEqual(result["detected_context_window"], 1024)
        self.assertGreater(result["compacted_message_count"], 0)
        self.assertEqual(result["thread_summary"], "summary")
        self.assertEqual(len(summarized_batches), 1)

        before_tokens = service._count_thread_representation_tokens(
            model="gemma4:e4b",
            messages=state["messages"],
            thread_summary="",
            compacted_message_count=0,
        )
        after_tokens = service._count_thread_representation_tokens(
            model="gemma4:e4b",
            messages=state["messages"],
            thread_summary=result["thread_summary"],
            compacted_message_count=result["compacted_message_count"],
        )
        self.assertLess(after_tokens, before_tokens)

    def test_summarize_message_batch_disables_reasoning(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        captured: dict[str, object] = {}

        class _FakeChat:
            def invoke(self, messages: list[HumanMessage]) -> SimpleNamespace:
                captured["prompt"] = str(messages[0].content)
                return SimpleNamespace(content="summary output")

        def fake_chat(model: str, temperature: float | None = None, reasoning: bool | str | None = None):
            captured["model"] = model
            captured["temperature"] = temperature
            captured["reasoning"] = reasoning
            return _FakeChat()

        service.app = SimpleNamespace(llm_provider=SimpleNamespace(chat=fake_chat))

        summary = AtlasBackendService._summarize_message_batch(
            service,
            model="gpt-oss:20b",
            existing_summary="",
            messages=[HumanMessage(content="u1"), AIMessage(content="a1")],
        )

        self.assertEqual(summary, "summary output")
        self.assertEqual(captured["model"], "gpt-oss:20b")
        self.assertEqual(captured["temperature"], 0.0)
        self.assertIs(captured["reasoning"], False)
        prompt = str(captured["prompt"])
        self.assertIn("preserves exact details", prompt)
        self.assertIn("Canon details", prompt)
        self.assertIn("repository names, file paths, function names, commands", prompt)
        self.assertIn("Current objective", prompt)
        self.assertIn("Do not replace specific details with vague phrases", prompt)

    def test_summarize_message_batch_preserves_code_references_in_transcript(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        captured: dict[str, object] = {}

        class _FakeChat:
            def invoke(self, messages: list[HumanMessage]) -> SimpleNamespace:
                captured["prompt"] = str(messages[0].content)
                return SimpleNamespace(content="summary output")

        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(chat=lambda *_args, **_kwargs: _FakeChat())
        )

        AtlasBackendService._summarize_message_batch(
            service,
            model="gpt-oss:20b",
            existing_summary="Existing exact path: apps/atlas/src/pages/WorkspacePage.tsx",
            messages=[
                HumanMessage(
                    content=(
                        "Keep command `cargo check --manifest-path apps/atlas/src-tauri/Cargo.toml` "
                        "and error `ImportError: libtk8.6.so`."
                    )
                ),
                AIMessage(content="Patch file src/atlas_local/code_runner.py and tag v1.0.25."),
            ],
        )

        prompt = str(captured["prompt"])
        self.assertIn("apps/atlas/src/pages/WorkspacePage.tsx", prompt)
        self.assertIn("cargo check --manifest-path apps/atlas/src-tauri/Cargo.toml", prompt)
        self.assertIn("ImportError: libtk8.6.so", prompt)
        self.assertIn("src/atlas_local/code_runner.py", prompt)
        self.assertIn("v1.0.25", prompt)

    def test_summarize_message_batch_appends_missing_pinned_exact_details(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)

        class _FakeChat:
            def invoke(self, messages: list[HumanMessage]) -> SimpleNamespace:
                return SimpleNamespace(content="Current objective: keep working.")

        service.app = SimpleNamespace(llm_provider=SimpleNamespace(chat=lambda *_args, **_kwargs: _FakeChat()))

        summary = AtlasBackendService._summarize_message_batch(
            service,
            model="gpt-oss:20b",
            existing_summary="",
            messages=[
                HumanMessage(content="Patch apps/atlas/src/pages/WorkspacePage.tsx for gpt-oss:20b."),
                AIMessage(content="Release tag v1.0.33 passed with 147 passed, 19 subtests passed."),
            ],
        )

        self.assertIn("Pinned exact details", summary)
        self.assertIn("apps/atlas/src/pages/WorkspacePage.tsx", summary)
        self.assertIn("gpt-oss:20b", summary)
        self.assertIn("v1.0.33", summary)
        self.assertIn("147 passed", summary)

    def test_summarize_message_batch_times_out_to_fallback_and_aborts_request(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        aborted: list[str] = []

        class _SlowChat:
            def invoke(self, messages: list[HumanMessage]) -> SimpleNamespace:
                time.sleep(0.05)
                return SimpleNamespace(content="late summary")

        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                chat=lambda *_args, **_kwargs: _SlowChat(),
                abort_active_requests=lambda: aborted.append("abort"),
            )
        )
        service._compaction_timeout_seconds = lambda: 0.01  # type: ignore[method-assign]

        summary = AtlasBackendService._summarize_message_batch(
            service,
            model="gpt-oss:20b",
            existing_summary="",
            messages=[HumanMessage(content="Use src/atlas_local/api_service.py exactly.")],
        )

        self.assertEqual(aborted, ["abort"])
        self.assertIn("Recent exact details", summary)
        self.assertIn("src/atlas_local/api_service.py", summary)

    def test_auto_compaction_rejects_non_reducing_summary(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)

        def count_tokens(_model: str, messages: list[HumanMessage | AIMessage]) -> int:
            total = 0
            for message in messages:
                content = str(message.content)
                total += 10000 if content.startswith("Conversation summary") else len(content) // 4 + 8
            return total

        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                count_message_tokens=count_tokens,
            )
        )
        service._summarize_message_batch = lambda **_: "bloated summary " * 1000
        state = {
            "messages": [
                HumanMessage(content="u1" * 1000),
                AIMessage(content="a1" * 1000),
                HumanMessage(content="u2" * 1000),
                AIMessage(content="a2" * 1000),
            ],
            "thread_summary": "",
            "compacted_message_count": 0,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                auto_compact_long_chats=True,
                effective_context_window=1024,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._maybe_compact_context(service, state=state, runtime=runtime)

        self.assertEqual(result["thread_summary"], "")
        self.assertEqual(result["compacted_message_count"], 0)

    def test_auto_compaction_accepts_shallow_reduction_when_context_budget_is_exhausted(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)

        def count_tokens(_model: str, messages: list[HumanMessage | AIMessage]) -> int:
            total = 0
            for message in messages:
                content = str(message.content)
                total += 4970 if content.startswith("Conversation summary") else 1000
            return total

        service.app = SimpleNamespace(llm_provider=SimpleNamespace(count_message_tokens=count_tokens))
        service._summarize_message_batch = lambda **_: "small saving summary"  # type: ignore[method-assign]
        state = {
            "messages": [
                HumanMessage(content="u1"),
                AIMessage(content="a1"),
                HumanMessage(content="u2"),
                AIMessage(content="a2"),
                HumanMessage(content="u3"),
                AIMessage(content="a3"),
            ],
            "thread_summary": "",
            "compacted_message_count": 0,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                auto_compact_long_chats=True,
                effective_context_window=8192,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._maybe_compact_context(service, state=state, runtime=runtime)

        self.assertEqual(result["thread_summary"], "small saving summary")
        self.assertEqual(result["compacted_message_count"], 5)

    def test_auto_compaction_tightens_summary_when_summary_pushes_context_over_budget(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)

        def count_tokens(_model: str, messages: list[HumanMessage | AIMessage]) -> int:
            total = 0
            for message in messages:
                content = str(message.content)
                if "long summary" in content:
                    total += 6000
                elif "tight summary" in content:
                    total += 300
                else:
                    total += 100
            return total

        service.app = SimpleNamespace(llm_provider=SimpleNamespace(count_message_tokens=count_tokens))
        service._tighten_thread_summary = lambda **_: "tight summary"  # type: ignore[method-assign]
        state = {
            "messages": [
                HumanMessage(content="already compacted"),
                AIMessage(content="already compacted answer"),
                HumanMessage(content="latest"),
            ],
            "thread_summary": "long summary",
            "compacted_message_count": 2,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                auto_compact_long_chats=True,
                effective_context_window=8192,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._maybe_compact_context(service, state=state, runtime=runtime)

        self.assertEqual(result["thread_summary"], "tight summary")
        self.assertEqual(result["compacted_message_count"], 2)

    def test_auto_compaction_targets_headroom_and_keeps_current_prompt_raw(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        summarized_batches: list[list[HumanMessage | AIMessage]] = []

        def count_tokens(_model: str, messages: list[HumanMessage | AIMessage]) -> int:
            return sum(len(str(message.content)) // 4 + 8 for message in messages)

        def summarize_message_batch(
            *,
            model: str,
            existing_summary: str,
            messages: list[HumanMessage | AIMessage],
            target_words: int | None = None,
        ) -> str:
            summarized_batches.append(messages)
            return "tight summary"

        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(count_message_tokens=count_tokens)
        )
        service._summarize_message_batch = summarize_message_batch
        state = {
            "messages": [
                HumanMessage(content="Please build the Flask example."),
                AIMessage(content="Here is a long generated answer.\n" + ("code block\n" * 1600)),
                HumanMessage(content="Now make it a single file."),
            ],
            "thread_summary": "",
            "compacted_message_count": 0,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                auto_compact_long_chats=True,
                effective_context_window=4096,
                chat_model="gpt-oss:20b",
            )
        )

        result = AtlasBackendService._maybe_compact_context(service, state=state, runtime=runtime)

        self.assertEqual(result["thread_summary"], "tight summary")
        self.assertEqual(result["compacted_message_count"], 2)
        self.assertEqual([message.content for message in summarized_batches[0]], [
            "Please build the Flask example.",
            "Here is a long generated answer.\n" + ("code block\n" * 1600),
        ])
        after_tokens = service._count_thread_representation_tokens(
            model="gpt-oss:20b",
            messages=state["messages"],
            thread_summary=result["thread_summary"],
            compacted_message_count=result["compacted_message_count"],
        )
        self.assertLess(after_tokens, int(max(1024, 4096 * 0.72) * 0.55))

    def test_execute_run_persists_auto_compaction_before_synthesis_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            graph_updates: list[dict[str, object]] = []

            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(
                    llm_provider=SimpleNamespace(
                        abort_active_requests=lambda: None,
                        effective_context_window=lambda _model: 4096,
                        count_message_tokens=lambda _model, messages: sum(
                            len(str(message.content)) // 4 + 8 for message in messages
                        ),
                    ),
                    graph=SimpleNamespace(
                        update_state=lambda _config, payload, as_node=None: graph_updates.append(payload)
                    ),
                    close=lambda: None,
                ),
                run_store=store,
                run_hub=RunHub(),
            )
            service._run_graph_node = lambda **_: None  # type: ignore[method-assign]
            service._summarize_message_batch = lambda **_: "tight summary"  # type: ignore[method-assign]
            service._stream_answer = lambda **_: (_ for _ in ()).throw(  # type: ignore[method-assign]
                RuntimeError("CUDA error: operation not permitted")
            )
            service._get_snapshot = lambda **_: SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="u1" * 2000),
                        AIMessage(content="a1" * 2000),
                        HumanMessage(content="u2" * 2000),
                        AIMessage(content="a2" * 2000),
                    ],
                    "thread_summary": "",
                    "compacted_message_count": 0,
                    "timeline_events": [],
                }
            )
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="continue",
                status="running",
            )

            service._execute_run(
                run_id=artifact["run_id"],
                prompt="continue",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                reasoning_mode=None,
                cross_chat_memory=False,
                auto_compact_long_chats=True,
                attachments=[],
            )

            stored = store.get_run(artifact["run_id"])
            self.assertEqual(stored["status"], "failed")
            self.assertIn("CUDA error", stored["error"])
            self.assertTrue(graph_updates)
            self.assertEqual(graph_updates[0]["thread_summary"], "tight summary")
            self.assertEqual(graph_updates[0]["compacted_message_count"], 4)
            self.assertEqual(graph_updates[0]["timeline_events"][0]["compaction_reason"], "auto")

    def test_execute_run_persists_auto_summary_tightening_before_synthesis_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            graph_updates: list[dict[str, object]] = []

            def count_tokens(_model: str, messages: list[HumanMessage | AIMessage]) -> int:
                total = 0
                for message in messages:
                    content = str(message.content)
                    if "long summary" in content:
                        total += 6000
                    elif "tight summary" in content:
                        total += 300
                    else:
                        total += 100
                return total

            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(
                    llm_provider=SimpleNamespace(
                        abort_active_requests=lambda: None,
                        effective_context_window=lambda _model: 8192,
                        count_message_tokens=count_tokens,
                    ),
                    graph=SimpleNamespace(
                        update_state=lambda _config, payload, as_node=None: graph_updates.append(payload)
                    ),
                    close=lambda: None,
                ),
                run_store=store,
                run_hub=RunHub(),
            )
            service._run_graph_node = lambda **_: None  # type: ignore[method-assign]
            service._tighten_thread_summary = lambda **_: "tight summary"  # type: ignore[method-assign]
            service._stream_answer = lambda **_: (_ for _ in ()).throw(  # type: ignore[method-assign]
                RuntimeError("synthesis failed")
            )
            service._get_snapshot = lambda **_: SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="old"),
                        AIMessage(content="old answer"),
                    ],
                    "thread_summary": "long summary",
                    "compacted_message_count": 2,
                    "timeline_events": [],
                }
            )
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="continue",
                status="running",
            )

            service._execute_run(
                run_id=artifact["run_id"],
                prompt="continue",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                reasoning_mode=None,
                cross_chat_memory=False,
                auto_compact_long_chats=True,
                attachments=[],
            )

            self.assertTrue(graph_updates)
            self.assertEqual(graph_updates[0]["thread_summary"], "tight summary")
            self.assertEqual(graph_updates[0]["compacted_message_count"], 2)
            self.assertEqual(graph_updates[0]["timeline_events"][0]["newly_compacted_message_count"], 0)
            self.assertEqual(graph_updates[0]["timeline_events"][0]["compaction_reason"], "auto")

    def test_execute_run_auto_compacts_after_large_answer_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            graph_updates: list[dict[str, object]] = []
            summarized_batches: list[list[str]] = []

            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(
                    llm_provider=SimpleNamespace(
                        abort_active_requests=lambda: None,
                        effective_context_window=lambda _model: 4096,
                        count_message_tokens=lambda _model, messages: sum(
                            len(str(message.content)) // 4 + 8 for message in messages
                        ),
                    ),
                    graph=SimpleNamespace(
                        update_state=lambda _config, payload, as_node=None: graph_updates.append(payload)
                    ),
                    close=lambda: None,
                ),
                run_store=store,
                run_hub=RunHub(),
            )
            service._run_graph_node = lambda **_: None  # type: ignore[method-assign]

            def summarize_message_batch(**kwargs) -> str:
                summarized_batches.append([str(message.content) for message in kwargs["messages"]])
                return "post answer summary"

            service._summarize_message_batch = summarize_message_batch  # type: ignore[method-assign]
            service._stream_answer = lambda **_: "final answer\n" + ("code line\n" * 1000)  # type: ignore[method-assign]
            service._get_snapshot = lambda **_: SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="u1" * 1000),
                        AIMessage(content="a1" * 1000),
                        HumanMessage(content="u2" * 1000),
                        AIMessage(content="a2" * 1000),
                    ],
                    "thread_summary": "",
                    "compacted_message_count": 0,
                    "timeline_events": [],
                }
            )
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="continue",
                status="running",
            )

            service._execute_run(
                run_id=artifact["run_id"],
                prompt="continue",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                reasoning_mode=None,
                cross_chat_memory=False,
                auto_compact_long_chats=True,
                attachments=[],
            )

            stored = store.get_run(artifact["run_id"])
            self.assertEqual(stored["status"], "completed")
            self.assertEqual(summarized_batches, [["u1" * 1000, "a1" * 1000, "u2" * 1000, "a2" * 1000]])
            self.assertEqual(graph_updates[-1]["thread_summary"], "post answer summary")
            self.assertEqual(graph_updates[-1]["compacted_message_count"], 4)
            context_event = next(event for event in stored["events"] if event["type"] == "context_compacted")
            self.assertEqual(context_event["payload"]["compaction_reason"], "auto")
            self.assertEqual(context_event["payload"]["newly_compacted_message_count"], 4)
            self.assertEqual(graph_updates[-1]["timeline_events"][0]["after_message_count"], 6)

    def test_stream_answer_reports_empty_model_response(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)

        class _EmptyChat:
            def stream(self, messages):
                return iter(())

        emitted: list[tuple[str, dict[str, object]]] = []
        service.config = SimpleNamespace()
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(chat=lambda *_args, **_kwargs: _EmptyChat()),
            nodes=SimpleNamespace(answer_prompt_template=""),
        )
        service._raise_if_cancelled = lambda _run_id: None  # type: ignore[method-assign]
        service._emit_event = lambda _run_id, event_type, payload: emitted.append((event_type, payload))  # type: ignore[method-assign]
        runtime = SimpleNamespace(
            context=GraphContext(
                user_id="research_user",
                thread_id="main",
                session_id="research_user__main",
                chat_model="gpt-oss:20b",
                chat_temperature=0.2,
                cross_chat_memory=False,
                effective_context_window=8192,
            )
        )

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            AtlasBackendService._stream_answer(
                service,
                run_id="run-empty",
                state={"messages": [HumanMessage(content="continue")]},
                runtime=runtime,
            )

    def test_get_thread_context_usage_reports_summary_and_raw_breakdown(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service._ensure_user_unlocked = lambda _user_id: None
        service.run_store = SimpleNamespace(get_thread=lambda **_: {"chat_model": "gpt-oss:20b"})
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                effective_context_window=lambda _model: 8192,
                count_message_tokens=lambda _model, messages: len(messages) * 100,
            )
        )
        service._get_snapshot = lambda **_: SimpleNamespace(
            values={
                "messages": [
                    HumanMessage(content="old user"),
                    AIMessage(content="old answer"),
                    HumanMessage(content="recent user"),
                    AIMessage(content="recent answer"),
                ],
                "thread_summary": "- old user asked for exact paths",
                "compacted_message_count": 2,
                "detected_context_window": 4096,
            }
        )

        usage = AtlasBackendService.get_thread_context_usage(
            service,
            user_id="research_user",
            thread_id="main",
        )

        self.assertEqual(usage["context_window"], 8192)
        self.assertEqual(usage["summary_tokens"], 100)
        self.assertEqual(usage["raw_message_tokens"], 200)
        self.assertEqual(usage["representation_tokens"], 300)
        self.assertEqual(usage["compacted_message_count"], 2)
        self.assertEqual(usage["recent_raw_message_count"], 2)
        self.assertEqual(usage["message_count"], 4)
        self.assertEqual(usage["auto_compact_threshold"], 5898)

    def test_execute_run_stops_before_synthesis_if_cancelled_during_auto_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(
                    llm_provider=SimpleNamespace(
                        abort_active_requests=lambda: None,
                        effective_context_window=lambda _model: 1024,
                        count_message_tokens=lambda _model, messages: len(messages) * 700,
                    ),
                    graph=SimpleNamespace(update_state=lambda *_args, **_kwargs: None),
                    close=lambda: None,
                ),
                run_store=store,
                run_hub=RunHub(),
            )

            service._run_graph_node = lambda **_: None  # type: ignore[method-assign]
            service._stream_answer = lambda **_: (_ for _ in ()).throw(AssertionError("synthesis should not start"))  # type: ignore[method-assign]

            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="continue",
                status="running",
            )
            run_id = artifact["run_id"]

            def summarize_message_batch(
                *,
                model: str,
                existing_summary: str,
                messages: list[HumanMessage | AIMessage],
                target_words: int | None = None,
            ) -> str:
                service._cancelled_runs.add(run_id)
                return "summary"

            service._summarize_message_batch = summarize_message_batch  # type: ignore[method-assign]
            service._get_snapshot = lambda **_: SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="u1"),
                        AIMessage(content="a1"),
                        HumanMessage(content="u2"),
                        AIMessage(content="a2"),
                    ],
                    "thread_summary": "",
                    "compacted_message_count": 0,
                    "timeline_events": [],
                }
            )

            service._execute_run(
                run_id=run_id,
                prompt="continue",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                reasoning_mode=None,
                cross_chat_memory=False,
                auto_compact_long_chats=True,
                attachments=[],
            )

            stored = store.get_run(run_id)
            self.assertEqual(stored["status"], "failed")
            self.assertEqual(stored["error"], "Run stopped by user.")
            stage_events = [event for event in stored["events"] if event["type"] == "stage_changed"]
            self.assertIn("compaction", [event["payload"]["stage"] for event in stage_events])
            self.assertNotIn("synthesis", [event["payload"]["stage"] for event in stage_events])

    def test_get_run_includes_compaction_metadata_from_snapshot(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.run_store = SimpleNamespace(
            get_run=lambda run_id: {
                "run_id": run_id,
                "user_id": "research_user",
                "thread_id": "main",
                "status": "completed",
                "started_at": "2026-04-11T10:00:00+00:00",
                "completed_at": "2026-04-11T10:00:04+00:00",
                "answer": "hello world" * 20,
                "events": [
                    {"type": "token", "timestamp": "2026-04-11T10:00:01+00:00", "payload": {"text": "hello"}},
                    {
                        "type": "context_compacted",
                        "timestamp": "2026-04-11T10:00:02+00:00",
                        "payload": {
                            "history_representation_tokens_before_compaction": 1800,
                            "history_representation_tokens_after_compaction": 900,
                        },
                    },
                ],
            }
        )
        service._get_snapshot = lambda **_: SimpleNamespace(
            values={
                "thread_summary": "summary",
                "compacted_message_count": 4,
                "detected_context_window": 4096,
            }
        )

        artifact = AtlasBackendService.get_run(service, "run-123")

        self.assertEqual(artifact["thread_summary"], "summary")
        self.assertEqual(artifact["compacted_message_count"], 4)
        self.assertEqual(artifact["detected_context_window"], 4096)
        self.assertEqual(artifact["diagnostics"]["first_token_latency_ms"], 1000)
        self.assertEqual(artifact["diagnostics"]["total_duration_ms"], 4000)
        self.assertEqual(artifact["diagnostics"]["compaction_gain_tokens_estimate"], 900)

    def test_thread_history_inserts_context_compaction_marker_at_message_boundary(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.run_store = SimpleNamespace(list_runs_for_thread=lambda **_: [])
        service._get_snapshot = lambda **_: SimpleNamespace(
            values={
                "messages": [
                    HumanMessage(content="first question"),
                    AIMessage(content="first answer"),
                    HumanMessage(content="second question"),
                    AIMessage(content="second answer"),
                ],
                "timeline_events": [
                    {
                        "type": "context_compacted",
                        "timestamp": "2026-04-11T00:00:00Z",
                        "run_id": "run-2",
                        "after_message_count": 3,
                        "compacted_message_count": 2,
                        "newly_compacted_message_count": 2,
                        "thread_summary": "- first turn summary",
                        "detected_context_window": 4096,
                        "compaction_reason": "auto",
                        "history_representation_tokens_before_compaction": 1800,
                        "history_representation_tokens_after_compaction": 640,
                    }
                ],
            }
        )

        history = AtlasBackendService.get_thread_history(service, user_id="research_user", thread_id="main")

        self.assertEqual(
            [item["role"] for item in history],
            ["user", "assistant", "user", "system", "assistant"],
        )
        marker = history[3]
        self.assertEqual(marker["kind"], "context_compacted")
        self.assertEqual(marker["run_id"], "run-2")
        self.assertEqual(marker["thread_summary"], "- first turn summary")
        self.assertEqual(marker["compaction_reason"], "auto")
        self.assertEqual(marker["history_representation_tokens_before_compaction"], 1800)
        self.assertEqual(marker["history_representation_tokens_after_compaction"], 640)
        self.assertEqual(
            [item.get("history_index") for item in history if item["role"] != "system"],
            [0, 1, 2, 3],
        )

    def test_thread_history_ignores_run_lifecycle_events_from_run_artifacts(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.run_store = SimpleNamespace(
            list_runs_for_thread=lambda **_: [
                {
                    "run_id": "run-chat",
                    "mode": "chat",
                    "chat_model": "gemma4:e4b",
                    "temperature": 0.2,
                    "history_after_message_count": 1,
                    "events": [
                        {"type": "run_started", "timestamp": "2026-04-11T00:00:00Z", "payload": {}},
                    ],
                },
                {
                    "run_id": "run-restart",
                    "mode": "compact",
                    "chat_model": "gemma4:e4b",
                    "temperature": 0.0,
                    "history_after_message_count": 2,
                    "events": [
                        {
                            "type": "run_failed",
                            "timestamp": "2026-04-11T00:00:01Z",
                            "payload": {"error": "Atlas backend restarted while this run was active."},
                        }
                    ],
                },
            ]
        )
        service._get_snapshot = lambda **_: SimpleNamespace(
            values={
                "messages": [
                    HumanMessage(content="first question"),
                    AIMessage(content="first answer"),
                ],
                "timeline_events": [],
            }
        )

        history = AtlasBackendService.get_thread_history(service, user_id="research_user", thread_id="main")

        self.assertEqual(
            [(item["role"], item.get("kind")) for item in history],
            [
                ("user", None),
                ("assistant", None),
            ],
        )

    def test_thread_history_ignores_legacy_lifecycle_events_stored_in_snapshot(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.run_store = SimpleNamespace(list_runs_for_thread=lambda **_: [])
        service._get_snapshot = lambda **_: SimpleNamespace(
            values={
                "messages": [],
                "timeline_events": [
                    {
                        "type": "run_started",
                        "timestamp": "2026-04-11T00:00:00Z",
                        "run_id": "run-chat",
                        "chat_model": "gemma4:e4b",
                    },
                    {
                        "type": "context_compacted",
                        "timestamp": "2026-04-11T00:00:01Z",
                        "run_id": "run-compact",
                        "after_message_count": 0,
                        "compacted_message_count": 2,
                        "newly_compacted_message_count": 2,
                        "thread_summary": "- summary",
                        "detected_context_window": 4096,
                        "compaction_reason": "auto",
                        "history_representation_tokens_before_compaction": 1800,
                        "history_representation_tokens_after_compaction": 640,
                    },
                ],
            }
        )

        history = AtlasBackendService.get_thread_history(service, user_id="research_user", thread_id="main")

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["kind"], "context_compacted")
        self.assertEqual(history[0]["run_id"], "run-compact")

    def test_branch_thread_keeps_only_selected_prefix_and_resets_compaction_state(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        graph_updates: list[dict[str, object]] = []
        service.config = SimpleNamespace(chat_model="gemma4:e4b", chat_temperature=0.2)
        service.run_store = SimpleNamespace(
            get_thread=lambda **_: {
                "thread_id": "main",
                "title": "Novel outline",
                "chat_model": "gemma4:e4b",
                "temperature": 0.2,
                "last_mode": "chat",
                "last_prompt": "latest prompt",
            },
            upsert_thread=lambda **kwargs: kwargs,
        )
        service.app = SimpleNamespace(
            graph=SimpleNamespace(update_state=lambda _config, payload, as_node=None: graph_updates.append(payload)),
        )
        service._ensure_user_unlocked = lambda _user_id: None
        service._get_snapshot = lambda **_: SimpleNamespace(
            values={
                "messages": [
                    HumanMessage(content="u1"),
                    AIMessage(content="a1"),
                    HumanMessage(content="u2"),
                    AIMessage(content="a2"),
                ],
                "thread_summary": "- summary",
                "compacted_message_count": 2,
                "timeline_events": [
                    {
                        "type": "context_compacted",
                        "after_message_count": 3,
                    }
                ],
            }
        )

        branched = AtlasBackendService.branch_thread(
            service,
            user_id="research_user",
            thread_id="main",
            after_message_count=2,
        )

        self.assertEqual(branched["title"], "Novel outline branch")
        self.assertEqual(branched["last_prompt"], "u1")
        self.assertTrue(graph_updates)
        self.assertEqual(len(graph_updates[0]["messages"]), 2)
        self.assertEqual(graph_updates[0]["thread_summary"], "")
        self.assertEqual(graph_updates[0]["compacted_message_count"], 0)
        self.assertEqual(graph_updates[0]["timeline_events"], [])


class SearchTests(unittest.TestCase):
    def test_search_threads_returns_current_and_other_thread_matches(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.run_store = SimpleNamespace(
            list_threads=lambda **_: [
                {
                    "user_id": "research_user",
                    "thread_id": "current",
                    "title": "Novel outline",
                    "chat_model": "gemma4:e4b",
                    "updated_at": "2026-04-11T10:00:00Z",
                    "last_prompt": "let's build the world",
                },
                {
                    "user_id": "research_user",
                    "thread_id": "archive",
                    "title": "Atlantis notes",
                    "chat_model": "gpt-oss:20b",
                    "updated_at": "2026-04-10T09:00:00Z",
                    "last_prompt": "summarize the ruins",
                },
            ]
        )
        snapshots = {
            "current": SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="draft the opening scene"),
                        AIMessage(content="The opening scene begins in silence."),
                    ],
                    "timeline_events": [],
                }
            ),
            "archive": SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="what do we know about Atlantis?"),
                        AIMessage(content="Atlantis is described as a lost island civilization."),
                    ],
                    "timeline_events": [],
                }
            ),
        }
        service._get_snapshot = lambda **kwargs: snapshots[kwargs["thread_id"]]

        payload = AtlasBackendService.search_threads(
            service,
            user_id="research_user",
            query="Atlantis",
            current_thread_id="current",
            limit=5,
        )

        self.assertEqual(payload["query"], "Atlantis")
        self.assertEqual(payload["current_thread_results"], [])
        self.assertEqual(payload["other_thread_results"][0]["thread_id"], "archive")
        self.assertEqual(payload["other_thread_results"][0]["match_type"], "thread")
        self.assertEqual(payload["other_thread_results"][1]["history_index"], 1)

    def test_search_threads_returns_message_match_in_current_thread(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.run_store = SimpleNamespace(
            list_threads=lambda **_: [
                {
                    "user_id": "research_user",
                    "thread_id": "main",
                    "title": "Novel outline",
                    "chat_model": "gemma4:e4b",
                    "updated_at": "2026-04-11T10:00:00Z",
                    "last_prompt": "draft the opening",
                },
            ]
        )
        service._get_snapshot = lambda **_: SimpleNamespace(
            values={
                "messages": [
                    HumanMessage(content="draft the opening scene"),
                    AIMessage(content="The opening scene begins in silence."),
                ],
                "timeline_events": [],
            }
        )

        payload = AtlasBackendService.search_threads(
            service,
            user_id="research_user",
            query="opening",
            current_thread_id="main",
            limit=5,
        )

        self.assertEqual(len(payload["current_thread_results"]), 3)
        self.assertEqual(payload["current_thread_results"][0]["match_type"], "thread")
        self.assertEqual(payload["current_thread_results"][1]["history_index"], 1)

    def test_search_threads_uses_message_index_after_compaction_markers(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.run_store = SimpleNamespace(
            list_threads=lambda **_: [
                {
                    "user_id": "research_user",
                    "thread_id": "main",
                    "title": "Compacted notes",
                    "chat_model": "gemma4:e4b",
                    "updated_at": "2026-04-11T10:00:00Z",
                    "last_prompt": "latest prompt",
                },
            ]
        )
        service._get_snapshot = lambda **_: SimpleNamespace(
            values={
                "messages": [
                    HumanMessage(content="first question"),
                    AIMessage(content="first answer"),
                    HumanMessage(content="second question"),
                    AIMessage(content="unique compacted-search answer"),
                ],
                "timeline_events": [
                    {
                        "type": "context_compacted",
                        "timestamp": "2026-04-11T00:00:00Z",
                        "run_id": "run-2",
                        "after_message_count": 2,
                        "compacted_message_count": 2,
                        "newly_compacted_message_count": 2,
                        "thread_summary": "- first turn summary",
                    }
                ],
            }
        )

        payload = AtlasBackendService.search_threads(
            service,
            user_id="research_user",
            query="compacted-search",
            current_thread_id="main",
            limit=5,
        )

        self.assertEqual(payload["current_thread_results"][0]["match_type"], "message")
        self.assertEqual(payload["current_thread_results"][0]["history_index"], 3)

    def test_search_threads_uses_run_search_index_without_loading_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            run = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="archive",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="what do we know about Atlantis?",
                thread_title="Island notes",
                status="queued",
                history_after_message_count=1,
            )
            store.complete_run(run["run_id"], answer="Atlantis is described as a lost island civilization.")
            service = AtlasBackendService.__new__(AtlasBackendService)
            service.run_store = store
            service._get_snapshot = lambda **_: (_ for _ in ()).throw(AssertionError("snapshot loaded"))

            payload = AtlasBackendService.search_threads(
                service,
                user_id="research_user",
                query="lost island",
                current_thread_id="archive",
                limit=5,
            )

        self.assertEqual(payload["current_thread_results"][0]["match_type"], "message")
        self.assertEqual(payload["current_thread_results"][0]["role"], "assistant")
        self.assertEqual(payload["current_thread_results"][0]["history_index"], 1)


class ManualCompactionTests(unittest.TestCase):
    def test_manual_compact_context_summarizes_older_turns(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        summarized_batches: list[list[HumanMessage | AIMessage]] = []

        def summarize_message_batch(
            *,
            model: str,
            existing_summary: str,
            messages: list[HumanMessage | AIMessage],
            target_words: int | None = None,
        ) -> str:
            summarized_batches.append(messages)
            return "manual summary"

        service._summarize_message_batch = summarize_message_batch
        state = {
            "messages": [
                HumanMessage(content="u1" * 600),
                AIMessage(content="a1" * 600),
                HumanMessage(content="u2" * 600),
                AIMessage(content="a2" * 600),
                HumanMessage(content="u3"),
                AIMessage(content="a3"),
            ],
            "thread_summary": "",
            "compacted_message_count": 0,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                effective_context_window=4096,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._manual_compact_context(service, state=state, runtime=runtime)

        self.assertEqual(result["thread_summary"], "manual summary")
        self.assertGreater(result["compacted_message_count"], 0)
        self.assertEqual(result["manual_compaction_status"], "compacted")
        self.assertGreaterEqual(len(summarized_batches), 1)

        before_tokens = _estimate_thread_representation_tokens(
            messages=state["messages"],
            thread_summary="",
            compacted_message_count=0,
        )
        after_tokens = _estimate_thread_representation_tokens(
            messages=state["messages"],
            thread_summary=result["thread_summary"],
            compacted_message_count=result["compacted_message_count"],
        )
        self.assertLess(after_tokens, before_tokens)

    def test_manual_compact_context_retries_wider_batch_when_first_summary_does_not_reduce(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        summarized_batch_sizes: list[int] = []
        target_word_limits: list[int | None] = []

        def summarize_message_batch(
            *,
            model: str,
            existing_summary: str,
            messages: list[HumanMessage | AIMessage],
            target_words: int | None = None,
        ) -> str:
            summarized_batch_sizes.append(len(messages))
            target_word_limits.append(target_words)
            if len(messages) < 4:
                return "oversized manual summary " * 1000
            return "tight summary"

        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                count_message_tokens=lambda _model, messages: sum(
                    len(str(message.content)) // 4 + 8 for message in messages
                ),
            )
        )
        service._summarize_message_batch = summarize_message_batch
        state = {
            "messages": [
                HumanMessage(content="u1" * 1000),
                AIMessage(content="a1" * 1000),
                HumanMessage(content="u2" * 1000),
                AIMessage(content="a2" * 1000),
                HumanMessage(content="latest"),
                AIMessage(content="answer"),
            ],
            "thread_summary": "",
            "compacted_message_count": 0,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                effective_context_window=4096,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._manual_compact_context(service, state=state, runtime=runtime)

        self.assertEqual(summarized_batch_sizes, [2, 3, 4])
        self.assertTrue(all(limit is not None and limit < 520 for limit in target_word_limits))
        self.assertEqual(result["thread_summary"], "tight summary")
        self.assertEqual(result["compacted_message_count"], 4)
        self.assertEqual(result["manual_compaction_status"], "compacted")

    def test_manual_compact_context_keeps_searching_after_shallow_reduction(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        summarized_batch_sizes: list[int] = []

        def summarize_message_batch(
            *,
            model: str,
            existing_summary: str,
            messages: list[HumanMessage | AIMessage],
            target_words: int | None = None,
        ) -> str:
            summarized_batch_sizes.append(len(messages))
            if len(messages) < 4:
                return "medium summary " * 200
            return "tight summary"

        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                count_message_tokens=lambda _model, messages: sum(
                    len(str(message.content)) // 4 + 8 for message in messages
                ),
            )
        )
        service._summarize_message_batch = summarize_message_batch
        state = {
            "messages": [
                HumanMessage(content="u1" * 1000),
                AIMessage(content="a1" * 1000),
                HumanMessage(content="u2" * 2000),
                AIMessage(content="a2" * 2000),
                HumanMessage(content="latest"),
                AIMessage(content="answer"),
            ],
            "thread_summary": "",
            "compacted_message_count": 0,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                effective_context_window=4096,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._manual_compact_context(service, state=state, runtime=runtime)

        self.assertEqual(summarized_batch_sizes, [2, 3, 4])
        self.assertEqual(result["thread_summary"], "tight summary")
        self.assertEqual(result["compacted_message_count"], 4)
        self.assertEqual(result["manual_compaction_status"], "compacted")

    def test_manual_compact_context_tightens_existing_summary_without_raw_history(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                count_message_tokens=lambda _model, messages: sum(
                    len(str(message.content)) // 4 + 8 for message in messages
                ),
            )
        )
        service._tighten_thread_summary = lambda **_: "tight summary"  # type: ignore[method-assign]
        state = {
            "messages": [
                HumanMessage(content="u1"),
                AIMessage(content="a1"),
            ],
            "thread_summary": "word " * 400,
            "compacted_message_count": 2,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                effective_context_window=4096,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._manual_compact_context(service, state=state, runtime=runtime)

        self.assertEqual(result["thread_summary"], "tight summary")
        self.assertEqual(result["compacted_message_count"], 2)
        self.assertEqual(result["manual_compaction_status"], "tightened_summary")

    def test_manual_compact_context_rejects_non_reducing_summary(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)

        def count_tokens(_model: str, messages: list[HumanMessage | AIMessage]) -> int:
            total = 0
            for message in messages:
                content = str(message.content)
                total += 10000 if content.startswith("Conversation summary") else len(content) // 4 + 8
            return total

        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                count_message_tokens=count_tokens,
            )
        )
        service._summarize_message_batch = lambda **_: "oversized manual summary " * 1000
        state = {
            "messages": [
                HumanMessage(content="u1" * 1000),
                AIMessage(content="a1" * 1000),
                HumanMessage(content="u2" * 1000),
                AIMessage(content="a2" * 1000),
                HumanMessage(content="latest"),
                AIMessage(content="answer"),
            ],
            "thread_summary": "",
            "compacted_message_count": 0,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                effective_context_window=4096,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._manual_compact_context(service, state=state, runtime=runtime)

        self.assertEqual(result["thread_summary"], "")
        self.assertEqual(result["compacted_message_count"], 0)
        self.assertEqual(result["manual_compaction_status"], "summary_too_large")

    def test_manual_compact_context_accepts_tiny_reduction(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)

        def count_tokens(_model: str, messages: list[HumanMessage | AIMessage]) -> int:
            total = 0
            for message in messages:
                content = str(message.content)
                total += 3900 if content.startswith("Conversation summary") else 1000
            return total

        service.app = SimpleNamespace(llm_provider=SimpleNamespace(count_message_tokens=count_tokens))
        service._summarize_message_batch = lambda **_: "small saving summary"  # type: ignore[method-assign]
        state = {
            "messages": [
                HumanMessage(content="u1"),
                AIMessage(content="a1"),
                HumanMessage(content="u2"),
                AIMessage(content="a2"),
                HumanMessage(content="u3"),
                AIMessage(content="a3"),
            ],
            "thread_summary": "",
            "compacted_message_count": 0,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                effective_context_window=8192,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._manual_compact_context(service, state=state, runtime=runtime)

        self.assertEqual(result["thread_summary"], "small saving summary")
        self.assertEqual(result["compacted_message_count"], 4)
        self.assertEqual(result["manual_compaction_status"], "compacted")

    def test_execute_compact_run_persists_manual_timeline_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            graph_updates: list[dict[str, object]] = []
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(
                    llm_provider=SimpleNamespace(
                        abort_active_requests=lambda: None,
                        effective_context_window=lambda _model: 4096,
                        count_message_tokens=lambda _model, messages: sum(
                            len(str(message.content)) // 4 + 8 for message in messages
                        ),
                    ),
                    graph=SimpleNamespace(update_state=lambda _config, payload, as_node=None: graph_updates.append(payload)),
                    close=lambda: None,
                ),
                run_store=store,
                run_hub=RunHub(),
            )
            service._summarize_message_batch = lambda **_: "manual summary"  # type: ignore[method-assign]
            service._get_snapshot = lambda **_: SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="u1" * 600),
                        AIMessage(content="a1" * 600),
                        HumanMessage(content="u2" * 600),
                        AIMessage(content="a2" * 600),
                        HumanMessage(content="u3"),
                        AIMessage(content="a3"),
                    ],
                    "thread_summary": "",
                    "compacted_message_count": 0,
                    "timeline_events": [],
                }
            )

            artifact = store.create_run(
                mode="compact",
                user_id="research_user",
                thread_id="main",
                chat_model="gemma4:e4b",
                temperature=0.0,
                prompt="",
                status="running",
            )

            service._execute_compact_run(
                run_id=artifact["run_id"],
                user_id="research_user",
                thread_id="main",
                chat_model="gemma4:e4b",
            )

            stored = store.get_run(artifact["run_id"])
            self.assertEqual(stored["status"], "completed")
            self.assertEqual(stored["events"][-1]["type"], "run_completed")
            context_event = next(event for event in stored["events"] if event["type"] == "context_compacted")
            self.assertEqual(context_event["payload"]["compaction_reason"], "manual")
            self.assertTrue(graph_updates)
            self.assertEqual(graph_updates[-1]["timeline_events"][0]["compaction_reason"], "manual")

    def test_execute_compact_run_persists_tightened_summary_without_new_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            graph_updates: list[dict[str, object]] = []
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(
                    llm_provider=SimpleNamespace(
                        abort_active_requests=lambda: None,
                        effective_context_window=lambda _model: 4096,
                        count_message_tokens=lambda _model, messages: sum(
                            len(str(message.content)) // 4 + 8 for message in messages
                        ),
                    ),
                    graph=SimpleNamespace(update_state=lambda _config, payload, as_node=None: graph_updates.append(payload)),
                    close=lambda: None,
                ),
                run_store=store,
                run_hub=RunHub(),
            )
            service._tighten_thread_summary = lambda **_: "tight summary"  # type: ignore[method-assign]
            service._get_snapshot = lambda **_: SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="u1"),
                        AIMessage(content="a1"),
                    ],
                    "thread_summary": "word " * 400,
                    "compacted_message_count": 2,
                    "timeline_events": [],
                }
            )
            artifact = store.create_run(
                mode="compact",
                user_id="research_user",
                thread_id="main",
                chat_model="gemma4:e4b",
                temperature=0.0,
                prompt="",
                status="running",
            )

            service._execute_compact_run(
                run_id=artifact["run_id"],
                user_id="research_user",
                thread_id="main",
                chat_model="gemma4:e4b",
            )

            stored = store.get_run(artifact["run_id"])
            self.assertEqual(stored["status"], "completed")
            self.assertTrue(graph_updates)
            self.assertEqual(graph_updates[-1]["thread_summary"], "tight summary")
            context_event = next(event for event in stored["events"] if event["type"] == "context_compacted")
            self.assertEqual(context_event["payload"]["newly_compacted_message_count"], 0)

    def test_execute_compact_run_fails_when_no_older_context_can_be_folded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(
                    llm_provider=SimpleNamespace(
                        abort_active_requests=lambda: None,
                        effective_context_window=lambda _model: 4096,
                        count_message_tokens=lambda _model, messages: sum(
                            10000
                            if str(message.content).startswith("Conversation summary")
                            else len(str(message.content)) // 4 + 8
                            for message in messages
                        ),
                    ),
                    graph=SimpleNamespace(update_state=lambda *_args, **_kwargs: None),
                    close=lambda: None,
                ),
                run_store=store,
                run_hub=RunHub(),
            )
            service._get_snapshot = lambda **_: SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="question"),
                        AIMessage(content="answer"),
                    ],
                    "thread_summary": "",
                    "compacted_message_count": 0,
                    "timeline_events": [],
                }
            )

            artifact = store.create_run(
                mode="compact",
                user_id="research_user",
                thread_id="main",
                chat_model="gemma4:e4b",
                temperature=0.0,
                prompt="",
                status="running",
            )

            service._execute_compact_run(
                run_id=artifact["run_id"],
                user_id="research_user",
                thread_id="main",
                chat_model="gemma4:e4b",
            )

            stored = store.get_run(artifact["run_id"])
            self.assertEqual(stored["status"], "failed")
            self.assertIn("no older context to compact", stored["error"])

    def test_execute_compact_run_reports_non_reducing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(
                    llm_provider=SimpleNamespace(
                        abort_active_requests=lambda: None,
                        effective_context_window=lambda _model: 4096,
                        count_message_tokens=lambda _model, messages: sum(
                            10000
                            if str(message.content).startswith("Conversation summary")
                            else len(str(message.content)) // 4 + 8
                            for message in messages
                        ),
                    ),
                    graph=SimpleNamespace(update_state=lambda *_args, **_kwargs: None),
                    close=lambda: None,
                ),
                run_store=store,
                run_hub=RunHub(),
            )
            service._summarize_message_batch = lambda **_: "oversized manual summary " * 1000  # type: ignore[method-assign]
            service._get_snapshot = lambda **_: SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="u1" * 1000),
                        AIMessage(content="a1" * 1000),
                        HumanMessage(content="u2" * 1000),
                        AIMessage(content="a2" * 1000),
                        HumanMessage(content="latest"),
                        AIMessage(content="answer"),
                    ],
                    "thread_summary": "",
                    "compacted_message_count": 0,
                    "timeline_events": [],
                }
            )

            artifact = store.create_run(
                mode="compact",
                user_id="research_user",
                thread_id="main",
                chat_model="gemma4:e4b",
                temperature=0.0,
                prompt="",
                status="running",
            )

            service._execute_compact_run(
                run_id=artifact["run_id"],
                user_id="research_user",
                thread_id="main",
                chat_model="gemma4:e4b",
            )

            stored = store.get_run(artifact["run_id"])
            self.assertEqual(stored["status"], "failed")
            self.assertIn("generated summary would be larger than the messages it replaces", stored["error"])


class QueuedExecutionTests(unittest.TestCase):
    def _make_service(self, temp_dir: str, abort_calls: list[str]) -> AtlasBackendService:
        config = load_config(project_root=Path(temp_dir), env={})
        service = AtlasBackendService(
            config=config,
            app=SimpleNamespace(
                llm_provider=SimpleNamespace(abort_active_requests=lambda: abort_calls.append("abort")),
                graph=SimpleNamespace(get_state=lambda *_args, **_kwargs: SimpleNamespace(values={"messages": []})),
                close=lambda: None,
            ),
            run_store=RunStore(config),
            run_hub=RunHub(),
        )
        service.run_store.create_user("research_user")
        return service

    def test_runs_execute_serially_through_single_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            abort_calls: list[str] = []
            service = self._make_service(temp_dir, abort_calls)
            started: list[str] = []
            first_started = threading.Event()
            second_started = threading.Event()
            release_first = threading.Event()
            release_second = threading.Event()

            def fake_execute_run(**kwargs) -> None:
                run_id = kwargs["run_id"]
                started.append(run_id)
                if len(started) == 1:
                    first_started.set()
                    self.assertTrue(release_first.wait(2.0))
                else:
                    second_started.set()
                    self.assertTrue(release_second.wait(2.0))
                service.run_store.complete_run(run_id, answer=f"done:{run_id}")
                service._emit_event(run_id, "run_completed", {"answer": f"done:{run_id}"})

            service._execute_run = fake_execute_run  # type: ignore[method-assign]
            try:
                first = service.start_chat(
                    prompt="first",
                    user_id="research_user",
                    thread_id="one",
                    chat_model="queue-test-model",
                )
                second = service.start_chat(
                    prompt="second",
                    user_id="research_user",
                    thread_id="two",
                    chat_model="queue-test-model",
                )

                self.assertEqual(first["status"], "queued")
                self.assertEqual(second["status"], "queued")
                self.assertTrue(first_started.wait(1.0))
                time.sleep(0.05)
                self.assertEqual(started, [first["run_id"]])
                self.assertEqual(service.run_store.get_run(first["run_id"])["status"], "running")
                self.assertEqual(service.run_store.get_run(second["run_id"])["status"], "queued")
                self.assertEqual(service.run_store.get_run(second["run_id"])["events"][0]["type"], "run_queued")
                self.assertTrue(service.status()["busy"])

                release_first.set()
                self.assertTrue(second_started.wait(1.0))
                self.assertEqual(started, [first["run_id"], second["run_id"]])

                release_second.set()
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    if service.run_store.get_run(second["run_id"])["status"] == "completed":
                        break
                    time.sleep(0.02)

                self.assertEqual(service.run_store.get_run(first["run_id"])["status"], "completed")
                self.assertEqual(service.run_store.get_run(second["run_id"])["status"], "completed")
                deadline = time.time() + 2.0
                while time.time() < deadline and service.status()["busy"]:
                    time.sleep(0.02)
                self.assertFalse(service.status()["busy"])
                self.assertEqual(abort_calls, [])
            finally:
                service.close()

    def test_worker_fails_run_and_continues_after_unhandled_execution_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            abort_calls: list[str] = []
            service = self._make_service(temp_dir, abort_calls)
            started: list[str] = []
            second_completed = threading.Event()

            def fake_execute_run(**kwargs) -> None:
                run_id = kwargs["run_id"]
                started.append(run_id)
                if len(started) == 1:
                    raise RuntimeError("preflight exploded")
                service.run_store.complete_run(run_id, answer=f"done:{run_id}")
                service._emit_event(run_id, "run_completed", {"answer": f"done:{run_id}"})
                second_completed.set()

            service._execute_run = fake_execute_run  # type: ignore[method-assign]
            try:
                first = service.start_chat(
                    prompt="first",
                    user_id="research_user",
                    thread_id="one",
                    chat_model="queue-test-model",
                )
                second = service.start_chat(
                    prompt="second",
                    user_id="research_user",
                    thread_id="two",
                    chat_model="queue-test-model",
                )

                self.assertTrue(second_completed.wait(2.0))
                first_artifact = service.run_store.get_run(first["run_id"])
                second_artifact = service.run_store.get_run(second["run_id"])

                self.assertEqual(first_artifact["status"], "failed")
                self.assertEqual(first_artifact["error"], "preflight exploded")
                self.assertEqual(first_artifact["events"][-1]["type"], "run_failed")
                self.assertEqual(second_artifact["status"], "completed")
                self.assertEqual(started, [first["run_id"], second["run_id"]])
                deadline = time.time() + 2.0
                while time.time() < deadline and service.status()["busy"]:
                    time.sleep(0.02)
                self.assertFalse(service.status()["busy"])
            finally:
                service.close()

    def test_cancel_run_removes_queued_job_before_it_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            abort_calls: list[str] = []
            service = self._make_service(temp_dir, abort_calls)
            started: list[str] = []
            first_started = threading.Event()
            second_started = threading.Event()
            release_first = threading.Event()

            def fake_execute_run(**kwargs) -> None:
                run_id = kwargs["run_id"]
                started.append(run_id)
                if len(started) == 1:
                    first_started.set()
                    self.assertTrue(release_first.wait(2.0))
                else:
                    second_started.set()
                service.run_store.complete_run(run_id, answer=f"done:{run_id}")
                service._emit_event(run_id, "run_completed", {"answer": f"done:{run_id}"})

            service._execute_run = fake_execute_run  # type: ignore[method-assign]
            try:
                first = service.start_chat(
                    prompt="first",
                    user_id="research_user",
                    thread_id="one",
                    chat_model="queue-test-model",
                )
                second = service.start_chat(
                    prompt="second",
                    user_id="research_user",
                    thread_id="two",
                    chat_model="queue-test-model",
                )

                self.assertTrue(first_started.wait(1.0))
                response = service.cancel_run(second["run_id"])
                self.assertEqual(response["status"], "cancelling")
                cancelled = service.run_store.get_run(second["run_id"])
                self.assertEqual(cancelled["status"], "failed")
                self.assertEqual(cancelled["error"], "Run stopped by user.")
                self.assertFalse(second_started.wait(0.2))

                release_first.set()
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    if service.run_store.get_run(first["run_id"])["status"] == "completed":
                        break
                    time.sleep(0.02)

                self.assertEqual(started, [first["run_id"]])
                self.assertEqual(abort_calls, [])
                self.assertEqual(cancelled["events"][-1]["type"], "run_failed")
            finally:
                service.close()

    def test_cancel_run_marks_running_job_cancelling_and_aborts_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            abort_calls: list[str] = []
            service = self._make_service(temp_dir, abort_calls)
            started = threading.Event()

            def fake_execute_run(**kwargs) -> None:
                run_id = kwargs["run_id"]
                started.set()
                deadline = time.time() + 2.0
                while time.time() < deadline and not service._is_cancelled(run_id):
                    time.sleep(0.01)
                if service._is_cancelled(run_id):
                    service.run_store.fail_run(run_id, error="Run stopped by user.")
                    service._emit_event(run_id, "run_failed", {"error": "Run stopped by user."})
                    return
                service.run_store.complete_run(run_id, answer=f"done:{run_id}")
                service._emit_event(run_id, "run_completed", {"answer": f"done:{run_id}"})

            service._execute_run = fake_execute_run  # type: ignore[method-assign]
            try:
                run = service.start_chat(
                    prompt="first",
                    user_id="research_user",
                    thread_id="one",
                    chat_model="queue-test-model",
                )
                self.assertTrue(started.wait(1.0))
                response = service.cancel_run(run["run_id"])

                self.assertEqual(response["status"], "cancelling")
                self.assertIn(service.run_store.get_run(run["run_id"])["status"], {"cancelling", "failed"})
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    if service.run_store.get_run(run["run_id"])["status"] == "failed":
                        break
                    time.sleep(0.02)
                self.assertEqual(service.run_store.get_run(run["run_id"])["status"], "failed")
                self.assertEqual(abort_calls, ["abort"])
            finally:
                service.close()


if __name__ == "__main__":
    unittest.main()
