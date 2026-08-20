import unittest
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from atlas_local.api_service import (
    AtlasBackendService,
    _CompactionModelUnavailable,
    _INTERRUPTED_RUN_ERROR,
    _auto_compact_threshold,
    _estimate_thread_representation_tokens,
    _render_messages_for_summary,
    _select_messages_for_compaction,
)
from atlas_local.config import load_config
from atlas_local.graph.builder import execution_node_sequence, post_synthesis_node_sequence, pre_synthesis_node_sequence
from atlas_local.graph.context import GraphContext
from atlas_local.graph.nodes import GraphNodes
from atlas_local.llm import OllamaCatalogSnapshot, OllamaModelInfo
from atlas_local.run_contract import RunHub
from atlas_local.run_store import RunStore
from atlas_local.security import (
    application_secret_protection_available,
    local_secret_storage_label,
    open_application_sqlite,
    sqlcipher_enabled,
)
from atlas_local.session import scoped_thread_id


class ApiServiceCreateTests(unittest.TestCase):
    def test_small_context_compaction_threshold_preserves_generation_headroom(
        self,
    ) -> None:
        self.assertEqual(_auto_compact_threshold(1024), int(1024 * 0.72))
        self.assertLess(_auto_compact_threshold(1024), 1024)

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

    def test_close_shuts_down_full_runtime_once(self) -> None:
        calls: list[str] = []
        service = AtlasBackendService(
            config=object(),
            app=SimpleNamespace(
                llm_provider=SimpleNamespace(
                    abort_active_requests=lambda: calls.append("abort-provider")
                ),
                close=lambda: calls.append("app"),
            ),
            run_store=SimpleNamespace(
                flush_pending_events=lambda: calls.append("flush"),
                lock_all_user_keys=lambda: calls.append("lock-keys"),
            ),
            run_hub=RunHub(),
        )
        model_pulls = SimpleNamespace(shutdown=lambda: calls.append("model-pulls"))

        with (
            patch(
                "atlas_local.api_service.get_model_pull_manager",
                return_value=model_pulls,
            ),
            patch(
                "atlas_local.api_service.shutdown_runner",
                side_effect=lambda: calls.append("runner"),
            ),
        ):
            service.close()
            service.close()

        self.assertEqual(
            calls,
            ["model-pulls", "runner", "app", "flush", "lock-keys"],
        )

    def test_begin_shutdown_quiesces_once_without_closing_runtime_resources(self) -> None:
        calls: list[str] = []
        service = AtlasBackendService(
            config=object(),
            app=SimpleNamespace(
                llm_provider=SimpleNamespace(
                    abort_active_requests=lambda: calls.append("abort-provider")
                ),
                close=lambda: calls.append("app"),
            ),
            run_store=SimpleNamespace(
                flush_pending_events=lambda: calls.append("flush"),
                lock_all_user_keys=lambda: calls.append("lock-keys"),
            ),
            run_hub=RunHub(),
        )
        model_pulls = SimpleNamespace(shutdown=lambda: calls.append("model-pulls"))

        with (
            patch(
                "atlas_local.api_service.get_model_pull_manager",
                return_value=model_pulls,
            ),
            patch(
                "atlas_local.api_service.shutdown_runner",
                side_effect=lambda: calls.append("runner"),
            ),
        ):
            service.begin_shutdown()
            service.begin_shutdown()
            self.assertEqual(calls, ["model-pulls", "runner"])

            service.close()

        self.assertEqual(
            calls,
            ["model-pulls", "runner", "app", "flush", "lock-keys"],
        )

    def test_close_leaves_runtime_resources_open_while_worker_is_alive(self) -> None:
        calls: list[str] = []
        release_worker = threading.Event()
        worker = threading.Thread(target=release_worker.wait, daemon=True)
        service = AtlasBackendService(
            config=object(),
            app=SimpleNamespace(
                llm_provider=SimpleNamespace(abort_active_requests=lambda: None),
                close=lambda: calls.append("app"),
            ),
            run_store=SimpleNamespace(
                flush_pending_events=lambda: calls.append("flush"),
                lock_all_user_keys=lambda: calls.append("lock-keys"),
            ),
            run_hub=RunHub(),
        )
        service._worker_thread = worker
        model_pulls = SimpleNamespace(shutdown=lambda: calls.append("model-pulls"))
        worker.start()

        try:
            with (
                patch(
                    "atlas_local.api_service.get_model_pull_manager",
                    return_value=model_pulls,
                ),
                patch(
                    "atlas_local.api_service.shutdown_runner",
                    side_effect=lambda: calls.append("runner"),
                ),
                patch(
                    "atlas_local.api_service.WORKER_SHUTDOWN_TIMEOUT_SECONDS",
                    0.01,
                ),
            ):
                service.close()

            self.assertTrue(worker.is_alive())
            self.assertEqual(calls, ["model-pulls", "runner"])
        finally:
            release_worker.set()
            worker.join(timeout=1.0)

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
            self.assertEqual(
                payload["security"]["sqlite_encrypted_at_rest"],
                application_secret_protection_available() and sqlcipher_enabled(),
            )
            self.assertEqual(payload["security"]["vector_store"], "local-qdrant")
            self.assertEqual(
                payload["security"]["vector_store_encrypted_at_rest"],
                application_secret_protection_available() and sqlcipher_enabled(),
            )


class UserProtectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.secret_storage = patch(
            "atlas_local.run_store.application_secret_protection_available",
            return_value=True,
        )
        self.secret_storage.start()

    def tearDown(self) -> None:
        self.secret_storage.stop()

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

    def test_interrupted_protected_run_is_recovered_after_unlock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            original_store = RunStore(config)
            original_store.create_user("protected_user", password="atlas-secret")
            original_store.unlock_user_key(
                "protected_user",
                password="atlas-secret",
            )
            artifact = original_store.create_run(
                mode="chat",
                user_id="protected_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="hello",
                status="running",
            )
            original_store.lock_all_user_keys()

            restarted_store = RunStore(config)
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(close=lambda: None),
                run_store=restarted_store,
                run_hub=RunHub(),
            )

            service._recover_incomplete_runs()

            self.assertEqual(
                restarted_store._read_index()["runs"][artifact["run_id"]][
                    "status"
                ],
                "running",
            )
            self.assertEqual(
                service._deferred_recovery_runs,
                {"protected_user": {artifact["run_id"]}},
            )

            unlocked = service.unlock_user(
                user_id="protected_user",
                password="atlas-secret",
            )
            recovered = restarted_store.get_run(artifact["run_id"])

            self.assertFalse(unlocked["locked"])
            self.assertEqual(recovered["status"], "failed")
            self.assertEqual(
                recovered["error"],
                "Atlas backend restarted while this run was active.",
            )
            self.assertEqual(recovered["events"][-1]["type"], "run_failed")
            self.assertEqual(service._deferred_recovery_runs, {})

    def test_recovery_quarantines_one_corrupt_run_and_recovers_healthy_runs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            original_store = RunStore(config)
            healthy = original_store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="healthy",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="healthy prompt",
                status="running",
            )
            corrupt = original_store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="corrupt",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="private corrupt prompt",
                status="running",
            )
            original_store._run_path(corrupt["run_id"]).write_text(
                "not valid JSON",
                encoding="utf-8",
            )

            restarted_store = RunStore(config)
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(close=lambda: None),
                run_store=restarted_store,
                run_hub=RunHub(),
            )

            service._recover_incomplete_runs()

            healthy_recovered = restarted_store.get_run(healthy["run_id"])
            corrupt_recovered = restarted_store.get_run(corrupt["run_id"])
            index_runs = restarted_store._read_index()["runs"]
            quarantine_files = list(
                (restarted_store.runs_dir / "quarantine").glob(
                    f"{corrupt['run_id']}.*.json"
                )
            )

            self.assertEqual(healthy_recovered["status"], "failed")
            self.assertEqual(healthy_recovered["error"], _INTERRUPTED_RUN_ERROR)
            self.assertEqual(corrupt_recovered["status"], "failed")
            self.assertIn("was unreadable and was quarantined", corrupt_recovered["error"])
            self.assertEqual(corrupt_recovered["prompt"], "")
            self.assertEqual(len(quarantine_files), 1)
            self.assertEqual(quarantine_files[0].read_text(encoding="utf-8"), "not valid JSON")
            self.assertEqual(index_runs[healthy["run_id"]]["status"], "failed")
            self.assertEqual(index_runs[corrupt["run_id"]]["status"], "failed")

            restarted_store.delete_thread(
                user_id="research_user",
                thread_id="corrupt",
            )
            self.assertFalse(quarantine_files[0].exists())

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

    def test_global_thread_listing_omits_locked_profile_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("public_user")
            store.upsert_thread(
                user_id="public_user",
                thread_id="public",
                title="Public notes",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                last_prompt="safe preview",
            )
            store.create_user("protected_user", password="atlas-secret")
            store.upsert_thread(
                user_id="protected_user",
                thread_id="private",
                title="Private notes",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                last_prompt="locked secret prompt",
            )
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(close=lambda: None),
                run_store=store,
                run_hub=RunHub(),
            )

            locked_view = service.list_threads()
            service.unlock_user(user_id="protected_user", password="atlas-secret")
            unlocked_view = service.list_threads()

            self.assertEqual(
                [(item["user_id"], item["last_prompt"]) for item in locked_view],
                [("public_user", "safe preview")],
            )
            self.assertCountEqual(
                [item["user_id"] for item in unlocked_view],
                ["public_user", "protected_user"],
            )

    def test_lock_refuses_active_profile_run_then_clears_cached_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("protected_user", password="atlas-secret")
            store.unlock_user_key("protected_user", password="atlas-secret")
            artifact = store.create_run(
                mode="chat",
                user_id="protected_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="hello",
            )
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(close=lambda: None),
                run_store=store,
                run_hub=RunHub(),
            )
            service._unlocked_users.add("protected_user")
            service._active_run_id = artifact["run_id"]

            with self.assertRaisesRegex(RuntimeError, "active runs"):
                service.lock_user(user_id="protected_user")
            self.assertTrue(store.is_user_key_unlocked("protected_user"))

            store.complete_run(artifact["run_id"], answer="done")
            locked = service.lock_user(user_id="protected_user")

            self.assertTrue(locked["locked"])
            self.assertFalse(store.is_user_key_unlocked("protected_user"))

    def test_service_close_clears_all_cached_profile_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("protected_user", password="atlas-secret")
            store.unlock_user_key("protected_user", password="atlas-secret")
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(close=lambda: None),
                run_store=store,
                run_hub=RunHub(),
            )
            service._unlocked_users.add("protected_user")

            service.close()

            self.assertEqual(service._unlocked_users, set())
            self.assertFalse(store.is_user_key_unlocked("protected_user"))

    def test_delete_memory_passes_claimed_owner_to_memory_service(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        deleted: list[tuple[str, str]] = []
        service._ensure_user_unlocked = lambda _user_id: None
        service.app = SimpleNamespace(
            memory_service=SimpleNamespace(
                delete=lambda memory_id, *, user_id: deleted.append((memory_id, user_id))
            )
        )

        result = AtlasBackendService.delete_memory(
            service,
            user_id="research_user",
            memory_id="memory-1",
        )

        self.assertEqual(deleted, [("memory-1", "research_user")])
        self.assertEqual(result["memory_id"], "memory-1")

    def test_reset_thread_rejects_thread_owned_by_another_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("owner")
            store.create_user("other")
            store.upsert_thread(
                user_id="owner",
                thread_id="main",
                title="Owner thread",
                chat_model="gpt-oss:20b",
                temperature=0.2,
            )
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(close=lambda: None),
                run_store=store,
                run_hub=RunHub(),
            )

            with self.assertRaisesRegex(RuntimeError, "Thread not found"):
                service.reset_thread(user_id="other", thread_id="main")

    def test_reset_thread_deletes_only_the_owned_checkpoint_only_thread(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("owner")
            store.create_user("other")
            owner_thread_id = scoped_thread_id("owner", "main")
            other_thread_id = scoped_thread_id("other", "main")
            with open_application_sqlite(
                config.langgraph_checkpoint_db,
                data_dir=config.data_dir,
            ) as conn:
                conn.execute("CREATE TABLE checkpoints (thread_id TEXT)")
                conn.execute("CREATE TABLE writes (thread_id TEXT)")
                for runtime_thread_id in (owner_thread_id, other_thread_id):
                    conn.execute(
                        "INSERT INTO checkpoints (thread_id) VALUES (?)",
                        (runtime_thread_id,),
                    )
                    conn.execute(
                        "INSERT INTO writes (thread_id) VALUES (?)",
                        (runtime_thread_id,),
                    )
                conn.commit()
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(close=lambda: None),
                run_store=store,
                run_hub=RunHub(),
            )

            service.reset_thread(user_id="owner", thread_id="main")

            with open_application_sqlite(
                config.langgraph_checkpoint_db,
                data_dir=config.data_dir,
            ) as conn:
                checkpoint_rows = conn.execute(
                    "SELECT thread_id FROM checkpoints"
                ).fetchall()
                write_rows = conn.execute("SELECT thread_id FROM writes").fetchall()
            self.assertEqual(checkpoint_rows, [(other_thread_id,)])
            self.assertEqual(write_rows, [(other_thread_id,)])

    def test_reset_user_deletes_checkpoint_only_threads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("owner")
            runtime_thread_id = scoped_thread_id("owner", "orphan")
            with open_application_sqlite(
                config.langgraph_checkpoint_db,
                data_dir=config.data_dir,
            ) as conn:
                conn.execute("CREATE TABLE checkpoints (thread_id TEXT)")
                conn.execute("CREATE TABLE writes (thread_id TEXT)")
                conn.execute(
                    "INSERT INTO checkpoints (thread_id) VALUES (?)",
                    (runtime_thread_id,),
                )
                conn.execute(
                    "INSERT INTO writes (thread_id) VALUES (?)",
                    (runtime_thread_id,),
                )
                conn.commit()
            deleted_memories: list[str] = []
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(
                    memory_service=SimpleNamespace(
                        delete_all=lambda *, user_id: deleted_memories.append(
                            user_id
                        )
                    ),
                    close=lambda: None,
                ),
                run_store=store,
                run_hub=RunHub(),
            )

            service.reset_user(
                user_id="owner",
                confirmation_user_id="owner",
            )

            with open_application_sqlite(
                config.langgraph_checkpoint_db,
                data_dir=config.data_dir,
            ) as conn:
                checkpoint_count = conn.execute(
                    "SELECT count(*) FROM checkpoints"
                ).fetchone()[0]
                write_count = conn.execute(
                    "SELECT count(*) FROM writes"
                ).fetchone()[0]
            self.assertEqual(checkpoint_count, 0)
            self.assertEqual(write_count, 0)
            self.assertEqual(deleted_memories, ["owner"])
            self.assertIsNone(store.get_user("owner"))


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


class MemoryPersistenceResilienceTests(unittest.TestCase):
    def test_graph_persist_reports_memory_add_failure_without_raising(self) -> None:
        class FailingMemoryService:
            def list(self, **_kwargs):
                return []

            def add(self, *_args, **_kwargs):
                raise RuntimeError("embedding model unavailable")

        nodes = GraphNodes(
            config=SimpleNamespace(prompt_dir=Path("/missing")),
            llm_provider=SimpleNamespace(),
            memory_service=FailingMemoryService(),
        )
        runtime = SimpleNamespace(
            context=GraphContext(
                user_id="research_user",
                thread_id="main",
                session_id="research_user:main",
                chat_model="local-model",
                chat_temperature=0.2,
                cross_chat_memory=True,
            )
        )

        result = nodes.persist(
            {
                "update_candidates": [
                    {
                        "category": "profile",
                        "value": "name: Alice",
                        "confidence": 0.9,
                    }
                ]
            },
            runtime,
        )

        self.assertEqual(result["persisted_memories"], [])
        self.assertEqual(len(result["memory_persistence_warnings"]), 1)
        self.assertEqual(
            result["memory_persistence_warnings"][0]["category"],
            "profile",
        )
        self.assertIn(
            "embedding model unavailable",
            result["memory_persistence_warnings"][0]["error"],
        )

    def test_execute_run_completes_and_checkpoints_when_memory_add_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            graph_updates: list[dict[str, object]] = []

            class FailingMemoryService:
                def search(self, *_args, **_kwargs):
                    return []

                def list(self, **_kwargs):
                    return []

                def add(self, *_args, **_kwargs):
                    raise RuntimeError("local vector store unavailable")

            provider = SimpleNamespace(
                abort_active_requests=lambda: None,
                effective_context_window=lambda _model: 8192,
                count_message_tokens=lambda _model, messages: sum(
                    len(str(message.content)) // 4 + 8 for message in messages
                ),
            )
            nodes = GraphNodes(
                config=config,
                llm_provider=provider,
                memory_service=FailingMemoryService(),
            )
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(
                    llm_provider=provider,
                    nodes=nodes,
                    graph=SimpleNamespace(
                        update_state=lambda _config, payload, as_node=None: graph_updates.append(
                            payload
                        )
                    ),
                    close=lambda: None,
                ),
                run_store=store,
                run_hub=RunHub(),
            )
            service._stream_answer = lambda **_: "The answer is preserved."  # type: ignore[method-assign]
            service._get_snapshot = lambda **_: SimpleNamespace(  # type: ignore[method-assign]
                values={
                    "messages": [],
                    "thread_summary": "",
                    "compacted_message_count": 0,
                    "timeline_events": [],
                }
            )
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="local-model",
                temperature=0.2,
                prompt="My name is Alice.",
                status="running",
            )

            service._execute_run(
                run_id=artifact["run_id"],
                prompt="My name is Alice.",
                user_id="research_user",
                thread_id="main",
                chat_model="local-model",
                temperature=0.2,
                reasoning_mode=None,
                cross_chat_memory=True,
                auto_compact_long_chats=True,
                attachments=[],
            )

            stored = store.get_run(artifact["run_id"])
            self.assertEqual(stored["status"], "completed")
            self.assertEqual(stored["answer"], "The answer is preserved.")
            self.assertEqual(stored["history_after_message_count"], 1)
            warning_event = next(
                event for event in stored["events"] if event["type"] == "run_warning"
            )
            self.assertEqual(
                warning_event["payload"]["stage"],
                "memory_persistence",
            )
            memory_trace = next(
                item
                for item in stored["trace_items"]
                if item["stage"] == "memory persistence"
            )
            self.assertEqual(memory_trace["outputs"]["stored"], 0)
            self.assertEqual(memory_trace["outputs"]["warning_count"], 1)
            self.assertEqual(len(graph_updates), 1)
            checkpoint_messages = graph_updates[0]["messages"]
            self.assertEqual(
                [str(message.content) for message in checkpoint_messages],
                ["My name is Alice.", "The answer is preserved."],
            )


class ModelCatalogCachingTests(unittest.TestCase):
    def test_runtime_model_mutations_are_blocked_during_active_or_queued_runs(self) -> None:
        calls: list[tuple[str, object]] = []
        service = AtlasBackendService.__new__(AtlasBackendService)
        service._control_lock = threading.RLock()
        service._active_run_id = "run-active"
        service._pending_runs = []
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                set_ollama_context_window=lambda value: calls.append(("context", value)),
                unload_model=lambda model: calls.append(("unload", model)) or {},
            )
        )

        with self.assertRaisesRegex(RuntimeError, "active and queued runs"):
            service.set_ollama_context_window(context_window=8192)
        service._active_run_id = None
        service._pending_runs.append(SimpleNamespace(run_id="run-queued"))
        with self.assertRaisesRegex(RuntimeError, "active and queued runs"):
            service.unload_ollama_model(model="qwen:latest")

        self.assertEqual(calls, [])

    @patch("atlas_local.api_service.inspect_local_models")
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

    @patch("atlas_local.api_service.inspect_local_models")
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

    @patch("atlas_local.api_service.inspect_local_models")
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

    @patch("atlas_local.api_service.inspect_local_models")
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
        service._reset_thread_data = reset_thread
        service._list_checkpoint_threads = lambda: []
        service._purge_legacy_migration_backups = lambda: None

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
        service._reset_thread_data = reset_thread
        service._list_checkpoint_threads = lambda: []

        with self.assertRaisesRegex(RuntimeError, "memory delete failed"):
            AtlasBackendService.reset_user(
                service,
                user_id="research_user",
                confirmation_user_id="research_user",
            )

        self.assertEqual(deleted_users, [])
        self.assertEqual(reset_thread_calls, [])

    def test_reset_user_purges_legacy_plaintext_migration_backups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            backup_dir = data_dir / "migration-backups"
            backup_dir.mkdir(parents=True)
            (backup_dir / "legacy.plaintext.sqlite").write_text(
                "sensitive",
                encoding="utf-8",
            )
            service = AtlasBackendService.__new__(AtlasBackendService)
            service.config = SimpleNamespace(
                data_dir=data_dir,
                langgraph_checkpoint_db=data_dir / "checkpoints.sqlite",
                mem0_history_db=data_dir / "memory.sqlite",
                qdrant_path=data_dir / "qdrant",
            )
            service.app = SimpleNamespace(
                memory_service=SimpleNamespace(delete_all=lambda **_: None)
            )
            service.run_store = SimpleNamespace(
                list_threads=lambda **_: [],
                delete_user=lambda _user_id: None,
                lock_user_key=lambda _user_id: None,
            )
            service._unlocked_users = {"research_user"}
            service._ensure_user_unlocked = lambda _user_id: None
            service._ensure_profile_has_no_active_runs = lambda *_args, **_kwargs: None

            result = AtlasBackendService.reset_user(
                service,
                user_id="research_user",
                confirmation_user_id="research_user",
            )

            self.assertEqual(result, {"status": "ok", "user_id": "research_user"})
            self.assertFalse(backup_dir.exists())

    def test_reset_all_purges_legacy_plaintext_migration_backups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            backup_dir = data_dir / "migration-backups"
            backup_dir.mkdir(parents=True)
            (backup_dir / "legacy.plaintext.qdrant").write_text(
                "sensitive",
                encoding="utf-8",
            )
            connection = SimpleNamespace(
                execute=lambda *_args, **_kwargs: None,
                commit=lambda: None,
                close=lambda: None,
            )
            reset_calls: list[str] = []
            service = AtlasBackendService.__new__(AtlasBackendService)
            service.config = SimpleNamespace(
                data_dir=data_dir,
                langgraph_checkpoint_db=data_dir / "checkpoints.sqlite",
                mem0_history_db=data_dir / "memory.sqlite",
                qdrant_path=data_dir / "qdrant",
            )
            service.app = SimpleNamespace(
                memory_service=SimpleNamespace(
                    reset=lambda: reset_calls.append("memory")
                )
            )
            service.run_store = SimpleNamespace(
                reset_all=lambda: reset_calls.append("runs")
            )

            with patch(
                "atlas_local.api_service.open_application_sqlite",
                return_value=connection,
            ):
                result = AtlasBackendService.reset_all(
                    service,
                    confirmation="RESET ATLAS",
                )

            self.assertEqual(result, {"status": "ok"})
            self.assertEqual(reset_calls, ["memory", "runs"])
            self.assertFalse(backup_dir.exists())


class MutationBarrierTests(unittest.TestCase):
    def _make_service(
        self,
        temp_dir: str,
        *,
        memory_service: object,
    ) -> AtlasBackendService:
        config = load_config(project_root=Path(temp_dir), env={})
        service = AtlasBackendService(
            config=config,
            app=SimpleNamespace(memory_service=memory_service, close=lambda: None),
            run_store=RunStore(config),
            run_hub=RunHub(),
        )
        service.run_store.create_user("research_user")
        service._ensure_worker_started = lambda: None  # type: ignore[method-assign]
        service._thread_history_after_message_count = (  # type: ignore[method-assign]
            lambda **_kwargs: 0
        )
        service._resolve_thread_model = (  # type: ignore[method-assign]
            lambda **_kwargs: "queue-test-model"
        )
        service._resolve_thread_temperature = (  # type: ignore[method-assign]
            lambda **_kwargs: 0.2
        )
        service._resolve_thread_title = (  # type: ignore[method-assign]
            lambda **_kwargs: "Main"
        )
        return service

    @staticmethod
    def _start_test_run(service: AtlasBackendService) -> dict[str, object]:
        return service._start_run(
            mode="chat",
            prompt="hello",
            user_id="research_user",
            thread_id="main",
            chat_model="queue-test-model",
            temperature=0.2,
            reasoning_mode="off",
            thread_title="Main",
            cross_chat_memory=True,
            auto_compact_long_chats=True,
            attachments=[],
        )

    def test_profile_delete_blocks_run_enqueue_for_entire_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mutation_started = threading.Event()
            release_mutation = threading.Event()
            reset_errors: list[BaseException] = []

            def delete_all(*, user_id: str) -> None:
                self.assertEqual(user_id, "research_user")
                mutation_started.set()
                if not release_mutation.wait(5.0):
                    raise AssertionError("Timed out waiting to release profile deletion.")

            service = self._make_service(
                temp_dir,
                memory_service=SimpleNamespace(delete_all=delete_all),
            )

            def reset_profile() -> None:
                try:
                    service.reset_user(
                        user_id="research_user",
                        confirmation_user_id="research_user",
                    )
                except BaseException as exc:
                    reset_errors.append(exc)

            reset_thread = threading.Thread(target=reset_profile)
            reset_thread.start()
            try:
                self.assertTrue(mutation_started.wait(2.0))
                with self.assertRaisesRegex(RuntimeError, "profile is being modified"):
                    self._start_test_run(service)
                with self.assertRaisesRegex(RuntimeError, "profile is being modified"):
                    service.add_memory(
                        user_id="research_user",
                        text="late memory",
                    )
                self.assertEqual(service.run_store._read_index()["runs"], {})
            finally:
                release_mutation.set()
                reset_thread.join(5.0)

            self.assertFalse(reset_thread.is_alive())
            self.assertEqual(reset_errors, [])
            self.assertIsNone(service.run_store.get_user("research_user"))

    def test_in_flight_profile_write_prevents_delete_overtaking_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_started = threading.Event()
            release_write = threading.Event()
            persisted_memories: list[tuple[str, str]] = []
            write_errors: list[BaseException] = []

            def add_memory(record, *, user_id: str, metadata: dict[str, str]):
                write_started.set()
                if not release_write.wait(5.0):
                    raise AssertionError("Timed out waiting to release memory write.")
                persisted_memories.append((user_id, record.text))
                return {"results": [{"id": "memory-1"}]}

            service = self._make_service(
                temp_dir,
                memory_service=SimpleNamespace(add=add_memory),
            )

            def write_memory() -> None:
                try:
                    service.add_memory(
                        user_id="research_user",
                        text="remember this",
                    )
                except BaseException as exc:
                    write_errors.append(exc)

            write_thread = threading.Thread(target=write_memory)
            write_thread.start()
            try:
                self.assertTrue(write_started.wait(2.0))
                with self.assertRaisesRegex(RuntimeError, "active operations"):
                    service.reset_user(
                        user_id="research_user",
                        confirmation_user_id="research_user",
                    )
                self.assertIsNotNone(
                    service.run_store.get_user("research_user"),
                )
            finally:
                release_write.set()
                write_thread.join(5.0)

            self.assertFalse(write_thread.is_alive())
            self.assertEqual(write_errors, [])
            self.assertEqual(
                persisted_memories,
                [("research_user", "remember this")],
            )
            self.assertEqual(service._active_profile_operations, {})

    def test_start_run_rechecks_user_after_delete_during_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._make_service(
                temp_dir,
                memory_service=SimpleNamespace(delete_all=lambda **_kwargs: None),
            )
            preflight_started = threading.Event()
            release_preflight = threading.Event()
            run_errors: list[BaseException] = []

            def resolve_title(**_kwargs) -> str:
                preflight_started.set()
                if not release_preflight.wait(5.0):
                    raise AssertionError("Timed out waiting to release run preflight.")
                return "Main"

            service._resolve_thread_title = resolve_title  # type: ignore[method-assign]

            def start_run() -> None:
                try:
                    self._start_test_run(service)
                except BaseException as exc:
                    run_errors.append(exc)

            run_thread = threading.Thread(target=start_run)
            run_thread.start()
            try:
                self.assertTrue(preflight_started.wait(2.0))
                service.reset_user(
                    user_id="research_user",
                    confirmation_user_id="research_user",
                )
            finally:
                release_preflight.set()
                run_thread.join(5.0)

            self.assertFalse(run_thread.is_alive())
            self.assertEqual(len(run_errors), 1)
            self.assertIsInstance(run_errors[0], RuntimeError)
            self.assertIn("User not found: research_user", str(run_errors[0]))
            self.assertEqual(service.run_store._read_index()["runs"], {})
            self.assertIsNone(service.run_store.get_user("research_user"))

    def test_reset_all_rejects_queued_and_active_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._make_service(
                temp_dir,
                memory_service=SimpleNamespace(reset=lambda: None),
            )
            service._pending_runs.append(SimpleNamespace(user_id="research_user"))

            with self.assertRaisesRegex(RuntimeError, "all active runs"):
                service.reset_all(confirmation="RESET ATLAS")

            service._pending_runs.clear()
            artifact = service.run_store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="queue-test-model",
                temperature=0.2,
                prompt="hello",
                status="running",
            )
            service._active_run_id = artifact["run_id"]

            with self.assertRaisesRegex(RuntimeError, "all active runs"):
                service.reset_all(confirmation="RESET ATLAS")

    def test_global_reset_blocks_run_and_profile_creation_until_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reset_started = threading.Event()
            release_reset = threading.Event()
            reset_errors: list[BaseException] = []

            def reset_memory() -> None:
                reset_started.set()
                if not release_reset.wait(5.0):
                    raise AssertionError("Timed out waiting to release global reset.")

            service = self._make_service(
                temp_dir,
                memory_service=SimpleNamespace(reset=reset_memory),
            )
            with open_application_sqlite(
                service.config.langgraph_checkpoint_db,
                data_dir=service.config.data_dir,
            ) as conn:
                conn.execute("CREATE TABLE checkpoints (thread_id TEXT)")
                conn.execute("CREATE TABLE writes (thread_id TEXT)")
                conn.commit()

            def reset_all() -> None:
                try:
                    service.reset_all(confirmation="RESET ATLAS")
                except BaseException as exc:
                    reset_errors.append(exc)

            reset_thread = threading.Thread(target=reset_all)
            reset_thread.start()
            try:
                self.assertTrue(reset_started.wait(2.0))
                with self.assertRaisesRegex(RuntimeError, "reset is in progress"):
                    self._start_test_run(service)
                with self.assertRaisesRegex(RuntimeError, "reset is in progress"):
                    service.create_user(user_id="new_user")
                self.assertIsNone(service.run_store.get_user("new_user"))
                self.assertEqual(service.run_store._read_index()["runs"], {})
            finally:
                release_reset.set()
                reset_thread.join(5.0)

            self.assertFalse(reset_thread.is_alive())
            self.assertEqual(reset_errors, [])
            self.assertEqual(service.run_store.list_users(), [])

    def test_unscoped_thread_read_prevents_profile_lock_overtaking_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("protected_user", password="atlas-secret")
            store.upsert_thread(
                user_id="protected_user",
                thread_id="main",
                title="Private thread",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                last_prompt="private prompt",
            )
            service = AtlasBackendService(
                config=config,
                app=SimpleNamespace(close=lambda: None),
                run_store=store,
                run_hub=RunHub(),
            )
            service.unlock_user(
                user_id="protected_user",
                password="atlas-secret",
            )
            read_started = threading.Event()
            release_read = threading.Event()
            read_errors: list[BaseException] = []
            read_results: list[list[dict[str, object]]] = []
            original_list_threads = store.list_threads
            list_calls = 0

            def slow_list_threads(*, user_id=None):
                nonlocal list_calls
                result = original_list_threads(user_id=user_id)
                list_calls += 1
                if list_calls == 2:
                    read_started.set()
                    if not release_read.wait(5.0):
                        raise AssertionError("Timed out waiting to release thread read.")
                return result

            store.list_threads = slow_list_threads  # type: ignore[method-assign]

            def read_threads() -> None:
                try:
                    read_results.append(service.list_threads())
                except BaseException as exc:
                    read_errors.append(exc)

            reader = threading.Thread(target=read_threads)
            reader.start()
            try:
                self.assertTrue(read_started.wait(2.0))
                with self.assertRaisesRegex(RuntimeError, "active operations"):
                    service.lock_user(user_id="protected_user")
            finally:
                release_read.set()
                reader.join(5.0)

            self.assertFalse(reader.is_alive())
            self.assertEqual(read_errors, [])
            self.assertEqual(read_results[0][0]["last_prompt"], "private prompt")
            self.assertEqual(service._active_profile_operations, {})
            self.assertTrue(service.lock_user(user_id="protected_user")["locked"])

    def test_memory_read_prevents_profile_delete_overtaking_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            read_started = threading.Event()
            release_read = threading.Event()
            read_errors: list[BaseException] = []
            read_results: list[list[dict[str, object]]] = []

            service = self._make_service(
                temp_dir,
                memory_service=SimpleNamespace(delete_all=lambda **_kwargs: None),
            )

            def list_memories(*, user_id: str, limit: int):
                self.assertEqual(user_id, "research_user")
                self.assertEqual(limit, 50)
                read_started.set()
                if not release_read.wait(5.0):
                    raise AssertionError("Timed out waiting to release memory read.")
                return [SimpleNamespace(id="memory-1", text="private memory")]

            service.app.list_memories = list_memories

            def read_memories() -> None:
                try:
                    read_results.append(
                        service.list_memories(user_id="research_user")
                    )
                except BaseException as exc:
                    read_errors.append(exc)

            reader = threading.Thread(target=read_memories)
            reader.start()
            try:
                self.assertTrue(read_started.wait(2.0))
                with self.assertRaisesRegex(RuntimeError, "active operations"):
                    service.reset_user(
                        user_id="research_user",
                        confirmation_user_id="research_user",
                    )
                self.assertIsNotNone(service.run_store.get_user("research_user"))
            finally:
                release_read.set()
                reader.join(5.0)

            self.assertFalse(reader.is_alive())
            self.assertEqual(read_errors, [])
            self.assertEqual(read_results, [[{"id": "memory-1", "text": "private memory"}]])
            self.assertEqual(service._active_profile_operations, {})
            service.reset_user(
                user_id="research_user",
                confirmation_user_id="research_user",
            )
            self.assertIsNone(service.run_store.get_user("research_user"))

    def test_stream_subscription_holds_profile_read_gate_until_unsubscribe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._make_service(
                temp_dir,
                memory_service=SimpleNamespace(delete_all=lambda **_kwargs: None),
            )
            run = service.run_store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="queue-test-model",
                temperature=0.2,
                prompt="private prompt",
            )
            service.run_store.complete_run(run["run_id"], answer="private answer")

            subscriber = service.subscribe(run["run_id"])
            self.assertEqual(
                service._active_profile_operations,
                {"research_user": 1},
            )
            with self.assertRaisesRegex(RuntimeError, "active operations"):
                service.lock_user(user_id="research_user")

            service.unsubscribe(run["run_id"], subscriber)

            self.assertEqual(service._active_profile_operations, {})
            self.assertEqual(service._stream_profile_operations, {})
            service.lock_user(user_id="research_user")


class ContextCompactionTests(unittest.TestCase):
    def test_compaction_selector_never_overflows_or_splits_a_turn(self) -> None:
        first_turn = [
            HumanMessage(content="small request"),
            AIMessage(content="small response"),
        ]
        oversized_turn = [
            HumanMessage(content="next request"),
            AIMessage(content="x" * 500),
        ]
        first_turn_chars = len(_render_messages_for_summary(first_turn))

        selected, consumed = _select_messages_for_compaction(
            [*first_turn, *oversized_turn],
            max_chars=first_turn_chars,
        )

        self.assertEqual(selected, first_turn)
        self.assertEqual(consumed, 2)
        self.assertLessEqual(
            len(_render_messages_for_summary(selected)),
            first_turn_chars,
        )

        selected, consumed = _select_messages_for_compaction(
            oversized_turn,
            max_chars=100,
        )

        self.assertEqual(selected, [])
        self.assertEqual(consumed, 0)

    def test_auto_compaction_skips_oversized_oldest_turn_without_model_call(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        summary_calls: list[list[HumanMessage | AIMessage]] = []
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                count_message_tokens=lambda _model, messages: len(messages) * 1000,
            )
        )
        service._summarize_message_batch = (  # type: ignore[method-assign]
            lambda **kwargs: summary_calls.append(kwargs["messages"]) or "must not be used"
        )
        state = {
            "messages": [
                HumanMessage(content="x" * 7000),
                AIMessage(content="old answer"),
                HumanMessage(content="latest"),
                AIMessage(content="latest answer"),
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

        result = AtlasBackendService._maybe_compact_context(
            service,
            state=state,
            runtime=runtime,
        )

        self.assertEqual(summary_calls, [])
        self.assertEqual(result["thread_summary"], "")
        self.assertEqual(result["compacted_message_count"], 0)

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

    def test_summarize_message_batch_times_out_without_lossy_fallback(self) -> None:
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

        with self.assertRaisesRegex(
            _CompactionModelUnavailable,
            "timed out",
        ):
            AtlasBackendService._summarize_message_batch(
                service,
                model="gpt-oss:20b",
                existing_summary="",
                messages=[HumanMessage(content="Use src/atlas_local/api_service.py exactly.")],
            )

        self.assertEqual(aborted, ["abort"])

    def test_compaction_model_wait_stops_promptly_when_cancelled(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        invoked = threading.Event()
        cancelled = threading.Event()
        release_model = threading.Event()
        aborted: list[str] = []

        class _BlockingChat:
            def invoke(self, messages: list[HumanMessage]) -> SimpleNamespace:
                invoked.set()
                release_model.wait(1.0)
                return SimpleNamespace(content="late summary")

        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                chat=lambda *_args, **_kwargs: _BlockingChat(),
                abort_active_requests=lambda: aborted.append("abort"),
            )
        )
        service._compaction_timeout_seconds = lambda: 5.0  # type: ignore[method-assign]

        def cancel_after_invoke() -> None:
            invoked.wait(1.0)
            cancelled.set()

        canceller = threading.Thread(target=cancel_after_invoke, daemon=True)
        canceller.start()
        started_at = time.monotonic()
        try:
            with self.assertRaisesRegex(RuntimeError, "Run stopped by user"):
                AtlasBackendService._invoke_compaction_model(
                    service,
                    model="gpt-oss:20b",
                    prompt="compact",
                    cancel_check=cancelled.is_set,
                )
        finally:
            release_model.set()
            canceller.join(timeout=1.0)

        self.assertLess(time.monotonic() - started_at, 0.5)
        self.assertEqual(aborted, ["abort"])

    def test_auto_compaction_reuses_known_representation_token_count(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        count_calls: list[int] = []
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                count_message_tokens=lambda _model, messages: count_calls.append(len(messages)) or 100,
            )
        )
        state = {
            "messages": [HumanMessage(content="latest")],
            "thread_summary": "",
            "compacted_message_count": 0,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                auto_compact_long_chats=True,
                effective_context_window=4096,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._maybe_compact_context(
            service,
            state=state,
            runtime=runtime,
            current_representation_tokens=100,
        )

        self.assertEqual(result["detected_context_window"], 4096)
        self.assertEqual(count_calls, [])

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
                AIMessage(content="Here is a long generated answer.\n" + ("code block\n" * 1100)),
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
            "Here is a long generated answer.\n" + ("code block\n" * 1100),
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
            self.assertEqual(graph_updates[0]["compacted_message_count"], 2)
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

    def test_execute_run_limits_auto_compaction_to_one_model_call_across_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            graph_updates: list[dict[str, object]] = []
            summarized_batches: list[int] = []

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
                summarized_batches.append(len(kwargs["messages"]))
                return "incremental summary"

            service._summarize_message_batch = summarize_message_batch  # type: ignore[method-assign]
            service._stream_answer = lambda **_: "large answer " * 3000  # type: ignore[method-assign]
            service._get_snapshot = lambda **_: SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="u1" * 2800),
                        AIMessage(content="a1" * 2800),
                        HumanMessage(content="u2" * 2800),
                        AIMessage(content="a2" * 2800),
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
            self.assertEqual(summarized_batches, [2])
            self.assertEqual(
                sum(event["type"] == "context_compacted" for event in stored["events"]),
                1,
            )
            self.assertEqual(graph_updates[-1]["thread_summary"], "incremental summary")
            self.assertEqual(graph_updates[-1]["compacted_message_count"], 2)

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

        self.assertEqual(emitted, [])

    def test_stream_answer_marks_generation_once_before_first_output(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)

        class _ThinkingThenAnswerChat:
            def stream(self, messages):
                del messages
                yield AIMessage(
                    content="",
                    additional_kwargs={"reasoning_content": "checking"},
                )
                yield AIMessage(content="done")

        emitted: list[tuple[str, dict[str, object]]] = []
        service.config = SimpleNamespace()
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(chat=lambda *_args, **_kwargs: _ThinkingThenAnswerChat()),
            nodes=SimpleNamespace(answer_prompt_template=""),
        )
        service._raise_if_cancelled = lambda _run_id: None  # type: ignore[method-assign]
        service._emit_event = lambda _run_id, event_type, payload: emitted.append(  # type: ignore[method-assign]
            (event_type, payload)
        )
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

        answer = AtlasBackendService._stream_answer(
            service,
            run_id="run-progress",
            state={"messages": [HumanMessage(content="continue")]},
            runtime=runtime,
        )

        self.assertEqual(answer, "done")
        self.assertEqual(
            emitted,
            [
                ("stage_changed", {"stage": "generation"}),
                ("thinking_token", {"text": "checking"}),
                ("token", {"text": "done"}),
            ],
        )

    def test_stream_answer_aborts_a_provider_that_exceeds_the_total_size_limit(
        self,
    ) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)

        class _OversizedChat:
            def stream(self, messages):
                del messages
                yield AIMessage(content="12345")
                yield AIMessage(content="67890")

        aborted: list[bool] = []
        emitted: list[tuple[str, dict[str, object]]] = []
        service.config = SimpleNamespace(
            chat_provider="ollama",
            chat_base_url="http://127.0.0.1:11434",
            ollama_url="http://127.0.0.1:11434",
            embed_model="embed",
        )
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                chat=lambda *_args, **_kwargs: _OversizedChat(),
                abort_active_requests=lambda: aborted.append(True),
            ),
            nodes=SimpleNamespace(answer_prompt_template=""),
        )
        service._raise_if_cancelled = lambda _run_id: None  # type: ignore[method-assign]
        service._is_cancelled = lambda _run_id: False  # type: ignore[method-assign]
        service._emit_event = lambda _run_id, event_type, payload: emitted.append(  # type: ignore[method-assign]
            (event_type, payload)
        )
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

        with (
            patch("atlas_local.api_service.MAX_MODEL_STREAM_CHARS", 8),
            self.assertRaisesRegex(RuntimeError, "safe size limit"),
        ):
            AtlasBackendService._stream_answer(
                service,
                run_id="run-oversized",
                state={"messages": [HumanMessage(content="continue")]},
                runtime=runtime,
            )

        self.assertEqual(aborted, [True])
        self.assertEqual(
            emitted,
            [
                ("stage_changed", {"stage": "generation"}),
                ("token", {"text": "12345"}),
            ],
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

    def test_thread_detail_reads_check_profile_access_before_loading_state(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        state_reads: list[dict[str, str]] = []
        run_reads: list[dict[str, str]] = []

        def reject_locked_user(_user_id: str) -> None:
            raise RuntimeError("Unlock this user before continuing.")

        service._ensure_user_unlocked = reject_locked_user
        service._get_snapshot = lambda **kwargs: state_reads.append(kwargs)  # type: ignore[method-assign]
        service.run_store = SimpleNamespace(
            list_runs_for_thread=lambda **kwargs: run_reads.append(kwargs)
        )

        calls = (
            lambda: AtlasBackendService.get_thread_history(
                service,
                user_id="protected_user",
                thread_id="atlas-thread-v2:forged",
            ),
            lambda: AtlasBackendService.get_thread_context_usage(
                service,
                user_id="protected_user",
                thread_id="atlas-thread-v2:forged",
            ),
            lambda: AtlasBackendService.list_thread_runs(
                service,
                user_id="protected_user",
                thread_id="main",
            ),
        )

        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(RuntimeError, "Unlock this user"):
                    call()

        self.assertEqual(state_reads, [])
        self.assertEqual(run_reads, [])

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
                cancel_check=None,
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
            self.assertNotIn("model_loading", [event["payload"]["stage"] for event in stage_events])

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
    def test_manual_compact_reports_oversized_oldest_turn_without_model_call(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        summary_calls: list[list[HumanMessage | AIMessage]] = []
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                count_message_tokens=lambda _model, messages: len(messages) * 1000,
            )
        )
        service._summarize_message_batch = (  # type: ignore[method-assign]
            lambda **kwargs: summary_calls.append(kwargs["messages"]) or "must not be used"
        )
        state = {
            "messages": [
                HumanMessage(content="x" * 7000),
                AIMessage(content="old answer"),
                HumanMessage(content="latest"),
                AIMessage(content="latest answer"),
            ],
            "thread_summary": "",
            "compacted_message_count": 0,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                effective_context_window=1024,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._manual_compact_context(
            service,
            state=state,
            runtime=runtime,
        )

        self.assertEqual(summary_calls, [])
        self.assertEqual(result["manual_compaction_status"], "input_too_large")
        self.assertEqual(result["thread_summary"], "")
        self.assertEqual(result["compacted_message_count"], 0)

    def test_manual_compact_keeps_state_when_summary_is_unavailable(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                count_message_tokens=lambda _model, messages: len(messages) * 1000,
            )
        )
        service._summarize_message_batch = lambda **_: (_ for _ in ()).throw(  # type: ignore[method-assign]
            _CompactionModelUnavailable("provider unavailable")
        )
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
                effective_context_window=4096,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._manual_compact_context(
            service,
            state=state,
            runtime=runtime,
        )

        self.assertEqual(result["manual_compaction_status"], "summary_unavailable")
        self.assertEqual(result["thread_summary"], "")
        self.assertEqual(result["compacted_message_count"], 0)

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

    def test_manual_compact_context_starts_with_largest_safe_batch(self) -> None:
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

        self.assertEqual(summarized_batch_sizes, [4])
        self.assertTrue(all(limit is not None and limit < 520 for limit in target_word_limits))
        self.assertEqual(result["thread_summary"], "tight summary")
        self.assertEqual(result["compacted_message_count"], 4)
        self.assertEqual(result["manual_compaction_status"], "compacted")

    def test_manual_compact_context_uses_one_model_call_for_long_thread(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        summarized_batch_sizes: list[int] = []
        service.app = SimpleNamespace(
            llm_provider=SimpleNamespace(
                count_message_tokens=lambda _model, messages: sum(
                    len(str(message.content)) // 4 + 8 for message in messages
                ),
            )
        )

        def summarize_message_batch(
            *,
            model: str,
            existing_summary: str,
            messages: list[HumanMessage | AIMessage],
            target_words: int | None = None,
        ) -> str:
            summarized_batch_sizes.append(len(messages))
            return "tight summary"

        service._summarize_message_batch = summarize_message_batch
        messages: list[HumanMessage | AIMessage] = []
        for index in range(10):
            messages.extend(
                [
                    HumanMessage(content=f"user {index} " * 100),
                    AIMessage(content=f"assistant {index} " * 100),
                ]
            )
        state = {
            "messages": messages,
            "thread_summary": "",
            "compacted_message_count": 0,
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                effective_context_window=8192,
                chat_model="gemma4:e4b",
            )
        )

        result = AtlasBackendService._manual_compact_context(
            service,
            state=state,
            runtime=runtime,
        )

        self.assertEqual(summarized_batch_sizes, [18])
        self.assertEqual(result["compacted_message_count"], 18)
        self.assertEqual(result["manual_compaction_status"], "compacted")

    def test_manual_compact_context_does_not_repeat_model_call_after_shallow_reduction(self) -> None:
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

        self.assertEqual(summarized_batch_sizes, [4])
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

    def test_execute_compact_run_does_not_persist_result_after_cancellation(self) -> None:
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
                        count_message_tokens=lambda _model, messages: len(messages) * 500,
                    ),
                    graph=SimpleNamespace(
                        update_state=lambda _config, payload, as_node=None: graph_updates.append(payload)
                    ),
                    close=lambda: None,
                ),
                run_store=store,
                run_hub=RunHub(),
            )
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
            artifact = store.create_run(
                mode="compact",
                user_id="research_user",
                thread_id="main",
                chat_model="gemma4:e4b",
                temperature=0.0,
                prompt="",
                status="running",
            )
            run_id = artifact["run_id"]

            def cancel_during_compaction(**_kwargs) -> dict[str, object]:
                service._cancelled_runs.add(run_id)
                return {
                    "thread_summary": "must not persist",
                    "compacted_message_count": 2,
                    "detected_context_window": 4096,
                    "manual_compaction_status": "compacted",
                }

            service._manual_compact_context = cancel_during_compaction  # type: ignore[method-assign]
            service._execute_compact_run(
                run_id=run_id,
                user_id="research_user",
                thread_id="main",
                chat_model="gemma4:e4b",
            )

            stored = store.get_run(run_id)
            self.assertEqual(stored["status"], "failed")
            self.assertEqual(stored["error"], "Run stopped by user.")
            self.assertFalse(any(event["type"] == "context_compacted" for event in stored["events"]))
            self.assertEqual(graph_updates, [])

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

    def test_cancel_run_is_idempotent_and_ignores_terminal_cleanup_race(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            abort_calls: list[str] = []
            service = self._make_service(temp_dir, abort_calls)
            service._ensure_worker_started = lambda: None  # type: ignore[method-assign]
            service._get_accessible_run = lambda run_id: service.run_store.get_run(run_id)  # type: ignore[method-assign]

            running = service.run_store.create_run(
                mode="compact",
                user_id="research_user",
                thread_id="main",
                chat_model="queue-test-model",
                temperature=0.0,
                prompt="",
                status="running",
            )
            service._active_run_id = running["run_id"]

            first = service.cancel_run(running["run_id"])
            second = service.cancel_run(running["run_id"])

            self.assertEqual(first["status"], "cancelling")
            self.assertEqual(second["status"], "cancelling")
            self.assertEqual(abort_calls, ["abort"])
            running_events = service.run_store.get_run(running["run_id"])["events"]
            self.assertEqual(
                [
                    event["payload"]["stage"]
                    for event in running_events
                    if event["type"] == "stage_changed"
                ],
                ["stopping"],
            )

            terminal = service.run_store.create_run(
                mode="compact",
                user_id="research_user",
                thread_id="done",
                chat_model="queue-test-model",
                temperature=0.0,
                prompt="",
                status="running",
            )
            service.run_store.complete_run(terminal["run_id"], answer="")
            service._active_run_id = terminal["run_id"]

            response = service.cancel_run(terminal["run_id"])

            self.assertEqual(response["status"], "completed")
            self.assertEqual(abort_calls, ["abort"])
            terminal_events = service.run_store.get_run(terminal["run_id"])["events"]
            self.assertFalse(
                any(
                    event["type"] == "stage_changed"
                    and event["payload"].get("stage") == "stopping"
                    for event in terminal_events
                )
            )

    def test_cancel_run_does_not_emit_stopping_after_concurrent_completion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            abort_calls: list[str] = []
            service = self._make_service(temp_dir, abort_calls)
            service._ensure_worker_started = lambda: None  # type: ignore[method-assign]
            service._get_accessible_run = (  # type: ignore[method-assign]
                lambda run_id: service.run_store.get_run(run_id)
            )
            running = service.run_store.create_run(
                mode="compact",
                user_id="research_user",
                thread_id="race",
                chat_model="queue-test-model",
                temperature=0.0,
                prompt="",
                status="running",
            )
            run_id = running["run_id"]
            service._active_run_id = run_id
            original_claim = service.run_store.mark_run_cancelling_with_event

            def complete_before_claim(candidate_run_id):
                service.run_store.complete_run(candidate_run_id, answer="won race")
                return original_claim(candidate_run_id)

            service.run_store.mark_run_cancelling_with_event = complete_before_claim  # type: ignore[method-assign]

            response = service.cancel_run(run_id)

            persisted = service.run_store.get_run(run_id)
            self.assertEqual(response["status"], "completed")
            self.assertEqual(persisted["status"], "completed")
            self.assertEqual(abort_calls, [])
            self.assertEqual(
                [event["type"] for event in persisted["events"]],
                ["run_completed"],
            )

    def test_cancel_run_does_not_abort_next_run_after_active_handoff(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            abort_calls: list[str] = []
            service = self._make_service(temp_dir, abort_calls)
            service._ensure_worker_started = lambda: None  # type: ignore[method-assign]
            service._get_accessible_run = (  # type: ignore[method-assign]
                lambda run_id: service.run_store.get_run(run_id)
            )
            first = service.run_store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="first",
                chat_model="queue-test-model",
                temperature=0.0,
                prompt="first",
                status="running",
            )
            second = service.run_store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="second",
                chat_model="queue-test-model",
                temperature=0.0,
                prompt="second",
                status="running",
            )
            first_run_id = first["run_id"]
            second_run_id = second["run_id"]
            service._active_run_id = first_run_id
            original_claim = service.run_store.mark_run_cancelling_with_event

            def hand_off_after_claim(candidate_run_id):
                result = original_claim(candidate_run_id)
                with service._control_lock:
                    service._active_run_id = second_run_id
                return result

            service.run_store.mark_run_cancelling_with_event = hand_off_after_claim  # type: ignore[method-assign]

            response = service.cancel_run(first_run_id)

            self.assertEqual(response["status"], "cancelling")
            self.assertEqual(service._active_run_id, second_run_id)
            self.assertEqual(abort_calls, [])
            self.assertEqual(
                service.run_store.get_run(first_run_id)["status"],
                "cancelling",
            )


if __name__ == "__main__":
    unittest.main()
