import base64
import hashlib
import json
import concurrent.futures
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from atlas_local.config import load_config
from atlas_local.run_store import RunStore
from atlas_local.run_store import _PASSWORD_KEY_LENGTH
from atlas_local.run_store import _RUN_FORMAT_LEGACY_PLAINTEXT
from atlas_local.run_store import _RUN_FORMAT_V1
from atlas_local.run_store import _RUN_FORMAT_V2
from atlas_local.run_store import _atomic_write_json
from atlas_local.run_store import _derive_password_hash
from atlas_local.run_store import _read_json_with_retry
from atlas_local.security import protect_bytes, unprotect_bytes_with_key
from atlas_local import security as security_module
from atlas_local.session import legacy_scoped_thread_id, scoped_thread_id


class RunStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.secret_storage = mock.patch(
            "atlas_local.run_store.application_secret_protection_available",
            return_value=True,
        )
        self.secret_storage.start()

    def tearDown(self) -> None:
        self.secret_storage.stop()

    def test_atomic_write_retries_after_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "artifact.json"
            original_replace = __import__("os").replace
            attempts = {"count": 0}

            def flaky_replace(src, dst):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise PermissionError("simulated windows file lock")
                return original_replace(src, dst)

            with mock.patch(
                "atlas_local.run_store.os.replace", side_effect=flaky_replace
            ):
                _atomic_write_json(path, {"status": "ok"})

            self.assertEqual(
                path.read_text(encoding="utf-8").strip(), '{\n  "status": "ok"\n}'
            )
            self.assertEqual(attempts["count"], 2)

    @unittest.skipIf(os.name == "nt", "POSIX permission bits are not used on Windows")
    def test_run_store_directory_and_json_files_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="private",
            )

            self.assertEqual(store.runs_dir.stat().st_mode & 0o077, 0)
            self.assertEqual(store._index_path.stat().st_mode & 0o077, 0)
            self.assertEqual(
                (store.runs_dir / f"{artifact['run_id']}.json").stat().st_mode & 0o077,
                0,
            )

    def test_read_json_retries_after_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "artifact.json"
            path.write_text('{\n  "status": "ok"\n}', encoding="utf-8")
            original_read_text = Path.read_text
            attempts = {"count": 0}

            def flaky_read_text(self, *args, **kwargs):
                if self == path:
                    attempts["count"] += 1
                    if attempts["count"] == 1:
                        raise PermissionError("simulated windows file lock")
                return original_read_text(self, *args, **kwargs)

            with mock.patch("pathlib.Path.read_text", new=flaky_read_text):
                payload = _read_json_with_retry(path)

            self.assertEqual(payload, {"status": "ok"})
            self.assertEqual(attempts["count"], 2)

    def test_run_ids_reject_posix_and_windows_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)

            for unsafe_id in (
                "../outside",
                r"..\outside",
                "segment/child",
                r"segment\child",
                "..",
            ):
                with self.subTest(run_id=unsafe_id):
                    with self.assertRaisesRegex(RuntimeError, "unsafe path characters"):
                        store.get_run(unsafe_id)
                    with self.assertRaisesRegex(RuntimeError, "unsafe path characters"):
                        store._write_run_file(unsafe_id, {"run_id": unsafe_id})

    def test_create_run_preserves_model_default_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)

            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=None,
                prompt="hello",
            )

            self.assertIsNone(artifact["temperature"])
            self.assertIsNone(store.get_run(artifact["run_id"])["temperature"])
            self.assertIsNone(
                store.get_thread(user_id="research_user", thread_id="main")[
                    "temperature"
                ]
            )

    def test_streamed_tokens_are_visible_immediately_but_persisted_in_batches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="hello",
            )
            run_id = artifact["run_id"]

            with mock.patch.object(
                store,
                "_write_run_file",
                wraps=store._write_run_file,
            ) as write_run:
                for token in ("Atlas", " ", "streams", " ", "smoothly"):
                    store.append_event(run_id, "token", {"text": token})

                live = store.get_run(run_id)
                self.assertEqual(live["answer"], "Atlas streams smoothly")
                self.assertEqual(len(live["events"]), 5)
                self.assertEqual(
                    [event["sequence"] for event in live["events"]],
                    [1, 2, 3, 4, 5],
                )
                self.assertEqual(write_run.call_count, 0)

                store.append_event(run_id, "stage_changed", {"stage": "finalize"})
                self.assertEqual(write_run.call_count, 1)

            persisted = store.get_run(run_id)
            self.assertEqual(persisted["answer"], "Atlas streams smoothly")
            self.assertEqual(persisted["events"][-1]["type"], "stage_changed")
            self.assertEqual(len(persisted["events"]), 2)
            self.assertEqual(persisted["events"][0]["sequence"], 1)
            self.assertEqual(persisted["events"][0]["sequence_end"], 5)
            self.assertEqual(
                persisted["events"][0]["payload"]["text"],
                "Atlas streams smoothly",
            )
            self.assertEqual(persisted["events"][1]["sequence"], 6)

    def test_streamed_reasoning_tokens_are_persisted_in_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="hello",
            )
            run_id = artifact["run_id"]

            with mock.patch.object(
                store,
                "_write_run_file",
                wraps=store._write_run_file,
            ) as write_run:
                for token in ("plan", " ", "the", " ", "answer"):
                    store.append_event(
                        run_id,
                        "thinking_token",
                        {"text": token},
                    )

                live = store.get_run(run_id)
                self.assertEqual(
                    [event["type"] for event in live["events"]],
                    ["thinking_token"] * 5,
                )
                self.assertEqual(write_run.call_count, 0)

                store.append_event(
                    run_id,
                    "stage_changed",
                    {"stage": "synthesis"},
                )
                self.assertEqual(write_run.call_count, 1)

            persisted = store.get_run(run_id)
            self.assertEqual(persisted["reasoning"], "plan the answer")
            self.assertEqual(
                [event["type"] for event in persisted["events"]],
                ["thinking_token", "stage_changed"],
            )
            self.assertEqual(persisted["events"][0]["sequence"], 1)
            self.assertEqual(persisted["events"][0]["sequence_end"], 5)
            self.assertEqual(
                persisted["events"][0]["payload"]["text"],
                "plan the answer",
            )
            self.assertEqual(persisted["events"][1]["sequence"], 6)

    def test_one_hundred_thousand_fragments_persist_as_bounded_exact_chunks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="stress",
            )
            run_id = artifact["run_id"]

            with (
                mock.patch(
                    "atlas_local.run_store.time.monotonic",
                    return_value=0.0,
                ),
                mock.patch(
                    "atlas_local.run_store.now_timestamp",
                    return_value="2026-08-20T00:00:00Z",
                ),
                mock.patch.object(
                    store,
                    "_write_run_file",
                    wraps=store._write_run_file,
                ) as write_run,
            ):
                for _ in range(50_000):
                    store.append_event(run_id, "thinking_token", {"text": "r"})
                for _ in range(50_000):
                    store.append_event(run_id, "token", {"text": "a"})
                store.complete_run(run_id, answer="a" * 50_000)

            persisted = store.get_run(run_id)
            token_text = "".join(
                event.get("payload", {}).get("text", "")
                for event in persisted["events"]
                if event.get("type") == "token"
            )
            thinking_text = "".join(
                event.get("payload", {}).get("text", "")
                for event in persisted["events"]
                if event.get("type") == "thinking_token"
            )

            self.assertEqual(persisted["answer"], "a" * 50_000)
            self.assertEqual(persisted["reasoning"], "r" * 50_000)
            self.assertEqual(token_text, persisted["answer"])
            self.assertEqual(thinking_text, persisted["reasoning"])
            self.assertLess(len(persisted["events"]), 40)
            self.assertLess(write_run.call_count, 40)
            self.assertEqual(persisted["events"][0]["sequence"], 1)
            self.assertEqual(persisted["events"][-1]["sequence"], 100_001)
            self.assertEqual(persisted["events"][-1]["type"], "run_completed")
            self.assertLess(store._run_path(run_id).stat().st_size, 1_000_000)

    def test_pathological_alternating_stream_events_collapse_by_channel(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="stress",
            )
            run_id = artifact["run_id"]

            with mock.patch(
                "atlas_local.run_store._MAX_PERSISTED_RUN_EVENTS",
                8,
            ):
                for index in range(20):
                    event_type = "thinking_token" if index % 2 == 0 else "token"
                    store.append_event(run_id, event_type, {"text": "x"})
                store.append_event(run_id, "stage_changed", {"stage": "synthesis"})

            persisted = store.get_run(run_id)
            self.assertEqual(
                [event["type"] for event in persisted["events"]],
                ["thinking_token", "token", "stage_changed"],
            )
            self.assertEqual(persisted["reasoning"], "x" * 10)
            self.assertEqual(persisted["answer"], "x" * 10)
            self.assertEqual(persisted["events"][0]["sequence"], 1)
            self.assertEqual(persisted["events"][0]["sequence_end"], 10)
            self.assertEqual(persisted["events"][1]["sequence"], 11)
            self.assertEqual(persisted["events"][1]["sequence_end"], 20)
            self.assertEqual(persisted["events"][2]["sequence"], 21)

    def test_stream_size_limit_fails_before_accepting_an_extra_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="bounded",
            )
            run_id = artifact["run_id"]

            store.append_event(run_id, "token", {"text": "0123456789"})
            store.flush_pending_events()
            reopened = RunStore(config)
            # A metadata read primes sequence tracking before generation resumes.
            # Stream-size accounting must still be reconstructed independently.
            reopened.get_run(run_id)
            with mock.patch(
                "atlas_local.run_store._MAX_PERSISTED_STREAM_BYTES",
                10,
            ):
                with self.assertRaisesRegex(RuntimeError, "output size limit"):
                    reopened.append_event(run_id, "token", {"text": "x"})

            failed = reopened.fail_run(run_id, error="output size limit")
            self.assertEqual(failed["answer"], "0123456789")
            self.assertEqual(failed["events"][0]["sequence"], 1)
            self.assertEqual(failed["events"][1]["sequence"], 2)
            self.assertEqual(failed["events"][1]["type"], "run_failed")

    def test_reading_pending_events_does_not_reuse_an_event_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="hello",
            )

            first = store.append_event(
                artifact["run_id"],
                "token",
                {"text": "A"},
            )
            store.get_run(artifact["run_id"])
            second = store.append_event(
                artifact["run_id"],
                "token",
                {"text": "B"},
            )

            self.assertEqual(first["sequence"], 1)
            self.assertEqual(second["sequence"], 2)

    def test_cancelling_run_cannot_be_overwritten_by_late_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="hello",
            )

            store.mark_run_cancelling(artifact["run_id"])
            completed = store.complete_run(
                artifact["run_id"],
                answer="late answer",
            )

            self.assertEqual(completed["status"], "failed")
            self.assertEqual(completed["error"], "Run stopped by user.")
            self.assertEqual(completed["events"][-1]["type"], "run_failed")
            self.assertNotEqual(completed["answer"], "late answer")

    def test_thread_index_keys_do_not_collide_on_delimiter_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)

            store.upsert_thread(
                user_id="a",
                thread_id="b::c",
                title="first",
                chat_model="model-a",
                temperature=0.2,
            )
            store.upsert_thread(
                user_id="a::b",
                thread_id="c",
                title="second",
                chat_model="model-b",
                temperature=0.3,
            )

            self.assertEqual(
                store.get_thread(user_id="a", thread_id="b::c")["title"], "first"
            )
            self.assertEqual(
                store.get_thread(user_id="a::b", thread_id="c")["title"], "second"
            )
            self.assertNotEqual(
                scoped_thread_id("a", "b::c"),
                scoped_thread_id("a::b", "c"),
            )

    def test_legacy_thread_index_key_is_read_and_migrated_on_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            legacy_key = legacy_scoped_thread_id("research_user", "main")
            store._write_index(
                {
                    "threads": {
                        legacy_key: {
                            "user_id": "research_user",
                            "thread_id": "main",
                            "title": "legacy title",
                            "chat_model": "model-a",
                            "temperature": 0.2,
                            "last_mode": "chat",
                            "updated_at": "2026-01-01T00:00:00Z",
                            "last_prompt": "legacy prompt",
                            "last_run_id": "",
                        }
                    },
                    "runs": {},
                    "users": {},
                }
            )

            self.assertEqual(
                store.get_thread(user_id="research_user", thread_id="main")["title"],
                "legacy title",
            )
            store.rename_thread(
                user_id="research_user", thread_id="main", title="migrated title"
            )
            migrated = store._read_index()

            self.assertNotIn(legacy_key, migrated["threads"])
            self.assertIn(
                scoped_thread_id("research_user", "main"), migrated["threads"]
            )
            self.assertEqual(
                migrated["threads"][scoped_thread_id("research_user", "main")]["title"],
                "migrated title",
            )

    def test_concurrent_create_run_preserves_every_index_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            original_read_index = store._read_index

            def slow_read_index():
                index = original_read_index()
                time.sleep(0.01)
                return index

            store._read_index = slow_read_index  # type: ignore[method-assign]

            def create(index: int) -> dict[str, object]:
                return store.create_run(
                    mode="chat",
                    user_id="research_user",
                    thread_id=f"thread-{index}",
                    chat_model="gpt-oss:20b",
                    temperature=0.2,
                    prompt=f"prompt {index}",
                    status="queued",
                )

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    artifacts = list(executor.map(create, range(8)))
            finally:
                store._read_index = original_read_index  # type: ignore[method-assign]

            self.assertEqual(len(store.list_threads(user_id="research_user")), 8)
            self.assertCountEqual(
                [
                    item["run_id"]
                    for item in store.list_runs_for_thread(
                        user_id="research_user", thread_id="thread-0"
                    )
                ],
                [artifacts[0]["run_id"]],
            )
            for artifact in artifacts:
                self.assertEqual(
                    store.get_run(str(artifact["run_id"]))["status"], "queued"
                )

    def test_search_messages_indexes_prompt_and_answer_without_snapshot_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)

            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="draft notes about Atlantis",
                history_after_message_count=3,
            )
            store.complete_run(
                artifact["run_id"],
                answer="Atlantis appears in the archived travel notes.",
            )

            user_hits = store.search_messages(
                user_id="research_user", query="draft notes"
            )
            assistant_hits = store.search_messages(
                user_id="research_user", query="archived travel"
            )

            self.assertEqual(user_hits[0]["role"], "user")
            self.assertEqual(user_hits[0]["history_index"], 2)
            self.assertEqual(assistant_hits[0]["role"], "assistant")
            self.assertEqual(assistant_hits[0]["history_index"], 3)

    def test_queued_same_thread_run_moves_search_positions_to_execution_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            first = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="first queued prompt",
                status="queued",
                history_after_message_count=1,
            )
            second = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="second queued prompt",
                status="queued",
                history_after_message_count=1,
            )
            store.complete_run(first["run_id"], answer="first queued answer")

            updated = store.update_run_history_after_message_count(
                second["run_id"],
                history_after_message_count=3,
            )
            store.complete_run(second["run_id"], answer="second queued answer")

            prompt_hit = store.search_messages(
                user_id="research_user",
                query="second queued prompt",
            )[0]
            answer_hit = store.search_messages(
                user_id="research_user",
                query="second queued answer",
            )[0]
            indexed = store._read_index()["runs"][second["run_id"]]

            self.assertEqual(updated["history_after_message_count"], 3)
            self.assertEqual(indexed["history_after_message_count"], 3)
            self.assertEqual(prompt_hit["history_index"], 2)
            self.assertEqual(answer_hit["history_index"], 3)

    def test_fail_incomplete_runs_marks_queued_and_running_runs_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)

            queued = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="queued",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="queued",
                status="queued",
            )
            running = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="running",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="running",
            )
            completed = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="completed",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="completed",
            )
            store.complete_run(completed["run_id"], answer="done")

            recovered = store.fail_incomplete_runs(
                error="Atlas backend restarted while this run was active."
            )

            self.assertCountEqual(recovered, [queued["run_id"], running["run_id"]])
            self.assertEqual(store.get_run(queued["run_id"])["status"], "failed")
            self.assertEqual(store.get_run(running["run_id"])["status"], "failed")
            self.assertEqual(store.get_run(completed["run_id"])["status"], "completed")
            self.assertEqual(
                store.get_run(queued["run_id"])["events"][-1]["type"], "run_failed"
            )

    def test_fail_incomplete_runs_repairs_a_stale_active_index_without_overwriting_terminal_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="hello",
            )
            store.complete_run(artifact["run_id"], answer="durable answer")
            index = store._read_index()
            index["runs"][artifact["run_id"]]["status"] = "running"
            index["runs"][artifact["run_id"]].pop("completed_at", None)
            store._write_index(index)

            recovered = store.fail_incomplete_runs(error="restart recovery")

            persisted = store.get_run(artifact["run_id"])
            repaired_index = store._read_index()["runs"][artifact["run_id"]]
            self.assertEqual(recovered, [])
            self.assertEqual(persisted["status"], "completed")
            self.assertEqual(persisted["answer"], "durable answer")
            self.assertEqual(
                [event["type"] for event in persisted["events"]],
                ["run_completed"],
            )
            self.assertEqual(repaired_index["status"], "completed")
            self.assertEqual(
                repaired_index["completed_at"], persisted["completed_at"]
            )

    def test_non_thread_touching_runs_do_not_overwrite_thread_lock_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)

            chat_run = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.7,
                prompt="hello",
            )
            store.create_run(
                mode="compact",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.0,
                prompt="",
                status="queued",
                touch_thread=False,
            )

            thread = store.get_thread(user_id="research_user", thread_id="main")

            self.assertEqual(thread["last_mode"], "chat")
            self.assertEqual(thread["last_run_id"], chat_run["run_id"])
            self.assertEqual(thread["temperature"], 0.7)

    def test_create_user_can_store_password_protection_and_verify_password(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)

            user = store.create_user("protected_user", password="atlas-secret")

            self.assertEqual(user["protection"], "password")
            self.assertEqual(user["password_kdf"], "atlas-scrypt-split-v2")
            self.assertIsNotNone(user["password_hash"])
            self.assertIsNotNone(user["password_salt"])
            self.assertTrue(
                store.verify_user_password("protected_user", "atlas-secret")
            )
            self.assertFalse(
                store.verify_user_password("protected_user", "wrong-secret")
            )

    def test_password_verifier_and_key_encryption_key_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            user = store.create_user("protected_user", password="atlas-secret")
            salt = base64.b64decode(user["password_salt"], validate=True)
            verifier = base64.b64decode(user["password_hash"], validate=True)
            wrapped_key = base64.b64decode(user["wrapped_profile_key"], validate=True)
            material = hashlib.scrypt(
                b"atlas-secret",
                salt=salt,
                n=2**14,
                r=8,
                p=1,
                dklen=_PASSWORD_KEY_LENGTH * 2,
            )

            self.assertEqual(verifier, material[:_PASSWORD_KEY_LENGTH])
            self.assertNotEqual(verifier, material[_PASSWORD_KEY_LENGTH:])
            profile_key = unprotect_bytes_with_key(
                wrapped_key,
                key=material[_PASSWORD_KEY_LENGTH:],
                aad=b"atlas-profile-key-v2:protected_user",
            )
            self.assertEqual(len(profile_key), _PASSWORD_KEY_LENGTH)
            with self.assertRaises(Exception):
                unprotect_bytes_with_key(
                    wrapped_key,
                    key=verifier,
                    aad=b"atlas-profile-key-v2:protected_user",
                )

    def test_tampered_profile_key_fails_with_a_deterministic_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("protected_user", password="atlas-secret")
            index = store._read_index()
            user = index["users"]["protected_user"]
            wrapped = bytearray(
                base64.b64decode(user["wrapped_profile_key"], validate=True)
            )
            wrapped[-1] ^= 1
            user["wrapped_profile_key"] = base64.b64encode(wrapped).decode("ascii")
            store._write_index(index)

            with self.assertRaisesRegex(
                RuntimeError, "profile key failed authentication"
            ):
                store.unlock_user_key(
                    "protected_user",
                    password="atlas-secret",
                )
            self.assertFalse(store.is_user_key_unlocked("protected_user"))

    def test_lock_and_failed_unlock_zeroize_cached_profile_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("protected_user", password="atlas-secret")
            store.unlock_user_key("protected_user", password="atlas-secret")
            cached = store._user_keys["protected_user"]

            store.lock_user_key("protected_user")

            self.assertNotIn("protected_user", store._user_keys)
            self.assertEqual(bytes(cached), b"\x00" * len(cached))

            store.unlock_user_key("protected_user", password="atlas-secret")
            cached_again = store._user_keys["protected_user"]
            with self.assertRaisesRegex(RuntimeError, "Password did not match"):
                store.unlock_user_key("protected_user", password="wrong-secret")
            self.assertNotIn("protected_user", store._user_keys)
            self.assertEqual(bytes(cached_again), b"\x00" * len(cached_again))

    def test_delete_thread_preserves_user_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("research_user", password="atlas-secret")
            store.unlock_user_key("research_user", password="atlas-secret")
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="hello",
            )
            store.append_event(artifact["run_id"], "token", {"text": "hello"})
            self.assertIn(artifact["run_id"], store._next_event_sequences)

            store.delete_thread(user_id="research_user", thread_id="main")

            user = store.get_user("research_user")
            self.assertIsNotNone(user)
            self.assertEqual(user["protection"], "password")
            self.assertNotIn(artifact["run_id"], store._next_event_sequences)

    def test_upsert_thread_preserves_existing_user_protection_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("research_user", password="atlas-secret")

            store.upsert_thread(
                user_id="research_user",
                thread_id="main",
                title="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
            )

            user = store.get_user("research_user")
            self.assertIsNotNone(user)
            self.assertEqual(user["protection"], "password")
            self.assertTrue(store.verify_user_password("research_user", "atlas-secret"))

    def test_run_artifact_is_encrypted_at_rest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("research_user")

            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="super secret prompt",
            )

            run_path = config.data_dir / "runs" / f"{artifact['run_id']}.json"
            raw_text = run_path.read_text(encoding="utf-8")
            self.assertNotIn("super secret prompt", raw_text)
            self.assertIn('"format": "atlas-profile-run-v2"', raw_text)
            raw_payload = json.loads(raw_text)
            ciphertext = base64.b64decode(raw_payload["payload"], validate=True)
            self.assertNotIn(b"super secret prompt", ciphertext)
            self.assertEqual(
                store.get_run(artifact["run_id"])["prompt"], "super secret prompt"
            )

    def test_tampered_run_ciphertext_fails_with_a_deterministic_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="secret",
            )
            run_path = store.runs_dir / f"{artifact['run_id']}.json"
            raw = json.loads(run_path.read_text(encoding="utf-8"))
            ciphertext = bytearray(base64.b64decode(raw["payload"], validate=True))
            ciphertext[-1] ^= 1
            raw["payload"] = base64.b64encode(ciphertext).decode("ascii")
            _atomic_write_json(run_path, raw)

            with self.assertRaisesRegex(
                RuntimeError, "run artifact failed authentication"
            ):
                store.get_run(artifact["run_id"])

    def test_legacy_plaintext_run_is_validated_and_migrated_to_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(
                project_root=Path(temp_dir),
                env={"ATLAS_ALLOW_LEGACY_PLAINTEXT_MIGRATION": "1"},
            )
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="legacy plaintext secret",
            )
            run_id = artifact["run_id"]
            index = store._read_index()
            index["runs"][run_id].pop("artifact_format", None)
            store._write_index(index)
            _atomic_write_json(store.runs_dir / f"{run_id}.json", artifact)

            migrated_store = RunStore(config)
            self.assertEqual(
                migrated_store._read_index()["runs"][run_id]["artifact_format"],
                _RUN_FORMAT_LEGACY_PLAINTEXT,
            )
            migrated = migrated_store.get_run(run_id)

            self.assertEqual(migrated["prompt"], "legacy plaintext secret")
            raw = json.loads(
                (migrated_store.runs_dir / f"{run_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(raw["format"], _RUN_FORMAT_V2)
            self.assertNotIn("legacy plaintext secret", json.dumps(raw))
            self.assertEqual(
                migrated_store._read_index()["runs"][run_id]["artifact_format"],
                _RUN_FORMAT_V2,
            )

    def test_v2_run_cannot_be_downgraded_to_plausible_plaintext(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="original",
            )
            _atomic_write_json(
                store.runs_dir / f"{artifact['run_id']}.json",
                {**artifact, "prompt": "forged plaintext"},
            )

            with self.assertRaisesRegex(
                RuntimeError, "legacy plaintext run artifact"
            ):
                store.get_run(artifact["run_id"])

    def test_unknown_run_wrapper_format_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            artifact = store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="hello",
            )
            _atomic_write_json(
                store.runs_dir / f"{artifact['run_id']}.json",
                {
                    "format": "atlas-run-v999",
                    "user_id": "research_user",
                    "payload": "ignored",
                },
            )

            with self.assertRaisesRegex(RuntimeError, "format is not supported"):
                store.get_run(artifact["run_id"])

    def test_legacy_password_profile_and_run_migrate_to_v2_after_unlock_and_read(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            password = "atlas-secret"
            salt = b"legacy-salt-1234"
            password_hash = _derive_password_hash(password, salt)
            profile_key = b"k" * _PASSWORD_KEY_LENGTH
            store._write_index(
                {
                    "threads": {},
                    "runs": {},
                    "users": {
                        "legacy_user": {
                            "user_id": "legacy_user",
                            "updated_at": "2026-01-01T00:00:00Z",
                            "protection": "password",
                            "password_hash": base64.b64encode(password_hash).decode(
                                "ascii"
                            ),
                            "password_salt": base64.b64encode(salt).decode("ascii"),
                            "wrapped_profile_key": base64.b64encode(
                                protect_bytes(profile_key, entropy=password_hash)
                            ).decode("ascii"),
                        }
                    },
                }
            )
            run_id = "legacy-run"
            artifact = {
                "run_id": run_id,
                "user_id": "legacy_user",
                "prompt": "legacy secret",
            }
            _atomic_write_json(
                config.data_dir / "runs" / f"{run_id}.json",
                {
                    "format": _RUN_FORMAT_V1,
                    "user_id": "legacy_user",
                    "payload": base64.b64encode(
                        protect_bytes(
                            json.dumps(artifact).encode("utf-8"),
                            entropy=profile_key,
                        )
                    ).decode("ascii"),
                },
            )

            store.unlock_user_key("legacy_user", password=password)
            migrated_user = store.get_user("legacy_user")
            self.assertEqual(migrated_user["password_kdf"], "atlas-scrypt-split-v2")
            self.assertEqual(store.get_run(run_id)["prompt"], "legacy secret")
            migrated_run = json.loads(
                (config.data_dir / "runs" / f"{run_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(migrated_run["format"], "atlas-profile-run-v2")

            store.lock_user_key("legacy_user")
            store.unlock_user_key("legacy_user", password=password)
            self.assertEqual(store.get_run(run_id)["prompt"], "legacy secret")

    def test_password_profile_creation_and_unlock_fail_without_os_secret_storage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)

            with mock.patch(
                "atlas_local.run_store.application_secret_protection_available",
                return_value=False,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "require available OS secret storage"
                ):
                    store.create_user("blocked_user", password="atlas-secret")
            self.assertIsNone(store.get_user("blocked_user"))

            store.create_user("protected_user", password="atlas-secret")
            with mock.patch(
                "atlas_local.run_store.application_secret_protection_available",
                return_value=False,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "require available OS secret storage"
                ):
                    store.unlock_user_key("protected_user", password="atlas-secret")
            self.assertFalse(store.is_user_key_unlocked("protected_user"))

    @unittest.skipIf(os.name == "nt", "Windows always has the DPAPI backend")
    def test_run_store_rejects_an_unavailable_linux_keyring_backend(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            with (
                mock.patch(
                    "atlas_local.security._non_windows_secret_storage_supported",
                    return_value=False,
                ),
                mock.patch(
                    "atlas_local.run_store.application_secret_protection_available",
                    side_effect=security_module.application_secret_protection_available,
                ),
            ):
                self.assertFalse(
                    security_module.application_secret_protection_available()
                )
                with self.assertRaisesRegex(
                    RuntimeError, "secure OS key storage"
                ):
                    RunStore(config)

    def test_index_is_encrypted_at_rest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            store.create_user("research_user")
            store.upsert_thread(
                user_id="research_user",
                thread_id="main",
                title="secret project",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                last_prompt="this should not leak",
            )

            index_text = (config.data_dir / "runs" / "index.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("secret project", index_text)
            self.assertNotIn("this should not leak", index_text)
            self.assertIn('"format": "atlas-dpapi-index-v1"', index_text)
            self.assertEqual(
                store.list_threads(user_id="research_user")[0]["title"],
                "secret project",
            )

    def test_plaintext_index_requires_explicit_one_time_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            strict_config = load_config(project_root=root, env={})
            store = RunStore(strict_config)
            plaintext_index = {
                "threads": {},
                "runs": {},
                "users": {},
            }
            _atomic_write_json(store._index_path, plaintext_index)

            strict_store = RunStore(strict_config)
            with self.assertRaisesRegex(
                RuntimeError, "ATLAS_ALLOW_LEGACY_PLAINTEXT_MIGRATION"
            ):
                strict_store.list_users()

            migration_config = load_config(
                project_root=root,
                env={"ATLAS_ALLOW_LEGACY_PLAINTEXT_MIGRATION": "true"},
            )
            migration_store = RunStore(migration_config)
            self.assertEqual(migration_store.list_users(), [])
            migrated_raw = json.loads(
                migration_store._index_path.read_text(encoding="utf-8")
            )
            self.assertEqual(migrated_raw["format"], "atlas-dpapi-index-v1")

            self.assertEqual(RunStore(strict_config).list_users(), [])

    def test_forged_unprotected_index_wrapper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)
            forged_payload = base64.b64encode(
                json.dumps(
                    {"threads": {}, "runs": {}, "users": {}}
                ).encode("utf-8")
            ).decode("ascii")
            _atomic_write_json(
                store._index_path,
                {
                    "format": "atlas-dpapi-index-v1",
                    "payload": forged_payload,
                },
            )

            with self.assertRaisesRegex(RuntimeError, "failed authentication"):
                store.list_users()


if __name__ == "__main__":
    unittest.main()
