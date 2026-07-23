import concurrent.futures
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from atlas_local.config import load_config
from atlas_local.run_store import RunStore
from atlas_local.run_store import _atomic_write_json
from atlas_local.run_store import _read_json_with_retry
from atlas_local.session import legacy_scoped_thread_id, scoped_thread_id


class RunStoreTests(unittest.TestCase):
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

            with mock.patch("atlas_local.run_store.os.replace", side_effect=flaky_replace):
                _atomic_write_json(path, {"status": "ok"})

            self.assertEqual(path.read_text(encoding="utf-8").strip(), '{\n  "status": "ok"\n}')
            self.assertEqual(attempts["count"], 2)

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
            self.assertIsNone(store.get_thread(user_id="research_user", thread_id="main")["temperature"])

    def test_streamed_tokens_are_visible_immediately_but_persisted_in_batches(self) -> None:
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
                self.assertEqual(write_run.call_count, 0)

                store.append_event(run_id, "stage_changed", {"stage": "finalize"})
                self.assertEqual(write_run.call_count, 1)

            persisted = store.get_run(run_id)
            self.assertEqual(persisted["answer"], "Atlas streams smoothly")
            self.assertEqual(persisted["events"][-1]["type"], "stage_changed")

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

            self.assertEqual(store.get_thread(user_id="a", thread_id="b::c")["title"], "first")
            self.assertEqual(store.get_thread(user_id="a::b", thread_id="c")["title"], "second")
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
            store.rename_thread(user_id="research_user", thread_id="main", title="migrated title")
            migrated = store._read_index()

            self.assertNotIn(legacy_key, migrated["threads"])
            self.assertIn(scoped_thread_id("research_user", "main"), migrated["threads"])
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
                [item["run_id"] for item in store.list_runs_for_thread(user_id="research_user", thread_id="thread-0")],
                [artifacts[0]["run_id"]],
            )
            for artifact in artifacts:
                self.assertEqual(store.get_run(str(artifact["run_id"]))["status"], "queued")

    def test_search_messages_indexes_prompt_and_answer_without_snapshot_state(self) -> None:
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
            store.complete_run(artifact["run_id"], answer="Atlantis appears in the archived travel notes.")

            user_hits = store.search_messages(user_id="research_user", query="draft notes")
            assistant_hits = store.search_messages(user_id="research_user", query="archived travel")

            self.assertEqual(user_hits[0]["role"], "user")
            self.assertEqual(user_hits[0]["history_index"], 2)
            self.assertEqual(assistant_hits[0]["role"], "assistant")
            self.assertEqual(assistant_hits[0]["history_index"], 3)

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

            recovered = store.fail_incomplete_runs(error="Atlas backend restarted while this run was active.")

            self.assertCountEqual(recovered, [queued["run_id"], running["run_id"]])
            self.assertEqual(store.get_run(queued["run_id"])["status"], "failed")
            self.assertEqual(store.get_run(running["run_id"])["status"], "failed")
            self.assertEqual(store.get_run(completed["run_id"])["status"], "completed")
            self.assertEqual(store.get_run(queued["run_id"])["events"][-1]["type"], "run_failed")

    def test_non_thread_touching_runs_do_not_overwrite_thread_lock_metadata(self) -> None:
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

    def test_create_user_can_store_password_protection_and_verify_password(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(project_root=Path(temp_dir), env={})
            store = RunStore(config)

            user = store.create_user("protected_user", password="atlas-secret")

            self.assertEqual(user["protection"], "password")
            self.assertIsNotNone(user["password_hash"])
            self.assertIsNotNone(user["password_salt"])
            self.assertTrue(store.verify_user_password("protected_user", "atlas-secret"))
            self.assertFalse(store.verify_user_password("protected_user", "wrong-secret"))

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
            store.create_run(
                mode="chat",
                user_id="research_user",
                thread_id="main",
                chat_model="gpt-oss:20b",
                temperature=0.2,
                prompt="hello",
            )

            store.delete_thread(user_id="research_user", thread_id="main")

            user = store.get_user("research_user")
            self.assertIsNotNone(user)
            self.assertEqual(user["protection"], "password")

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
            self.assertIn('"format": "atlas-dpapi-run-v1"', raw_text)
            self.assertEqual(store.get_run(artifact["run_id"])["prompt"], "super secret prompt")

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

            index_text = (config.data_dir / "runs" / "index.json").read_text(encoding="utf-8")
            self.assertNotIn("secret project", index_text)
            self.assertNotIn("this should not leak", index_text)
            self.assertIn('"format": "atlas-dpapi-index-v1"', index_text)
            self.assertEqual(store.list_threads(user_id="research_user")[0]["title"], "secret project")


if __name__ == "__main__":
    unittest.main()
