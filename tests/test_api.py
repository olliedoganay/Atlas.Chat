import os
import queue
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from atlas_local.api import create_api_app


class FakeService:
    def __init__(self) -> None:
        self.artifacts = {
            "run-1": {
                "run_id": "run-1",
                "status": "completed",
                "events": [
                    {"type": "run_started", "timestamp": "2026-04-08T00:00:00Z", "payload": {"mode": "chat"}},
                    {"type": "token", "timestamp": "2026-04-08T00:00:01Z", "payload": {"text": "hello"}},
                    {"type": "run_completed", "timestamp": "2026-04-08T00:00:02Z", "payload": {"answer": "hello"}},
                ],
                "trace_items": [],
                "answer": "hello",
            }
        }
        self.subscribers: dict[str, list[queue.Queue[dict[str, object]]]] = {}
        self.close_calls = 0
        self.closed = threading.Event()

    def close(self) -> None:
        self.close_calls += 1
        self.closed.set()

    def health(self):
        return {"status": "ok", "product": "Atlas Chat"}

    def status(self):
        return {
            "status": "ok",
            "product_name": "Atlas Chat",
            "backend": "Atlas Chat local runtime",
            "default_chat_temperature": None,
            "chat_temperature": None,
            "embed_model": "embed-model",
            "ollama_url": "http://127.0.0.1:11434",
            "runtime_mode": "chat-only",
            "busy": False,
        }

    def list_models(self):
        return {
            "default_temperature": None,
            "ollama_online": True,
            "has_local_models": True,
            "catalog_source": "ollama",
            "temperature_presets": [
                {"label": "0.0", "value": 0.0},
                {"label": "0.1", "value": 0.1},
                {"label": "0.2", "value": 0.2},
            ],
            "context_window_presets": [4096, 8192, 16384],
            "ollama_context_window": {
                "configured_context_window": 8192,
                "effective_context_window": 8192,
                "source": "configured",
            },
            "loaded_models": ["test-model"],
            "models": ["test-model", "model-b"],
            "model_details": [
                {"name": "test-model", "supports_images": False},
                {"name": "model-b", "supports_images": True},
            ],
        }

    def get_provider_settings(self):
        return {
            "provider": "ollama",
            "provider_label": "Ollama",
            "base_url": "http://127.0.0.1:11434",
            "has_api_key": False,
            "secure_key_storage_available": True,
            "providers": [
                {
                    "id": "ollama",
                    "label": "Ollama",
                    "default_base_url": "http://127.0.0.1:11434",
                }
            ],
        }

    def save_provider_settings(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str | None,
        preserve_existing_key: bool,
    ):
        return {
            **self.get_provider_settings(),
            "provider": provider,
            "base_url": base_url,
            "has_api_key": bool(api_key) or preserve_existing_key,
            "restart_required": True,
        }

    def clear_provider_api_key(self):
        return {**self.get_provider_settings(), "restart_required": True}

    def start_model_pull(self, *, model: str):
        return {
            "pull_id": "pull-1",
            "model": model,
            "status": "pulling",
            "detail": "Downloading",
            "completed": 1,
            "total": 2,
            "progress": 0.5,
        }

    def list_model_pulls(self):
        return [self.start_model_pull(model="test-model")]

    def get_model_pull(self, pull_id: str):
        return {**self.start_model_pull(model="test-model"), "pull_id": pull_id}

    def cancel_model_pull(self, pull_id: str):
        return {
            **self.get_model_pull(pull_id),
            "status": "cancelled",
            "detail": "Download cancelled",
        }

    def set_ollama_context_window(self, *, context_window):
        return {
            "configured_context_window": context_window,
            "ollama_context_window": {
                "configured_context_window": context_window,
                "effective_context_window": context_window or 4096,
                "source": "configured" if context_window else "ollama",
            },
        }

    def unload_ollama_model(self, *, model: str):
        return {"status": "unloaded", "model": model, "loaded_models": []}

    def discovery(self):
        return {
            "system": {
                "os": "Windows 11",
                "platform": "win32",
                "cpu": {"model": "Test CPU", "logical_cores": 16},
                "memory": {"total_gb": 32.0},
                "gpus": [{"name": "Test GPU", "memory_gb": 12.0}],
                "detection": {"confidence": "full", "notes": []},
            },
            "atlas": {
                "status": "memory-degraded",
                "summary": "Atlas can start chats, but memory retrieval is degraded until the embed model is installed.",
                "notes": ["Choose any installed chat model in Workspace before starting a new thread."],
                "ollama_url": "http://127.0.0.1:11434",
                "ollama_online": True,
                "has_local_chat_models": True,
                "configured_embed_model": "embed-model",
                "configured_embed_model_installed": False,
            },
            "installed_models": [
                {
                    "name": "test-model",
                    "atlas_role": "chat",
                    "configured_embed_model": False,
                    "supports_images": False,
                    "supports_reasoning": False,
                }
            ],
            "recommended_models": [
                {
                    "name": "test-model",
                    "title": "Configured chat model",
                    "use_case": "chat",
                    "atlas_role": "chat",
                    "installed": True,
                    "supports_images": False,
                    "fit": "good",
                    "runtime": "GPU",
                    "reason": "Fits comfortably.",
                    "pull_command": "ollama pull test-model",
                }
            ],
        }

    def list_users(self):
        return [
            {
                "user_id": "research_user",
                "updated_at": "2026-04-08T00:00:00Z",
                "protection": "passwordless",
                "locked": False,
            }
        ]

    def create_user(self, *, user_id: str, password: str | None = None):
        return {
            "user_id": user_id,
            "updated_at": "2026-04-08T00:00:00Z",
            "protection": "password" if password else "passwordless",
            "locked": False,
        }

    def unlock_user(self, *, user_id: str, password: str | None = None):
        return {
            "user_id": user_id,
            "updated_at": "2026-04-08T00:00:00Z",
            "protection": "password",
            "locked": False,
        }

    def lock_user(self, *, user_id: str):
        return {
            "user_id": user_id,
            "updated_at": "2026-04-08T00:00:00Z",
            "protection": "password",
            "locked": True,
        }

    def list_memories(self, *, user_id: str, limit: int = 50):
        return [{"memory": "name: Atlas Tester", "memory_id": "mem-1", "metadata": {"source": "manual"}}][:limit]

    def add_memory(self, *, user_id: str, text: str):
        return {"status": "ok", "user_id": user_id, "memory_id": "mem-2", "text": text}

    def delete_memory(self, *, user_id: str, memory_id: str):
        return {"status": "ok", "user_id": user_id, "memory_id": memory_id}

    def reset_user(self, *, user_id: str, confirmation_user_id: str):
        if user_id != confirmation_user_id:
            raise RuntimeError("User confirmation did not match the requested user id.")
        return {"status": "ok", "user_id": user_id}

    def reset_thread(self, *, thread_id: str, user_id: str):
        return {"status": "ok", "user_id": user_id, "thread_id": thread_id}

    def list_threads(self, *, user_id=None):
        return [
            {
                "user_id": user_id or "research_user",
                "thread_id": "main",
                "title": "Main chat",
                "chat_model": "test-model",
                "temperature": 0.2,
                "last_mode": "chat",
            }
        ]

    def rename_thread(self, *, user_id: str, thread_id: str, title: str):
        return {"user_id": user_id, "thread_id": thread_id, "title": title}

    def duplicate_thread(self, *, user_id: str, thread_id: str):
        return {"user_id": user_id, "thread_id": f"{thread_id}-copy", "title": "Main chat copy", "chat_model": "test-model", "temperature": 0.2}

    def branch_thread(self, *, user_id: str, thread_id: str, after_message_count: int):
        return {
            "user_id": user_id,
            "thread_id": f"{thread_id}-branch",
            "title": "Main chat branch",
            "chat_model": "test-model",
            "temperature": 0.2,
            "after_message_count": after_message_count,
        }

    def start_manual_compact(self, *, user_id: str, thread_id: str):
        return {
            "run_id": "run-compact-1",
            "status": "queued",
            "mode": "compact",
            "chat_model": "test-model",
            "temperature": 0.0,
            "user_id": user_id,
            "thread_id": thread_id,
        }

    def cancel_run(self, run_id: str):
        return {"status": "cancelling", "run_id": run_id}

    def get_thread_history(self, *, user_id=None, thread_id: str):
        return [{"role": "user", "content": f"{user_id or 'research_user'}:{thread_id}", "attachments": []}]

    def search_threads(self, *, user_id: str, query: str, current_thread_id: str | None = None, limit: int = 8):
        return {
            "query": query,
            "current_thread_id": current_thread_id or "",
            "current_thread_results": [
                {
                    "thread_id": current_thread_id or "main",
                    "thread_title": "Main chat",
                    "chat_model": "test-model",
                    "updated_at": "2026-04-08T00:00:00Z",
                    "match_type": "message",
                    "role": "assistant",
                    "history_index": 1,
                    "snippet": "matching answer",
                }
            ],
            "other_thread_results": [
                {
                    "thread_id": "archive",
                    "thread_title": "Archive",
                    "chat_model": "model-b",
                    "updated_at": "2026-04-07T00:00:00Z",
                    "match_type": "thread",
                    "role": None,
                    "history_index": None,
                    "snippet": "matching archive",
                }
            ],
        }

    def get_run(self, run_id: str):
        if run_id not in self.artifacts:
            raise RuntimeError(f"Run not found: {run_id}")
        return self.artifacts[run_id]

    def subscribe(self, run_id: str):
        subscriber: queue.Queue[dict[str, object]] = queue.Queue()
        self.subscribers.setdefault(run_id, []).append(subscriber)
        return subscriber

    def unsubscribe(self, run_id: str, subscriber):
        self.subscribers[run_id] = [item for item in self.subscribers.get(run_id, []) if item is not subscriber]

    def start_chat(
        self,
        *,
        prompt: str,
        user_id: str,
        thread_id: str,
        chat_model=None,
        temperature=None,
        reasoning_mode: str | None = None,
        thread_title=None,
        cross_chat_memory: bool = True,
        auto_compact_long_chats: bool = True,
        attachments=None,
        images=None,
    ):
        return {
            "run_id": "run-1",
            "status": "running",
            "prompt": prompt,
            "user_id": user_id,
            "thread_id": thread_id,
            "thread_title": thread_title or thread_id,
            "chat_model": chat_model or "test-model",
            "temperature": temperature,
            "reasoning_mode": reasoning_mode,
            "cross_chat_memory": cross_chat_memory,
            "auto_compact_long_chats": auto_compact_long_chats,
            "attachments": attachments or [],
            "images": images or [],
        }

    def reset_all(self, *, confirmation: str):
        return {"status": "ok", "confirmation": confirmation}


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_api_app(FakeService()))

    def test_health_and_status(self) -> None:
        health = self.client.get("/health")
        status = self.client.get("/status")
        models = self.client.get("/models")
        discovery = self.client.get("/discovery")
        users = self.client.get("/users")
        memories = self.client.get("/memories", params={"user_id": "research_user"})
        self.assertEqual(health.status_code, 200)
        self.assertEqual(status.status_code, 200)
        self.assertEqual(models.status_code, 200)
        self.assertEqual(discovery.status_code, 200)
        self.assertEqual(users.status_code, 200)
        self.assertEqual(memories.status_code, 200)
        self.assertEqual(health.json()["product"], "Atlas Chat")
        self.assertEqual(models.json()["models"], ["test-model", "model-b"])
        self.assertIsNone(models.json()["default_temperature"])
        self.assertTrue(models.json()["ollama_online"])
        self.assertTrue(models.json()["has_local_models"])
        self.assertEqual(models.json()["loaded_models"], ["test-model"])
        self.assertTrue(models.json()["model_details"][1]["supports_images"])
        self.assertEqual(models.json()["context_window_presets"], [4096, 8192, 16384])
        self.assertEqual(models.json()["ollama_context_window"]["configured_context_window"], 8192)
        self.assertEqual(models.json()["ollama_context_window"]["effective_context_window"], 8192)
        self.assertEqual(discovery.json()["atlas"]["status"], "memory-degraded")
        self.assertEqual(discovery.json()["recommended_models"][0]["pull_command"], "ollama pull test-model")
        self.assertEqual(users.json()[0]["user_id"], "research_user")
        self.assertEqual(memories.json()[0]["memory_id"], "mem-1")

    def test_ollama_context_window_update(self) -> None:
        response = self.client.patch(
            "/models/context-window",
            json={"context_window": 16384},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["configured_context_window"], 16384)
        self.assertEqual(response.json()["ollama_context_window"]["configured_context_window"], 16384)

    def test_provider_settings_and_model_pull_endpoints(self) -> None:
        settings = self.client.get("/settings/provider")
        updated = self.client.put(
            "/settings/provider",
            json={
                "provider": "lmstudio",
                "base_url": "http://127.0.0.1:1234/v1",
                "api_key": "local-secret",
            },
        )
        started = self.client.post("/models/pulls", json={"model": "test-model"})
        listed = self.client.get("/models/pulls")
        cancelled = self.client.delete("/models/pulls/pull-1")

        self.assertEqual(settings.status_code, 200)
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["provider"], "lmstudio")
        self.assertNotIn("local-secret", updated.text)
        self.assertEqual(started.json()["progress"], 0.5)
        self.assertEqual(listed.json()[0]["pull_id"], "pull-1")
        self.assertEqual(cancelled.json()["status"], "cancelled")

    def test_provider_settings_reject_credentials_in_url(self) -> None:
        response = self.client.put(
            "/settings/provider",
            json={
                "provider": "openai-compatible",
                "base_url": "http://user:secret@127.0.0.1:8000/v1",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("without credentials", response.json()["detail"])

    def test_ollama_model_unload(self) -> None:
        response = self.client.post("/models/unload", json={"model": "test-model"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unloaded")
        self.assertEqual(response.json()["model"], "test-model")
        self.assertEqual(response.json()["loaded_models"], [])

    def test_thread_listing_and_run_details(self) -> None:
        response = self.client.get("/threads", params={"user_id": "research_user"})
        details = self.client.get("/runs/run-1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(details.status_code, 200)
        self.assertEqual(response.json()[0]["thread_id"], "main")
        self.assertEqual(response.json()[0]["title"], "Main chat")
        self.assertEqual(details.json()["answer"], "hello")

    def test_chat_search(self) -> None:
        response = self.client.get(
            "/search",
            params={"user_id": "research_user", "q": "match", "current_thread_id": "main", "limit": 6},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query"], "match")
        self.assertEqual(payload["current_thread_results"][0]["thread_id"], "main")
        self.assertEqual(payload["other_thread_results"][0]["thread_id"], "archive")

    def test_chat_run_creation(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "prompt": "hello",
                "user_id": "research_user",
                "thread_id": "main",
                "chat_model": "model-b",
                "temperature": 0.6,
                "thread_title": "Renamed chat",
                "cross_chat_memory": False,
                "auto_compact_long_chats": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], "run-1")
        self.assertEqual(response.json()["chat_model"], "model-b")
        self.assertEqual(response.json()["temperature"], 0.6)

    def test_chat_run_creation_allows_model_default_temperature(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "prompt": "hello",
                "user_id": "research_user",
                "thread_id": "main",
                "chat_model": "model-b",
                "temperature": None,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["temperature"])

    def test_chat_run_creation_allows_attachment_only_prompt(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "prompt": "",
                "user_id": "research_user",
                "thread_id": "main",
                "chat_model": "model-b",
                "attachments": [
                    {
                        "kind": "image",
                        "name": "sample.png",
                        "media_type": "image/png",
                        "data_url": "data:image/png;base64,AAAA",
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], "run-1")

    def test_chat_run_creation_rejects_empty_prompt_without_attachments(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "prompt": "",
                "user_id": "research_user",
                "thread_id": "main",
                "chat_model": "model-b",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Prompt or attachment is required", response.json()["detail"])

    def test_request_budgets_reject_oversized_and_control_character_inputs(self) -> None:
        oversized_prompt = self.client.post(
            "/chat",
            json={
                "prompt": "x" * 200_001,
                "user_id": "research_user",
                "thread_id": "main",
            },
        )
        invalid_user = self.client.post(
            "/users",
            json={"user_id": "unsafe\nprofile"},
        )
        oversized_code = self.client.post(
            "/runner/exec",
            json={"language": "python", "code": "x" * 1_000_001},
        )

        self.assertEqual(oversized_prompt.status_code, 422)
        self.assertEqual(invalid_user.status_code, 422)
        self.assertEqual(oversized_code.status_code, 422)

    def test_cancel_run(self) -> None:
        response = self.client.post("/runs/run-1/cancel")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "cancelling")
        self.assertEqual(response.json()["run_id"], "run-1")

    def test_manual_compact_run_creation(self) -> None:
        response = self.client.post("/threads/main/compact", json={"user_id": "research_user"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], "run-compact-1")
        self.assertEqual(response.json()["mode"], "compact")

    def test_user_creation_and_thread_rename(self) -> None:
        created = self.client.post("/users", json={"user_id": "atlas_user", "password": "atlas-secret"})
        unlocked = self.client.post("/users/atlas_user/unlock", json={"password": "atlas-secret"})
        locked = self.client.post("/users/atlas_user/lock")
        renamed = self.client.patch(
            "/threads/main/title",
            json={"user_id": "research_user", "title": "Research notes"},
        )
        duplicated = self.client.post("/threads/main/duplicate", json={"user_id": "research_user"})
        branched = self.client.post("/threads/main/branch", json={"user_id": "research_user", "after_message_count": 2})
        self.assertEqual(created.status_code, 200)
        self.assertEqual(unlocked.status_code, 200)
        self.assertEqual(locked.status_code, 200)
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(duplicated.status_code, 200)
        self.assertEqual(branched.status_code, 200)
        self.assertEqual(created.json()["user_id"], "atlas_user")
        self.assertEqual(created.json()["protection"], "password")
        self.assertFalse(created.json()["locked"])
        self.assertTrue(locked.json()["locked"])
        self.assertEqual(renamed.json()["title"], "Research notes")
        self.assertEqual(duplicated.json()["thread_id"], "main-copy")
        self.assertEqual(branched.json()["thread_id"], "main-branch")

    def test_user_delete(self) -> None:
        deleted = self.client.delete("/users/research_user", params={"confirmation_user_id": "research_user"})
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json()["user_id"], "research_user")

    def test_thread_reset_requires_profile_owner(self) -> None:
        missing_owner = self.client.post(
            "/admin/reset/thread",
            json={"thread_id": "main"},
        )
        reset = self.client.post(
            "/admin/reset/thread",
            json={"thread_id": "main", "user_id": "research_user"},
        )

        self.assertEqual(missing_owner.status_code, 422)
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(reset.json()["user_id"], "research_user")

    def test_memory_create_and_delete(self) -> None:
        created = self.client.post("/memories", json={"user_id": "research_user", "text": "remember this"})
        deleted = self.client.delete("/memories/mem-2", params={"user_id": "research_user"})
        self.assertEqual(created.status_code, 200)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(created.json()["memory_id"], "mem-2")
        self.assertEqual(deleted.json()["memory_id"], "mem-2")

    def test_stream_replays_events_in_order(self) -> None:
        with self.client.stream("GET", "/chat/stream/run-1") as response:
            self.assertEqual(response.status_code, 200)
            payload = "".join(response.iter_text())
        self.assertIn("event: run_started", payload)
        self.assertIn("event: token", payload)
        self.assertIn("event: run_completed", payload)
        self.assertLess(payload.index("event: run_started"), payload.index("event: token"))
        self.assertLess(payload.index("event: token"), payload.index("event: run_completed"))

    def test_reset_endpoints(self) -> None:
        reset_user = self.client.delete("/users/research_user", params={"confirmation_user_id": "research_user"})
        reset_all = self.client.post("/admin/reset/all", json={"confirmation": "RESET ATLAS"})
        self.assertEqual(reset_user.status_code, 200)
        self.assertEqual(reset_all.status_code, 200)

    def test_create_api_app_requires_instance_token_without_explicit_insecure_override(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ATLAS_INSTANCE_TOKEN is required"):
                create_api_app()

    def test_requests_require_instance_token_when_configured(self) -> None:
        with patch.dict(os.environ, {"ATLAS_INSTANCE_TOKEN": "test-token"}, clear=False):
            client = TestClient(create_api_app(FakeService()))
            unauthorized = client.get("/status")
            authorized = client.get("/status", headers={"X-Atlas-Instance-Token": "test-token"})

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)

    def test_prepare_shutdown_is_authenticated_and_idempotent(self) -> None:
        service = FakeService()
        server_shutdown_calls: list[str] = []
        server_shutdown_requested = threading.Event()

        def request_server_shutdown() -> None:
            server_shutdown_calls.append("shutdown")
            server_shutdown_requested.set()

        with patch.dict(os.environ, {"ATLAS_INSTANCE_TOKEN": "test-token"}, clear=False):
            client = TestClient(
                create_api_app(
                    service,
                    request_server_shutdown=request_server_shutdown,
                )
            )
            unauthorized = client.post("/admin/prepare-shutdown")
            scheduled = client.post(
                "/admin/prepare-shutdown",
                headers={"X-Atlas-Instance-Token": "test-token"},
            )
            self.assertTrue(service.closed.wait(1.0))
            self.assertTrue(server_shutdown_requested.wait(1.0))
            duplicate = client.post(
                "/admin/prepare-shutdown",
                headers={"X-Atlas-Instance-Token": "test-token"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(scheduled.status_code, 200)
        self.assertEqual(
            scheduled.json(),
            {"status": "shutdown-scheduled", "scheduled": True},
        )
        self.assertEqual(
            duplicate.json(),
            {"status": "shutdown-already-scheduled", "scheduled": False},
        )
        self.assertEqual(service.close_calls, 1)
        self.assertEqual(server_shutdown_calls, ["shutdown"])

    def test_requests_reject_untrusted_origins(self) -> None:
        with patch.dict(os.environ, {"ATLAS_INSTANCE_TOKEN": "test-token"}, clear=False):
            client = TestClient(create_api_app(FakeService()))
            response = client.get(
                "/status",
                headers={
                    "Origin": "https://evil.example",
                    "X-Atlas-Instance-Token": "test-token",
                },
            )

        self.assertEqual(response.status_code, 403)

    def test_stream_requires_header_token_not_query_token(self) -> None:
        with patch.dict(os.environ, {"ATLAS_INSTANCE_TOKEN": "test-token"}, clear=False):
            client = TestClient(create_api_app(FakeService()))
            query_response = client.get("/chat/stream/run-1?token=test-token")
            header_response = client.get(
                "/chat/stream/run-1",
                headers={"X-Atlas-Instance-Token": "test-token"},
            )

        self.assertEqual(query_response.status_code, 401)
        self.assertEqual(header_response.status_code, 200)

    def test_options_preflight_bypasses_instance_token(self) -> None:
        with patch.dict(os.environ, {"ATLAS_INSTANCE_TOKEN": "test-token"}, clear=False):
            client = TestClient(create_api_app(FakeService()))
            response = client.options(
                "/status",
                headers={
                    "Origin": "tauri://localhost",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "x-atlas-instance-token,content-type",
                },
            )

        self.assertNotEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
