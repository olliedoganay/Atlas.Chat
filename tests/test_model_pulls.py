import json
import time
import unittest
from io import BytesIO
from unittest.mock import patch

from atlas_local.model_pulls import ModelPull, ModelPullManager


class _Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class ModelPullManagerTests(unittest.TestCase):
    def test_pull_rejects_remote_provider_before_starting_worker(self) -> None:
        manager = ModelPullManager()

        with (
            patch("atlas_local.model_pulls.threading.Thread") as thread_mock,
            self.assertRaisesRegex(RuntimeError, "loopback"),
        ):
            manager.start(
                ollama_url="https://example.com",
                model="qwen3:8b",
            )

        thread_mock.assert_not_called()
        self.assertEqual(manager.list(), [])

    def test_cancelled_pull_cannot_be_overwritten_by_late_progress_or_completion(self) -> None:
        manager = ModelPullManager()
        pull = ModelPull(pull_id="pull-1", model="model:latest", status="pulling")
        manager._pulls[pull.pull_id] = pull

        cancelled = manager.cancel(pull.pull_id)
        manager._update(pull, detail="Model ready", completed=100, total=100)
        manager._update(pull, status="completed", detail="Model ready")

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(pull.status, "cancelled")
        self.assertEqual(pull.detail, "Download cancelled")
        self.assertEqual(pull.completed, 0)

    def test_pull_stream_reports_progress_and_completion(self) -> None:
        response = _Response(
            b"\n".join(
                [
                    json.dumps({"status": "pulling manifest"}).encode(),
                    json.dumps({"status": "downloading", "completed": 50, "total": 100}).encode(),
                    json.dumps({"status": "success", "completed": 100, "total": 100}).encode(),
                ]
            )
        )
        manager = ModelPullManager()
        with patch("atlas_local.model_pulls.provider_urlopen", return_value=response):
            started = manager.start(ollama_url="http://127.0.0.1:11434", model="qwen3:8b")
            final = self._wait_for_terminal(manager, started["pull_id"])

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["progress"], 1.0)
        self.assertEqual(final["detail"], "Model ready")

    def test_only_one_download_runs_at_a_time(self) -> None:
        gate = _BlockingResponse()
        manager = ModelPullManager(max_concurrent=1)
        with patch("atlas_local.model_pulls.provider_urlopen", return_value=gate):
            started = manager.start(ollama_url="http://127.0.0.1:11434", model="qwen3:8b")
            with self.assertRaisesRegex(RuntimeError, "already active"):
                manager.start(ollama_url="http://127.0.0.1:11434", model="gemma3:4b")
            manager.cancel(started["pull_id"])

    @staticmethod
    def _wait_for_terminal(manager: ModelPullManager, pull_id: str) -> dict:
        for _ in range(100):
            item = manager.get(pull_id)
            if item["status"] in {"completed", "failed", "cancelled"}:
                return item
            time.sleep(0.01)
        raise AssertionError("model pull did not finish")


class _BlockingResponse:
    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def __iter__(self):
        while not self.closed:
            time.sleep(0.01)
        return
        yield b""

    def close(self):
        self.closed = True


if __name__ == "__main__":
    unittest.main()
