import json
import time
import unittest
from io import BytesIO
from unittest.mock import patch

from atlas_local.model_pulls import ModelPullManager


class _Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class ModelPullManagerTests(unittest.TestCase):
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
        with patch("atlas_local.model_pulls.urlopen", return_value=response):
            started = manager.start(ollama_url="http://127.0.0.1:11434", model="qwen3:8b")
            final = self._wait_for_terminal(manager, started["pull_id"])

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["progress"], 1.0)
        self.assertEqual(final["detail"], "Model ready")

    def test_only_one_download_runs_at_a_time(self) -> None:
        gate = _BlockingResponse()
        manager = ModelPullManager(max_concurrent=1)
        with patch("atlas_local.model_pulls.urlopen", return_value=gate):
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
