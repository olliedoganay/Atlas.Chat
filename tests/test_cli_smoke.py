import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from atlas_local.cli import main


class FakeService:
    def __init__(self, *, locked: bool = False):
        self.locked = locked
        self.closed = False
        self.unlock_calls: list[tuple[str, str | None]] = []
        self.lock_calls: list[str] = []
        self.started: dict[str, str] | None = None

    def list_users(self):
        return [{"user_id": "u1", "locked": self.locked}]

    def unlock_user(self, *, user_id: str, password: str | None = None):
        self.unlock_calls.append((user_id, password))
        if self.locked and password != "atlas-secret":
            raise RuntimeError("Password did not match this user.")
        self.locked = False
        return {"user_id": user_id, "locked": False}

    def lock_user(self, *, user_id: str):
        self.lock_calls.append(user_id)
        self.locked = True
        return {"user_id": user_id, "locked": True}

    def start_chat(self, *, prompt: str, user_id: str, thread_id: str, chat_model: str):
        self.started = {
            "prompt": prompt,
            "user_id": user_id,
            "thread_id": thread_id,
            "chat_model": chat_model,
        }
        return {"run_id": "run-1", "status": "queued"}

    def get_run(self, run_id: str):
        return {
            "run_id": run_id,
            "status": "completed",
            "answer": (
                f"echo:{self.started['prompt']}|{self.started['user_id']}|"
                f"{self.started['thread_id']}|{self.started['chat_model']}"
            ),
        }

    def list_memories(self, *, user_id: str, limit: int = 20):
        return []

    def close(self):
        self.closed = True


class CliSmokeTests(unittest.TestCase):
    @patch("atlas_local.cli.AtlasBackendService.create")
    def test_ask_command_uses_authenticated_service_and_prints_answer(self, create_service) -> None:
        service = FakeService()
        create_service.return_value = service
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                ["ask", "hello", "--user-id", "u1", "--thread-id", "t1", "--model", "model-a"]
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("echo:hello|u1|t1|model-a", stdout.getvalue())
        self.assertEqual(service.unlock_calls, [("u1", None)])
        self.assertEqual(service.lock_calls, ["u1"])
        self.assertTrue(service.closed)

    @patch("atlas_local.cli.AtlasBackendService.create")
    def test_password_profile_is_unlocked_from_secure_prompt(self, create_service) -> None:
        service = FakeService(locked=True)
        create_service.return_value = service

        with (
            patch("atlas_local.cli.getpass.getpass", return_value="atlas-secret") as password_prompt,
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(
                ["ask", "hello", "--user-id", "u1", "--thread-id", "t1", "--model", "model-a"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(service.unlock_calls, [("u1", "atlas-secret")])
        password_prompt.assert_called_once()
        self.assertTrue(service.closed)

    @patch("atlas_local.cli.AtlasBackendService.create")
    def test_wrong_profile_password_is_rejected_before_chat(self, create_service) -> None:
        service = FakeService(locked=True)
        create_service.return_value = service
        stderr = io.StringIO()

        with (
            patch("atlas_local.cli.getpass.getpass", return_value="wrong-secret"),
            redirect_stderr(stderr),
        ):
            exit_code = main(
                ["ask", "hello", "--user-id", "u1", "--thread-id", "t1", "--model", "model-a"]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("Password did not match", stderr.getvalue())
        self.assertIsNone(service.started)
        self.assertTrue(service.closed)


if __name__ == "__main__":
    unittest.main()
