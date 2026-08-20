import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atlas_local.code_runner import (
    CodeRunner,
    DEFAULT_RUNNER_MAX_FILE_BYTES,
    LANGUAGES,
    LEGACY_PYTHON_GUI_IMAGES,
    PYTHON_GUI_IMAGE,
    PYTHON_GUI_RUNTIME_ALLOWED_PACKAGES,
    PYTHON_GUI_RUNTIME_CONTEXT,
    PYTHON_GUI_RUNTIME_NAME,
    PYTHON_GUI_RUNTIME_VERSION,
    PythonGuiRuntimeManager,
    RUNNER_INTERNAL_NETWORK,
    RUNNER_MAX_CODE_BYTES,
    RunPlan,
    RunnerProcess,
    _ensure_internal_network,
    _inspect_python_gui_runtime,
    _python_gui_runtime_definition_hash,
    prepare_runner_runtime,
    resolve_plan,
    _remove_legacy_python_gui_images,
    _runner_network_policy,
    _runner_storage_limit,
    _runner_storage_limit_supported,
    _runner_timeout_seconds,
    docker_status,
)


def _wait_for_runner_cleanup(path: Path, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if path.exists():
        raise AssertionError(f"Runner work directory was not cleaned up: {path}")


class CodeRunnerPolicyTests(unittest.TestCase):
    def test_docker_status_hides_invocation_errors_and_logs_them_server_side(self) -> None:
        secret_detail = "private docker socket path"

        with (
            patch("atlas_local.code_runner._docker_binary", return_value="docker"),
            patch(
                "atlas_local.code_runner.subprocess.run",
                side_effect=OSError(secret_detail),
            ),
            self.assertLogs("atlas_local.code_runner", level="WARNING") as captured,
        ):
            status = docker_status()

        self.assertEqual(
            status,
            {
                "available": False,
                "reason": "Docker could not be checked. Verify Docker Desktop is installed and try again.",
            },
        )
        self.assertNotIn(secret_detail, json.dumps(status))
        self.assertIn(secret_detail, "\n".join(captured.output))

    def test_docker_status_hides_process_output_and_logs_it_server_side(self) -> None:
        public_reason = (
            "Docker Desktop is installed but not running. Start Docker Desktop and try again."
        )
        cases = (
            {"stderr": "private stderr detail", "stdout": ""},
            {"stderr": "", "stdout": "private stdout detail"},
        )

        for completed_output in cases:
            with self.subTest(completed_output=completed_output):
                with (
                    patch("atlas_local.code_runner._docker_binary", return_value="docker"),
                    patch(
                        "atlas_local.code_runner.subprocess.run",
                        return_value=SimpleNamespace(
                            returncode=1,
                            **completed_output,
                        ),
                    ),
                    self.assertLogs("atlas_local.code_runner", level="WARNING") as captured,
                ):
                    status = docker_status()

                self.assertEqual(status, {"available": False, "reason": public_reason})
                serialized = json.dumps(status)
                private_detail = completed_output["stderr"] or completed_output["stdout"]
                self.assertNotIn(private_detail, serialized)
                self.assertIn(private_detail, "\n".join(captured.output))

    def test_default_network_is_isolated_for_non_gui_runs(self) -> None:
        plan = RunPlan(image="python:3.12-slim", filename="main.py", command=["python", "/work/main.py"])

        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(_runner_network_policy(plan), "none")

    def test_gui_runs_use_an_internal_network_for_vnc_without_outbound_access(self) -> None:
        plan = RunPlan(
            image=PYTHON_GUI_IMAGE,
            filename="main.py",
            command=["python", "/work/main.py"],
            ports={12345: 6080},
            gui=True,
        )

        with patch.dict("os.environ", {"ATLAS_RUNNER_NETWORK": "none"}):
            self.assertEqual(_runner_network_policy(plan), RUNNER_INTERNAL_NETWORK)

    def test_bridge_network_requires_explicit_configuration(self) -> None:
        plan = RunPlan(
            image=PYTHON_GUI_IMAGE,
            filename="main.py",
            command=["python", "/work/main.py"],
            ports={12345: 6080},
            gui=True,
            requires_network=True,
        )

        with patch.dict("os.environ", {"ATLAS_RUNNER_NETWORK": "bridge"}):
            self.assertEqual(_runner_network_policy(plan), "bridge")

    def test_timeout_policy_uses_env_override(self) -> None:
        plan = RunPlan(image="python:3.12-slim", filename="main.py", command=["python", "/work/main.py"])

        with patch.dict("os.environ", {"ATLAS_RUNNER_TIMEOUT_SECONDS": "7"}):
            self.assertEqual(_runner_timeout_seconds(plan), 7)

    def test_start_adds_runner_safety_docker_flags(self) -> None:
        captured: dict[str, list[str]] = {}

        class FakeProcess:
            stdout: list[str] = []
            stderr: list[str] = []

            def wait(self) -> int:
                return 0

            def kill(self) -> None:
                return None

        def fake_popen(args, **_kwargs):
            captured["args"] = args
            return FakeProcess()

        with tempfile.TemporaryDirectory() as tmp:
            plan = RunPlan(image="python:3.12-slim", filename="main.py", command=["python", "/work/main.py"])
            with (
                patch("atlas_local.code_runner._docker_binary", return_value="docker"),
                patch("atlas_local.code_runner.resolve_plan", return_value=plan),
                patch("atlas_local.code_runner.subprocess.Popen", side_effect=fake_popen),
                patch(
                    "atlas_local.code_runner.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
                ),
                patch("atlas_local.code_runner.tempfile.mkdtemp", return_value=tmp),
            ):
                response = CodeRunner().start("python", "print('hello')")
                _wait_for_runner_cleanup(Path(tmp))

        args = captured["args"]
        self.assertIn("--network", args)
        self.assertEqual(args[args.index("--network") + 1], "none")
        self.assertIn("--init", args)
        self.assertIn("--memory-swap", args)
        self.assertIn("--tmpfs", args)
        self.assertIn("--ulimit", args)
        self.assertIn(
            f"fsize={DEFAULT_RUNNER_MAX_FILE_BYTES}:{DEFAULT_RUNNER_MAX_FILE_BYTES}",
            args,
        )
        self.assertIn("--read-only", args)
        self.assertIn("--user", args)
        self.assertEqual(args[args.index("--user") + 1], "65534:65534")
        self.assertIn("--security-opt", args)
        self.assertIn("no-new-privileges", args)
        self.assertIn("--cap-drop", args)
        self.assertIn("ALL", args)
        self.assertIn("atlas.runner=1", args)
        self.assertTrue(any(value.startswith("atlas.runner.owner_id=") for value in args))
        self.assertEqual(response["configured_network"], "none")
        self.assertEqual(response["network"], "none")
        self.assertFalse(response["outbound_network"])
        self.assertEqual(response["filesystem_mode"], "read-only")
        self.assertEqual(response["timeout_seconds"], 120)

    def test_offline_web_plan_does_not_load_model_html_in_host_webview(self) -> None:
        captured: dict[str, list[str]] = {}

        class FakeProcess:
            stdout: list[str] = []
            stderr: list[str] = []

            def wait(self) -> int:
                return 0

            def kill(self) -> None:
                return None

        def fake_popen(args, **_kwargs):
            captured["args"] = args
            return FakeProcess()

        with tempfile.TemporaryDirectory() as tmp:
            plan = RunPlan(
                image="python:3.12-slim",
                filename="main.py",
                command=["python", "/work/main.py"],
                ports={12345: 5000},
                web_container_port=5000,
            )
            with (
                patch("atlas_local.code_runner._docker_binary", return_value="docker"),
                patch("atlas_local.code_runner.resolve_plan", return_value=plan),
                patch("atlas_local.code_runner._ensure_internal_network"),
                patch("atlas_local.code_runner.subprocess.Popen", side_effect=fake_popen),
                patch(
                    "atlas_local.code_runner.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
                ),
                patch("atlas_local.code_runner.tempfile.mkdtemp", return_value=tmp),
            ):
                response = CodeRunner().start("python", "print('hello')")
                _wait_for_runner_cleanup(Path(tmp))

        args = captured["args"]
        self.assertRegex(
            args[args.index("--network") + 1],
            rf"^{RUNNER_INTERNAL_NETWORK}-[0-9a-f]{{16}}$",
        )
        self.assertIn("-p", args)
        self.assertIn("127.0.0.1:12345:5000", args)
        self.assertNotIn("web_url", response)
        self.assertTrue(response["web_preview_disabled"])
        self.assertFalse(response["outbound_network"])

    def test_explicit_bridge_web_plan_returns_web_url(self) -> None:
        class FakeProcess:
            stdout: list[str] = []
            stderr: list[str] = []

            def wait(self) -> int:
                return 0

            def kill(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmp:
            plan = RunPlan(
                image="python:3.12-slim",
                filename="main.py",
                command=["python", "/work/main.py"],
                ports={12345: 5000},
                web_container_port=5000,
            )
            with (
                patch.dict("os.environ", {"ATLAS_RUNNER_NETWORK": "bridge"}),
                patch("atlas_local.code_runner._docker_binary", return_value="docker"),
                patch("atlas_local.code_runner.resolve_plan", return_value=plan),
                patch(
                    "atlas_local.code_runner.subprocess.Popen",
                    return_value=FakeProcess(),
                ),
                patch(
                    "atlas_local.code_runner.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout="",
                        stderr="",
                    ),
                ),
                patch(
                    "atlas_local.code_runner.tempfile.mkdtemp",
                    return_value=tmp,
                ),
            ):
                response = CodeRunner().start("python", "print('hello')")
                _wait_for_runner_cleanup(Path(tmp))

        self.assertEqual(
            response["web_url"],
            "http://127.0.0.1:12345/",
        )
        self.assertTrue(response["outbound_network"])
        self.assertNotIn("web_preview_disabled", response)

    def test_prepared_gui_start_stays_non_root_read_only_and_capability_free(self) -> None:
        captured: dict[str, list[str]] = {}

        class FakeProcess:
            stdout: list[str] = []
            stderr: list[str] = []

            def wait(self) -> int:
                return 0

            def kill(self) -> None:
                return None

        def fake_popen(args, **_kwargs):
            captured["args"] = args
            return FakeProcess()

        with tempfile.TemporaryDirectory() as tmp:
            plan = RunPlan(
                image=PYTHON_GUI_IMAGE,
                filename="main.py",
                command=["sh", "-c", "echo gui"],
                ports={12345: 6080},
                gui=True,
                isolated_gui_preview=True,
                runtime=PYTHON_GUI_RUNTIME_NAME,
            )
            with (
                patch("atlas_local.code_runner._docker_binary", return_value="docker"),
                patch("atlas_local.code_runner.resolve_plan", return_value=plan),
                patch(
                    "atlas_local.code_runner._python_gui_runtime_manager.status",
                    return_value={"state": "ready"},
                ),
                patch("atlas_local.code_runner._ensure_internal_network"),
                patch("atlas_local.code_runner.subprocess.Popen", side_effect=fake_popen),
                patch(
                    "atlas_local.code_runner.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
                ),
                patch("atlas_local.code_runner.tempfile.mkdtemp", return_value=tmp),
            ):
                CodeRunner().start("python", "print('hello')")
                _wait_for_runner_cleanup(Path(tmp))

        args = captured["args"]
        added_caps = [
            args[index + 1]
            for index, value in enumerate(args)
            if value == "--cap-add" and index + 1 < len(args)
        ]
        self.assertEqual(added_caps, [])
        self.assertIn("--read-only", args)
        self.assertIn("--user", args)
        self.assertEqual(args[args.index("--user") + 1], "65534:65534")
        self.assertIn("--cap-drop", args)
        self.assertEqual(args[args.index("--cap-drop") + 1], "ALL")
        self.assertIn("no-new-privileges", args)
        self.assertTrue(any(value.endswith(":/work:ro") for value in args))

    def test_gui_preview_fails_closed_without_privilege_boundary(self) -> None:
        plan = RunPlan(
            image=PYTHON_GUI_IMAGE,
            filename="main.py",
            command=["sh", "-c", "echo unsafe"],
            ports={12345: 6080},
            gui=True,
            uses_apt=True,
        )
        with (
            patch("atlas_local.code_runner._docker_binary", return_value="docker"),
            patch("atlas_local.code_runner.resolve_plan", return_value=plan),
            patch("atlas_local.code_runner.subprocess.Popen") as popen,
        ):
            with self.assertRaisesRegex(RuntimeError, "privilege boundary"):
                CodeRunner().start("python", "print('hello')")
        popen.assert_not_called()

    def test_gui_start_fails_closed_until_prepared_runtime_is_ready(self) -> None:
        plan = RunPlan(
            image=PYTHON_GUI_IMAGE,
            filename="main.py",
            command=["sh", "-c", "echo gui"],
            ports={12345: 6080},
            gui=True,
            isolated_gui_preview=True,
            runtime=PYTHON_GUI_RUNTIME_NAME,
        )
        with (
            patch("atlas_local.code_runner._docker_binary", return_value="docker"),
            patch("atlas_local.code_runner.resolve_plan", return_value=plan),
            patch(
                "atlas_local.code_runner._python_gui_runtime_manager.status",
                return_value={"state": "preparing"},
            ),
            patch("atlas_local.code_runner.subprocess.Popen") as popen,
            self.assertRaisesRegex(RuntimeError, "runtime is not ready"),
        ):
            CodeRunner().start("python", "import pygame")
        popen.assert_not_called()

    def test_docker_commands_do_not_use_login_shells(self) -> None:
        for language, spec in LANGUAGES.items():
            with self.subTest(language=language):
                if len(spec.command) >= 2 and spec.command[0] == "sh":
                    self.assertNotEqual(spec.command[1], "-lc")

    def test_runner_images_are_fully_qualified(self) -> None:
        for language, spec in LANGUAGES.items():
            with self.subTest(language=language):
                self.assertRegex(spec.image, r"^(docker\.io|mcr\.microsoft\.com)/")

    def test_package_manager_runners_remain_offline_by_default(self) -> None:
        for language in ("javascript", "typescript", "go", "rust", "ruby", "perl", "r", "dart"):
            with self.subTest(language=language):
                plan = resolve_plan(language, "print('ok')")
                self.assertTrue(plan.requires_network)
                with patch.dict("os.environ", {}, clear=False):
                    self.assertEqual(_runner_network_policy(plan), "none")

    def test_mutable_latest_and_stable_image_tags_are_not_used(self) -> None:
        for language, spec in LANGUAGES.items():
            with self.subTest(language=language):
                self.assertNotRegex(spec.image, r":(?:latest|stable)$")

    def test_internal_preview_network_is_created_without_outbound_routing(self) -> None:
        calls: list[list[str]] = []

        def fake_run(args, **_kwargs):
            calls.append(args)
            if args[1:3] == ["network", "inspect"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="not found")
            return SimpleNamespace(returncode=0, stdout=RUNNER_INTERNAL_NETWORK, stderr="")

        with patch("atlas_local.code_runner.subprocess.run", side_effect=fake_run):
            _ensure_internal_network("docker")

        create_call = next(args for args in calls if args[1:3] == ["network", "create"])
        self.assertIn("--internal", create_call)
        self.assertIn("--driver", create_call)
        self.assertIn("atlas.runner.network=1", create_call)

    def test_existing_non_internal_preview_network_is_rejected(self) -> None:
        with patch(
            "atlas_local.code_runner.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="false\n", stderr=""),
        ):
            with self.assertRaisesRegex(RuntimeError, "already exists but is not internal"):
                _ensure_internal_network("docker")

    def test_dependency_run_reports_offline_requirement_instead_of_upgrading_network(self) -> None:
        class FakeProcess:
            stdout: list[str] = []
            stderr: list[str] = []

            def wait(self) -> int:
                return 0

            def kill(self) -> None:
                return None

        plan = RunPlan(
            image="docker.io/library/node:20-alpine",
            filename="main.js",
            command=["node", "/work/main.js"],
            requires_network=True,
        )
        with (
            patch.dict("os.environ", {"ATLAS_RUNNER_NETWORK": "none"}),
            patch("atlas_local.code_runner._docker_binary", return_value="docker"),
            patch("atlas_local.code_runner.resolve_plan", return_value=plan),
            patch("atlas_local.code_runner.subprocess.Popen", return_value=FakeProcess()),
            patch(
                "atlas_local.code_runner.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ),
        ):
            runner = CodeRunner()
            response = runner.start("javascript", "console.log('hello')")
            history, _, _ = runner.subscribe(response["run_id"])

        self.assertEqual(response["configured_network"], "none")
        self.assertEqual(response["network"], "none")
        self.assertFalse(response["outbound_network"])
        self.assertTrue(response["dependency_network_required"])
        self.assertTrue(response["network_requirement_unmet"])
        self.assertIn("network_warning", response)
        self.assertTrue(
            any("explicitly allow outbound access" in event.get("chunk", "") for event in history)
        )

    def test_explicit_bridge_reports_outbound_network_available(self) -> None:
        captured: dict[str, list[str]] = {}

        class FakeProcess:
            stdout: list[str] = []
            stderr: list[str] = []

            def wait(self) -> int:
                return 0

            def kill(self) -> None:
                return None

        def fake_popen(args, **_kwargs):
            captured["args"] = args
            return FakeProcess()

        plan = RunPlan(
            image="docker.io/library/node:20-alpine",
            filename="main.js",
            command=["node", "/work/main.js"],
            requires_network=True,
        )
        with (
            patch.dict("os.environ", {"ATLAS_RUNNER_NETWORK": "bridge"}),
            patch("atlas_local.code_runner._docker_binary", return_value="docker"),
            patch("atlas_local.code_runner.resolve_plan", return_value=plan),
            patch("atlas_local.code_runner.subprocess.Popen", side_effect=fake_popen),
            patch(
                "atlas_local.code_runner.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ),
        ):
            response = CodeRunner().start("javascript", "console.log('hello')")

        self.assertEqual(captured["args"][captured["args"].index("--network") + 1], "bridge")
        self.assertTrue(response["outbound_network"])
        self.assertFalse(response["network_requirement_unmet"])
        self.assertNotIn("network_warning", response)

    def test_storage_limit_is_strictly_validated(self) -> None:
        with patch.dict("os.environ", {"ATLAS_RUNNER_STORAGE_LIMIT": "2GiB"}):
            self.assertEqual(_runner_storage_limit(), "2GiB")

        with patch.dict("os.environ", {"ATLAS_RUNNER_STORAGE_LIMIT": "2G;--privileged"}):
            with self.assertRaisesRegex(RuntimeError, "must be a positive byte count"):
                _runner_storage_limit()

    def test_storage_limit_support_is_detected_from_engine_help(self) -> None:
        with patch(
            "atlas_local.code_runner.subprocess.run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="Usage: docker run [OPTIONS]\n  --storage-opt list\n",
                stderr="",
            ),
        ):
            self.assertTrue(_runner_storage_limit_supported("docker"))

        with patch(
            "atlas_local.code_runner.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="Usage: podman run\n", stderr=""),
        ):
            self.assertFalse(_runner_storage_limit_supported("docker"))

    def test_code_size_is_bounded_before_docker_is_invoked(self) -> None:
        runner = CodeRunner()
        with (
            patch("atlas_local.code_runner._docker_binary") as docker_binary,
            self.assertRaisesRegex(RuntimeError, "Code is too large"),
        ):
            runner.start("python", "x" * (RUNNER_MAX_CODE_BYTES + 1))
        docker_binary.assert_not_called()

    def test_global_run_concurrency_limit_rejects_excess_runs_and_shutdown_cleans_up(self) -> None:
        class BlockingProcess:
            stdout: list[str] = []
            stderr: list[str] = []

            def __init__(self) -> None:
                self.done = threading.Event()
                self.killed = False

            def wait(self) -> int:
                self.done.wait()
                return 137 if self.killed else 0

            def kill(self) -> None:
                self.killed = True
                self.done.set()

        process = BlockingProcess()
        calls: list[list[str]] = []

        def fake_run(args, **_kwargs):
            calls.append(args)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        plan = RunPlan(
            image="docker.io/library/python:3.12-slim",
            filename="main.py",
            command=["python", "/work/main.py"],
        )
        runner = CodeRunner(max_concurrent=1)
        with (
            patch("atlas_local.code_runner._docker_binary", return_value="docker"),
            patch("atlas_local.code_runner.resolve_plan", return_value=plan),
            patch("atlas_local.code_runner.subprocess.Popen", return_value=process) as popen,
            patch("atlas_local.code_runner.subprocess.run", side_effect=fake_run),
        ):
            response = runner.start("python", "print('one')")
            with self.assertRaisesRegex(RuntimeError, "concurrency limit reached"):
                runner.start("python", "print('two')")
            runner.shutdown()
            runner.shutdown()
            with self.assertRaisesRegex(RuntimeError, "shutting down"):
                runner.start("python", "print('three')")

        self.assertEqual(popen.call_count, 1)
        self.assertTrue(process.killed)
        self.assertIn(
            ["docker", "rm", "-f", response["container"]],
            calls,
        )

    def test_history_output_and_subscriber_queues_are_bounded(self) -> None:
        code_runner = CodeRunner(
            history_limit=3,
            subscriber_queue_size=2,
            max_output_bytes=4,
        )
        runner = RunnerProcess(
            run_id="bounded",
            language="python",
            container_name="atlas-run-bounded",
            work_dir=Path(tempfile.gettempdir()) / "atlas-run-bounded-test",
            started_at=time.time(),
            timeout_seconds=120,
            network="none",
            process=SimpleNamespace(),
            history_limit=3,
            subscriber_queue_size=2,
            max_output_bytes=4,
        )
        subscriber: queue.Queue[dict[str, object]] = queue.Queue(maxsize=2)
        runner.subscribers.append(subscriber)

        for value in range(5):
            code_runner._emit(runner, {"type": "marker", "value": value})

        self.assertEqual([event["value"] for event in runner.history], [2, 3, 4])
        self.assertEqual(subscriber.qsize(), 2)
        self.assertEqual(subscriber.get_nowait()["value"], 3)
        self.assertEqual(subscriber.get_nowait()["value"], 4)

        output_runner = RunnerProcess(
            run_id="output",
            language="python",
            container_name="atlas-run-output",
            work_dir=Path(tempfile.gettempdir()) / "atlas-run-output-test",
            started_at=time.time(),
            timeout_seconds=120,
            network="none",
            process=SimpleNamespace(),
            history_limit=10,
            subscriber_queue_size=2,
            max_output_bytes=4,
        )
        code_runner._emit(
            output_runner,
            {"type": "output", "stream": "stdout", "chunk": "abcdef"},
        )
        code_runner._emit(
            output_runner,
            {"type": "output", "stream": "stdout", "chunk": "ignored"},
        )

        chunks = [event["chunk"] for event in output_runner.history]
        self.assertEqual(chunks[0], "abcd")
        self.assertEqual(sum("output truncated" in chunk for chunk in chunks), 1)
        self.assertEqual(output_runner.output_bytes, 4)

    def test_active_run_count_includes_starting_and_unfinished_runs_only(self) -> None:
        runner = CodeRunner()
        runner._starting_names.add("atlas-run-starting")
        active = RunnerProcess(
            run_id="active",
            language="python",
            container_name="atlas-run-active",
            work_dir=Path(tempfile.gettempdir()) / "atlas-run-active-test",
            started_at=time.time(),
            timeout_seconds=120,
            network="none",
            process=SimpleNamespace(),
            history_limit=10,
            subscriber_queue_size=2,
            max_output_bytes=100,
        )
        finished = RunnerProcess(
            run_id="finished",
            language="python",
            container_name="atlas-run-finished",
            work_dir=Path(tempfile.gettempdir()) / "atlas-run-finished-test",
            started_at=time.time(),
            timeout_seconds=120,
            network="none",
            process=SimpleNamespace(),
            history_limit=10,
            subscriber_queue_size=2,
            max_output_bytes=100,
            finished=True,
        )
        runner._runs = {active.run_id: active, finished.run_id: finished}

        self.assertEqual(runner.active_run_count(), 2)

    def test_cleanup_removes_current_and_dead_owner_containers_but_preserves_live_owners(self) -> None:
        runner = CodeRunner()
        removed: list[str] = []

        def fake_run(args, **_kwargs):
            if args[1] == "ps":
                return SimpleNamespace(
                    returncode=0,
                    stdout="owned-current\nowned-dead\nowned-live\n",
                    stderr="",
                )
            if args[1] == "inspect":
                name = args[-1]
                labels = {
                    "owned-current": (
                        '{"atlas.runner.owner_id":"'
                        + runner._owner_id
                        + '","atlas.runner.owner_pid":"1"}'
                    ),
                    "owned-dead": (
                        '{"atlas.runner.owner_id":"other","atlas.runner.owner_pid":"222"}'
                    ),
                    "owned-live": (
                        '{"atlas.runner.owner_id":"other","atlas.runner.owner_pid":"333"}'
                    ),
                }
                return SimpleNamespace(returncode=0, stdout=labels[name], stderr="")
            if args[1:3] == ["rm", "-f"]:
                removed.append(args[-1])
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(f"Unexpected command: {args}")

        with (
            patch("atlas_local.code_runner.subprocess.run", side_effect=fake_run),
            patch("atlas_local.code_runner._process_is_alive", side_effect=lambda pid: pid == 333),
        ):
            runner._cleanup_stale_containers("docker")

        self.assertEqual(set(removed), {"owned-current", "owned-dead"})

    def test_cleanup_removes_only_stale_labeled_atlas_networks_and_is_idempotent(
        self,
    ) -> None:
        runner = CodeRunner()
        active = f"{RUNNER_INTERNAL_NETWORK}-aaaaaaaaaaaaaaaa"
        dead = f"{RUNNER_INTERNAL_NETWORK}-bbbbbbbbbbbbbbbb"
        live = f"{RUNNER_INTERNAL_NETWORK}-cccccccccccccccc"
        unlabeled = f"{RUNNER_INTERNAL_NETWORK}-dddddddddddddddd"
        owned = f"{RUNNER_INTERNAL_NETWORK}-eeeeeeeeeeeeeeee"
        legacy = RUNNER_INTERNAL_NETWORK
        invalid_name = f"{RUNNER_INTERNAL_NETWORK}-not-a-run-id"
        removed: set[str] = set()
        labels = {
            active: {
                "atlas.runner.network": "1",
                "atlas.runner.owner_id": runner._owner_id,
                "atlas.runner.owner_pid": str(os.getpid()),
            },
            dead: {
                "atlas.runner.network": "1",
                "atlas.runner.owner_id": "other",
                "atlas.runner.owner_pid": "222",
            },
            live: {
                "atlas.runner.network": "1",
                "atlas.runner.owner_id": "other",
                "atlas.runner.owner_pid": "333",
            },
            unlabeled: {
                "atlas.runner.owner_id": runner._owner_id,
                "atlas.runner.owner_pid": str(os.getpid()),
            },
            owned: {
                "atlas.runner.network": "1",
                "atlas.runner.owner_id": runner._owner_id,
                "atlas.runner.owner_pid": str(os.getpid()),
            },
            legacy: {
                "atlas.runner.network": "1",
            },
        }
        runner._starting_names.add("atlas-run-aaaaaaaaaaaaaaaa")

        def fake_run(args, **_kwargs):
            if args[1:3] == ["network", "ls"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout="\n".join(
                        [
                            active,
                            dead,
                            live,
                            unlabeled,
                            owned,
                            legacy,
                            invalid_name,
                        ]
                    ),
                    stderr="",
                )
            if args[1:3] == ["network", "inspect"]:
                name = args[3]
                if name in removed:
                    return SimpleNamespace(returncode=1, stdout="", stderr="not found")
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(labels.get(name, {})),
                    stderr="",
                )
            if args[1:3] == ["network", "rm"]:
                removed.add(args[-1])
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(f"Unexpected command: {args}")

        with (
            patch("atlas_local.code_runner.subprocess.run", side_effect=fake_run),
            patch(
                "atlas_local.code_runner._process_is_alive",
                side_effect=lambda pid: pid == 333,
            ),
        ):
            runner._cleanup_stale_networks("docker")
            runner._cleanup_stale_networks("docker")

        self.assertEqual(removed, {dead, owned, legacy})
        self.assertNotIn(active, removed)
        self.assertNotIn(live, removed)
        self.assertNotIn(unlabeled, removed)
        self.assertNotIn(invalid_name, removed)

    def test_python_generated_runner_script_is_valid_shell_syntax(self) -> None:
        plan = resolve_plan("python", "name = 'requests'\nmodule = __import__(name)\nprint(module)")

        completed = subprocess.run(
            ["sh", "-n"],
            input=plan.command[-1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cleanup_removes_only_legacy_python_gui_images(self) -> None:
        calls: list[list[str]] = []

        def fake_run(args, **_kwargs):
            calls.append(args)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("atlas_local.code_runner.subprocess.run", side_effect=fake_run):
            _remove_legacy_python_gui_images("docker")

        removed_images = [args[-1] for args in calls]
        self.assertEqual(removed_images, list(LEGACY_PYTHON_GUI_IMAGES))
        self.assertNotIn(PYTHON_GUI_IMAGE, removed_images)

    def test_python_gui_runtime_definition_is_checked_in_and_hash_locked(self) -> None:
        dockerfile = (PYTHON_GUI_RUNTIME_CONTEXT / "Dockerfile").read_text(encoding="utf-8")
        requirements = (PYTHON_GUI_RUNTIME_CONTEXT / "runtime-requirements.txt").read_text(
            encoding="utf-8"
        )

        self.assertRegex(_python_gui_runtime_definition_hash(), r"^[0-9a-f]{64}$")
        self.assertIn("FROM docker.io/library/python:3.12.13-slim-bookworm@sha256:", dockerfile)
        self.assertIn("ATLAS_RUNTIME_DEFINITION_SHA256", dockerfile)
        self.assertIn("USER 65534:65534", dockerfile)
        self.assertIn("numpy==2.5.2", requirements)
        self.assertIn("pygame==2.6.1", requirements)
        self.assertIn("--hash=sha256:", requirements)
        self.assertEqual(
            PYTHON_GUI_RUNTIME_ALLOWED_PACKAGES,
            {"numpy": "2.5.2", "pygame": "2.6.1"},
        )

    def test_python_gui_runtime_inspection_requires_matching_definition_labels(self) -> None:
        matching = {
            "Config": {
                "Labels": {
                    "com.atlas.runner.runtime": PYTHON_GUI_RUNTIME_NAME,
                    "com.atlas.runner.runtime.version": PYTHON_GUI_RUNTIME_VERSION,
                    "com.atlas.runner.runtime.definition-sha256": "definition-hash",
                }
            },
            "Size": 1234,
        }
        with (
            patch(
                "atlas_local.code_runner._python_gui_runtime_definition_hash",
                return_value="definition-hash",
            ),
            patch(
                "atlas_local.code_runner.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps([matching]),
                    stderr="",
                ),
            ),
        ):
            self.assertEqual(_inspect_python_gui_runtime("docker"), (True, 1234, None))

        matching["Config"]["Labels"][
            "com.atlas.runner.runtime.definition-sha256"
        ] = "stale"
        with (
            patch(
                "atlas_local.code_runner._python_gui_runtime_definition_hash",
                return_value="definition-hash",
            ),
            patch(
                "atlas_local.code_runner.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps([matching]),
                    stderr="",
                ),
            ),
        ):
            ready, size, error = _inspect_python_gui_runtime("docker")

        self.assertFalse(ready)
        self.assertEqual(size, 1234)
        self.assertIn("stale", error or "")

    def test_python_gui_runtime_preparation_is_deduplicated(self) -> None:
        starts: list[tuple[object, tuple[object, ...]]] = []

        class FakeThread:
            def __init__(self, *, target, args, **_kwargs):
                self.target = target
                self.args = args

            def start(self) -> None:
                starts.append((self.target, self.args))

        manager = PythonGuiRuntimeManager()
        with (
            patch("atlas_local.code_runner._docker_binary", return_value="docker"),
            patch(
                "atlas_local.code_runner._inspect_python_gui_runtime",
                return_value=(False, None, None),
            ),
            patch("atlas_local.code_runner.threading.Thread", FakeThread),
        ):
            first = manager.prepare()
            second = manager.prepare()

        self.assertTrue(first["started"])
        self.assertFalse(second["started"])
        self.assertEqual(first["state"], "preparing")
        self.assertEqual(second["state"], "preparing")
        self.assertEqual(len(starts), 1)
        self.assertFalse(first["submitted_code_used_during_preparation"])

    def test_python_gui_runtime_build_uses_only_trusted_context(self) -> None:
        captured: dict[str, list[str]] = {}

        class FakeProcess:
            stdout: list[str] = []

            def wait(self) -> int:
                return 0

        def fake_popen(args, **_kwargs):
            captured["args"] = args
            return FakeProcess()

        manager = PythonGuiRuntimeManager()
        with (
            patch("atlas_local.code_runner.subprocess.Popen", side_effect=fake_popen),
            patch(
                "atlas_local.code_runner._inspect_python_gui_runtime",
                return_value=(True, 1234, None),
            ),
        ):
            manager._build("docker")
            self.assertEqual(manager.status("docker")["state"], "ready")

        args = captured["args"]
        self.assertEqual(args[:3], ["docker", "build", "--pull"])
        self.assertEqual(args[-1], str(PYTHON_GUI_RUNTIME_CONTEXT))
        self.assertIn(str(PYTHON_GUI_RUNTIME_CONTEXT / "Dockerfile"), args)
        self.assertIn(PYTHON_GUI_IMAGE, args)
        self.assertFalse(any("atlas-run-" in value for value in args))

    def test_runner_runtime_preparation_rejects_unbundled_gui_dependencies_early(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not include.*customtkinter"):
            prepare_runner_runtime("python", "import customtkinter")

        with patch(
            "atlas_local.code_runner.prepare_python_gui_runtime",
            return_value={"state": "preparing", "started": True},
        ):
            prepared = prepare_runner_runtime("python", "import pygame")

        self.assertTrue(prepared["required"])
        self.assertTrue(prepared["started"])
        self.assertEqual(prepared["runtime"]["state"], "preparing")

    def test_python_gui_uses_versioned_prepared_runtime_without_live_installs(self) -> None:
        plan = resolve_plan("python", "import tkinter\nroot = tkinter.Tk()\nroot.mainloop()")

        self.assertEqual(
            PYTHON_GUI_IMAGE,
            f"localhost/atlas-python-gui-runtime:{PYTHON_GUI_RUNTIME_VERSION}",
        )
        self.assertIn("atlas-python-gui:workspace1", LEGACY_PYTHON_GUI_IMAGES)
        self.assertIn("atlas-python-gui:workspace2", LEGACY_PYTHON_GUI_IMAGES)
        self.assertNotIn(PYTHON_GUI_IMAGE, LEGACY_PYTHON_GUI_IMAGES)
        self.assertEqual(plan.image, PYTHON_GUI_IMAGE)
        self.assertEqual(plan.runtime, PYTHON_GUI_RUNTIME_NAME)
        self.assertTrue(plan.isolated_gui_preview)
        self.assertFalse(plan.uses_apt)
        self.assertFalse(plan.requires_network)
        self.assertNotIn("apt-get", plan.command[-1])
        self.assertNotIn("pip install", plan.command[-1])
        self.assertIn("prepared offline Python GUI runtime", plan.command[-1])
        self.assertIn("session.screen0.workspaces: 1", plan.command[-1])
        self.assertIn("fluxbox -rc /tmp/atlas-fluxbox-init", plan.command[-1])
        self.assertNotIn("/root/.fluxbox", plan.command[-1])
        self.assertEqual(plan.unsupported_packages, ())

    def test_python_gui_separates_preview_services_and_submitted_code(self) -> None:
        plan = resolve_plan("python", "import pygame\npygame.display.set_mode((400, 300))")
        script = plan.command[-1]

        self.assertTrue(plan.isolated_gui_preview)
        self.assertEqual(plan.requested_packages, ("pygame",))
        self.assertEqual(plan.unsupported_packages, ())
        self.assertNotIn("chown", script)
        self.assertNotIn("setpriv", script)
        self.assertNotIn("apt-get", script)
        self.assertNotIn("pip install", script)
        self.assertRegex(
            script,
            r"env HOME=/tmp/atlas-web-home .* websockify --web /usr/share/novnc",
        )
        self.assertRegex(script, r"env HOME=/tmp/atlas-display-home .* Xvfb :99")
        self.assertRegex(script, r"env HOME=/tmp/atlas-display-home .* x11vnc -display :99")
        self.assertRegex(
            script,
            r"env HOME=/tmp/atlas-user .* python /tmp/atlas_python_repair.py "
            r"python -u /tmp/main.py",
        )
        self.assertIn("offline GUI runtime does not include", script)

    def test_python_gui_plan_uses_bundled_pygame_offline(self) -> None:
        plan = resolve_plan("python", "import pygame\npygame.display.set_mode((400, 300))")

        self.assertEqual(plan.image, PYTHON_GUI_IMAGE)
        self.assertTrue(plan.gui)
        self.assertFalse(plan.uses_apt)
        self.assertFalse(plan.requires_network)
        self.assertEqual(plan.runtime, PYTHON_GUI_RUNTIME_NAME)
        self.assertEqual(plan.requested_packages, ("pygame",))
        self.assertEqual(plan.unsupported_packages, ())

    def test_python_pygame_import_uses_gui_runner_even_without_literal_display_call(self) -> None:
        plan = resolve_plan("python", "import pygame\npygame.init()\nprint('ready')")

        self.assertTrue(plan.gui)
        self.assertFalse(plan.uses_apt)
        self.assertFalse(plan.requires_network)
        self.assertNotIn("libsdl2-2.0-0", plan.command[-1])
        self.assertIn("prepared offline Python GUI runtime", plan.command[-1])

    def test_python_input_script_uses_terminal_vnc_runner(self) -> None:
        plan = resolve_plan("python", "name = input('Name: ')\nprint(name)")

        self.assertTrue(plan.gui)
        self.assertFalse(plan.uses_apt)
        self.assertFalse(plan.requires_network)
        self.assertEqual(plan.unsupported_packages, ())
        self.assertIn("xterm", plan.command[-1])
        self.assertIn("TERM=xterm-256color", plan.command[-1])
        self.assertIn("starting terminal UI in virtual display", plan.command[-1])

    def test_click_cli_is_rejected_when_not_bundled_in_offline_runtime(self) -> None:
        plan = resolve_plan("python", "import click\nname = click.prompt('Name')\nprint(name)")

        self.assertTrue(plan.gui)
        self.assertFalse(plan.requires_network)
        self.assertEqual(plan.unsupported_packages, ("click",))
        self.assertIn("xterm", plan.command[-1])

    def test_curses_import_uses_terminal_vnc_runner(self) -> None:
        plan = resolve_plan("python", "import curses\ncurses.wrapper(lambda stdscr: stdscr.getch())")

        self.assertEqual(plan.image, PYTHON_GUI_IMAGE)
        self.assertTrue(plan.gui)
        self.assertFalse(plan.uses_apt)
        self.assertFalse(plan.requires_network)
        self.assertEqual(plan.unsupported_packages, ())
        self.assertIn("xterm", plan.command[-1])
        self.assertIn("TERM=xterm-256color", plan.command[-1])
        self.assertIn("starting terminal UI in virtual display", plan.command[-1])
        self.assertNotIn("pip install", plan.command[-1])

    def test_textual_import_is_rejected_when_not_bundled_in_offline_runtime(self) -> None:
        plan = resolve_plan("python", "from textual.app import App\nclass Demo(App): pass\nDemo().run()")

        self.assertTrue(plan.gui)
        self.assertFalse(plan.requires_network)
        self.assertEqual(plan.unsupported_packages, ("textual",))
        self.assertIn("xterm", plan.command[-1])

    def test_stdlib_python_keeps_default_network_isolation(self) -> None:
        plan = resolve_plan("python", "import json\nprint(json.dumps({'ok': True}))")

        self.assertFalse(plan.gui)
        self.assertFalse(plan.uses_apt)
        self.assertFalse(plan.requires_network)
        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(_runner_network_policy(plan), "none")

    def test_python_imports_trigger_needed_pip_and_system_dependencies(self) -> None:
        plan = resolve_plan("python", "import cv2\nprint(cv2.__version__)")

        self.assertFalse(plan.gui)
        self.assertTrue(plan.uses_apt)
        self.assertTrue(plan.requires_network)
        self.assertIn("opencv-python", plan.command[-1])
        self.assertIn("libgl1", plan.command[-1])
        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(_runner_network_policy(plan), "none")

    def test_python_dependency_hints_are_installed_without_running_shell_text(self) -> None:
        plan = resolve_plan(
            "python",
            "# pip install -q requests pandas>=2,<3 && echo no\nimport json\nprint(json.dumps({'ok': True}))",
        )

        self.assertFalse(plan.gui)
        self.assertTrue(plan.requires_network)
        self.assertIn("requests", plan.command[-1])
        self.assertIn("pandas>=2,<3", plan.command[-1])
        self.assertNotIn(" -q ", f" {plan.command[-1]} ")
        self.assertNotIn("echo no", plan.command[-1])

    def test_python_requirements_hint_adds_packages_not_imported_yet(self) -> None:
        plan = resolve_plan("python", "# Requirements: numpy, plotly\nprint('ready')")

        self.assertTrue(plan.requires_network)
        self.assertIn("numpy", plan.command[-1])
        self.assertIn("plotly", plan.command[-1])

    def test_python_no_external_requirements_hint_does_not_install_fake_packages(self) -> None:
        plan = resolve_plan("python", "# Requirements: no external packages required\nprint('ready')")

        self.assertFalse(plan.requires_network)
        self.assertNotIn("pip install", plan.command[-1])

    def test_python_dynamic_literal_imports_are_installed_before_run(self) -> None:
        plan = resolve_plan("python", "import importlib\nimage = importlib.import_module('PIL.Image')\nprint(image)")

        self.assertTrue(plan.requires_network)
        self.assertIn("Pillow", plan.command[-1])

    def test_python_dynamic_import_marker_enables_runtime_dependency_repair(self) -> None:
        plan = resolve_plan("python", "name = 'requests'\nmodule = __import__(name)\nprint(module)")

        self.assertTrue(plan.requires_network)
        self.assertIn("atlas_python_repair.py", plan.command[-1])
        self.assertIn("detected missing Python module", plan.command[-1])
        self.assertIn("retrying after dependency repair", plan.command[-1])
        self.assertNotIn("installing Python packages: requests", plan.command[-1])

    def test_python_dynamic_native_imports_add_system_dependencies(self) -> None:
        plan = resolve_plan("python", "import importlib\ncv2 = importlib.import_module('cv2')\nprint(cv2)")

        self.assertTrue(plan.uses_apt)
        self.assertTrue(plan.requires_network)
        self.assertIn("opencv-python", plan.command[-1])
        self.assertIn("libgl1", plan.command[-1])

    def test_python_namespace_imports_map_to_specific_pip_packages(self) -> None:
        plan = resolve_plan("python", "import google.generativeai as genai\nprint(genai)")

        self.assertTrue(plan.requires_network)
        self.assertIn("google-generativeai", plan.command[-1])
        self.assertNotIn(" google ", f" {plan.command[-1]} ")

    def test_python_from_namespace_import_maps_imported_member_package(self) -> None:
        plan = resolve_plan("python", "from google.cloud import storage\nprint(storage)")

        self.assertTrue(plan.requires_network)
        self.assertIn("google-cloud-storage", plan.command[-1])
        self.assertNotIn(" google ", f" {plan.command[-1]} ")

    def test_python_database_driver_imports_use_binary_friendly_packages(self) -> None:
        plan = resolve_plan("python", "import psycopg2\nprint(psycopg2.__version__)")

        self.assertTrue(plan.uses_apt)
        self.assertTrue(plan.requires_network)
        self.assertIn("psycopg2-binary", plan.command[-1])
        self.assertIn("libpq5", plan.command[-1])

    def test_python_media_imports_add_runtime_system_dependencies(self) -> None:
        plan = resolve_plan("python", "import moviepy.editor\nimport soundfile\nprint('ready')")

        self.assertTrue(plan.uses_apt)
        self.assertTrue(plan.requires_network)
        self.assertIn("moviepy", plan.command[-1])
        self.assertIn("soundfile", plan.command[-1])
        self.assertIn("ffmpeg", plan.command[-1])
        self.assertIn("libsndfile1", plan.command[-1])

    def test_python_flask_plan_exposes_web_preview_port(self) -> None:
        plan = resolve_plan(
            "python",
            "from flask import Flask\napp = Flask(__name__)\napp.run(host='0.0.0.0', port=5000)",
        )

        self.assertFalse(plan.gui)
        self.assertEqual(plan.web_container_port, 5000)
        self.assertIn(5000, plan.ports.values())
        self.assertTrue(plan.requires_network)
        self.assertIn("flask", plan.command[-1])
        self.assertIn("flask --app main:app run --host 0.0.0.0 --port 5000", plan.command[-1])
        self.assertIn("HOST=0.0.0.0", plan.command[-1])
        self.assertIn("web preview will use container port 5000", plan.command[-1])
        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(_runner_network_policy(plan), RUNNER_INTERNAL_NETWORK)

    def test_python_fastapi_definition_launches_with_uvicorn(self) -> None:
        plan = resolve_plan("python", "from fastapi import FastAPI\napi = FastAPI()\n")

        self.assertFalse(plan.gui)
        self.assertEqual(plan.web_container_port, 8000)
        self.assertIn("fastapi", plan.command[-1])
        self.assertIn("uvicorn", plan.command[-1])
        self.assertIn("uvicorn main:api --host 0.0.0.0 --port 8000", plan.command[-1])

    def test_python_streamlit_launches_with_streamlit_command(self) -> None:
        plan = resolve_plan("python", "import streamlit as st\nst.write('Hello')")

        self.assertFalse(plan.gui)
        self.assertEqual(plan.web_container_port, 8501)
        self.assertIn("streamlit", plan.command[-1])
        self.assertIn("streamlit run /tmp/main.py", plan.command[-1])

    def test_python_http_client_import_does_not_trigger_web_preview(self) -> None:
        plan = resolve_plan("python", "import http.client\nprint('ok')")

        self.assertFalse(plan.gui)
        self.assertIsNone(plan.web_container_port)
        self.assertNotIn("web preview will use", plan.command[-1])

    def test_python_http_server_marker_exposes_web_preview(self) -> None:
        plan = resolve_plan(
            "python",
            "from http.server import HTTPServer, SimpleHTTPRequestHandler\nserver = HTTPServer(('0.0.0.0', 8000), SimpleHTTPRequestHandler)\nserver.serve_forever()",
        )

        self.assertFalse(plan.gui)
        self.assertEqual(plan.web_container_port, 8000)
        self.assertIn(8000, plan.ports.values())

    def test_customtkinter_import_is_rejected_when_not_bundled(self) -> None:
        plan = resolve_plan("python", "import customtkinter as ctk\napp = ctk.CTk()\napp.mainloop()")

        self.assertTrue(plan.gui)
        self.assertFalse(plan.uses_apt)
        self.assertFalse(plan.requires_network)
        self.assertEqual(plan.unsupported_packages, ("customtkinter",))

    def test_tkinter_import_uses_gui_runner_for_system_tk_deps(self) -> None:
        plan = resolve_plan("python", "import tkinter as tk\nprint('cli mode')")

        self.assertEqual(plan.image, PYTHON_GUI_IMAGE)
        self.assertTrue(plan.gui)
        self.assertFalse(plan.uses_apt)
        self.assertFalse(plan.requires_network)
        self.assertEqual(plan.unsupported_packages, ())

    def test_tkinter_from_import_uses_gui_runner_for_system_tk_deps(self) -> None:
        plan = resolve_plan("python", "from tkinter import ttk\nprint('cli mode')")

        self.assertEqual(plan.image, PYTHON_GUI_IMAGE)
        self.assertTrue(plan.gui)

    def test_direct_tkinter_window_uses_gui_runner(self) -> None:
        plan = resolve_plan("python", "import tkinter\nroot = tkinter.Tk()\nroot.mainloop()")

        self.assertEqual(plan.image, PYTHON_GUI_IMAGE)
        self.assertTrue(plan.gui)

    def test_python_gui_plan_passes_gui_flag_when_declared(self) -> None:
        code = "\n".join(
            [
                "import argparse",
                "import tkinter as tk",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--gui', action='store_true')",
                "args = parser.parse_args()",
                "if args.gui:",
                "    root = tk.Tk()",
                "    root.mainloop()",
                "else:",
                "    print('cli mode')",
            ],
        )

        plan = resolve_plan("python", code)

        self.assertTrue(plan.gui)
        self.assertIn("python -u /tmp/main.py --gui", plan.command[-1])


if __name__ == "__main__":
    unittest.main()
