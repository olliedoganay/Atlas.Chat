import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from atlas_local.code_runner import (
    CodeRunner,
    LANGUAGES,
    LEGACY_PYTHON_GUI_IMAGES,
    PYTHON_GUI_IMAGE,
    RunPlan,
    resolve_plan,
    _remove_legacy_python_gui_images,
    _runner_network_policy,
    _runner_timeout_seconds,
)


class CodeRunnerPolicyTests(unittest.TestCase):
    def test_default_network_is_isolated_for_non_gui_runs(self) -> None:
        plan = RunPlan(image="python:3.12-slim", filename="main.py", command=["python", "/work/main.py"])

        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(_runner_network_policy(plan), "none")

    def test_gui_runs_force_bridge_network_for_vnc_port(self) -> None:
        plan = RunPlan(
            image=PYTHON_GUI_IMAGE,
            filename="main.py",
            command=["python", "/work/main.py"],
            ports={12345: 6080},
            gui=True,
        )

        with patch.dict("os.environ", {"ATLAS_RUNNER_NETWORK": "none"}):
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

        args = captured["args"]
        self.assertIn("--network", args)
        self.assertEqual(args[args.index("--network") + 1], "none")
        self.assertIn("--security-opt", args)
        self.assertIn("no-new-privileges", args)
        self.assertIn("--cap-drop", args)
        self.assertIn("ALL", args)
        self.assertIn("atlas.runner=1", args)
        self.assertEqual(response["network"], "none")
        self.assertEqual(response["timeout_seconds"], 120)

    def test_start_returns_web_url_for_web_plan(self) -> None:
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
                patch("atlas_local.code_runner.subprocess.Popen", side_effect=fake_popen),
                patch(
                    "atlas_local.code_runner.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
                ),
                patch("atlas_local.code_runner.tempfile.mkdtemp", return_value=tmp),
            ):
                response = CodeRunner().start("python", "print('hello')")

        args = captured["args"]
        self.assertIn("-p", args)
        self.assertIn("127.0.0.1:12345:5000", args)
        self.assertEqual(response["web_url"], "http://127.0.0.1:12345/")

    def test_gui_start_keeps_only_package_install_capabilities(self) -> None:
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
                uses_apt=True,
            )
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
                CodeRunner().start("python", "print('hello')")

        args = captured["args"]
        added_caps = [
            args[index + 1]
            for index, value in enumerate(args)
            if value == "--cap-add" and index + 1 < len(args)
        ]
        self.assertEqual(added_caps, ["CHOWN", "DAC_OVERRIDE", "FOWNER", "SETGID", "SETUID"])

    def test_docker_commands_do_not_use_login_shells(self) -> None:
        for language, spec in LANGUAGES.items():
            with self.subTest(language=language):
                if len(spec.command) >= 2 and spec.command[0] == "sh":
                    self.assertNotEqual(spec.command[1], "-lc")

    def test_runner_images_are_fully_qualified_for_podman_compatibility(self) -> None:
        for language, spec in LANGUAGES.items():
            with self.subTest(language=language):
                self.assertRegex(spec.image, r"^(docker\.io|mcr\.microsoft\.com)/")

    def test_package_manager_runners_use_bridge_network(self) -> None:
        for language in ("javascript", "typescript", "go", "rust", "ruby", "perl", "r", "dart"):
            with self.subTest(language=language):
                plan = resolve_plan(language, "print('ok')")
                self.assertTrue(plan.requires_network)
                with patch.dict("os.environ", {}, clear=False):
                    self.assertEqual(_runner_network_policy(plan), "bridge")

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

    def test_python_gui_uses_disposable_base_image_and_runtime_dependencies(self) -> None:
        plan = resolve_plan("python", "import tkinter\nroot = tkinter.Tk()\nroot.mainloop()")

        self.assertEqual(PYTHON_GUI_IMAGE, "docker.io/library/python:3.12-slim")
        self.assertIn("atlas-python-gui:workspace1", LEGACY_PYTHON_GUI_IMAGES)
        self.assertIn("atlas-python-gui:workspace2", LEGACY_PYTHON_GUI_IMAGES)
        self.assertNotIn(PYTHON_GUI_IMAGE, LEGACY_PYTHON_GUI_IMAGES)
        self.assertEqual(plan.image, PYTHON_GUI_IMAGE)
        self.assertIn("apt-get install", plan.command[-1])
        self.assertIn("session.screen0.workspaces: 1", plan.command[-1])
        self.assertIn("fluxbox -rc /root/.fluxbox/init", plan.command[-1])
        self.assertIn("tcl8.6", plan.command[-1])
        self.assertIn("tk8.6", plan.command[-1])

    def test_python_gui_plan_installs_gui_dependencies_on_demand(self) -> None:
        plan = resolve_plan("python", "import pygame\npygame.display.set_mode((400, 300))")

        self.assertEqual(plan.image, PYTHON_GUI_IMAGE)
        self.assertTrue(plan.gui)
        self.assertTrue(plan.uses_apt)
        self.assertTrue(plan.requires_network)

    def test_python_pygame_import_uses_gui_runner_even_without_literal_display_call(self) -> None:
        plan = resolve_plan("python", "import pygame\npygame.init()\nprint('ready')")

        self.assertTrue(plan.gui)
        self.assertTrue(plan.uses_apt)
        self.assertIn("libsdl2-2.0-0", plan.command[-1])
        self.assertIn("pygame", plan.command[-1])

    def test_python_input_script_uses_terminal_vnc_runner(self) -> None:
        plan = resolve_plan("python", "name = input('Name: ')\nprint(name)")

        self.assertTrue(plan.gui)
        self.assertTrue(plan.uses_apt)
        self.assertIn("xterm", plan.command[-1])
        self.assertIn("TERM=xterm-256color", plan.command[-1])
        self.assertIn("starting terminal UI in virtual display", plan.command[-1])

    def test_click_cli_uses_terminal_vnc_runner(self) -> None:
        plan = resolve_plan("python", "import click\nname = click.prompt('Name')\nprint(name)")

        self.assertTrue(plan.gui)
        self.assertTrue(plan.requires_network)
        self.assertIn("click", plan.command[-1])
        self.assertIn("xterm", plan.command[-1])

    def test_curses_import_uses_terminal_vnc_runner(self) -> None:
        plan = resolve_plan("python", "import curses\ncurses.wrapper(lambda stdscr: stdscr.getch())")

        self.assertEqual(plan.image, PYTHON_GUI_IMAGE)
        self.assertTrue(plan.gui)
        self.assertTrue(plan.uses_apt)
        self.assertTrue(plan.requires_network)
        self.assertIn("xterm", plan.command[-1])
        self.assertIn("ncurses-term", plan.command[-1])
        self.assertIn("TERM=xterm-256color", plan.command[-1])
        self.assertIn("starting terminal UI in virtual display", plan.command[-1])
        self.assertNotIn("pip install", plan.command[-1])

    def test_textual_import_uses_terminal_vnc_runner(self) -> None:
        plan = resolve_plan("python", "from textual.app import App\nclass Demo(App): pass\nDemo().run()")

        self.assertTrue(plan.gui)
        self.assertTrue(plan.requires_network)
        self.assertIn("textual", plan.command[-1])
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
            self.assertEqual(_runner_network_policy(plan), "bridge")

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
            self.assertEqual(_runner_network_policy(plan), "bridge")

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

    def test_customtkinter_import_triggers_pip_and_tk_system_dependencies(self) -> None:
        plan = resolve_plan("python", "import customtkinter as ctk\napp = ctk.CTk()\napp.mainloop()")

        self.assertTrue(plan.gui)
        self.assertTrue(plan.uses_apt)
        self.assertTrue(plan.requires_network)
        self.assertIn("customtkinter", plan.command[-1])
        self.assertIn("tcl8.6", plan.command[-1])

    def test_tkinter_import_uses_gui_runner_for_system_tk_deps(self) -> None:
        plan = resolve_plan("python", "import tkinter as tk\nprint('cli mode')")

        self.assertEqual(plan.image, PYTHON_GUI_IMAGE)
        self.assertTrue(plan.gui)
        self.assertTrue(plan.uses_apt)

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
