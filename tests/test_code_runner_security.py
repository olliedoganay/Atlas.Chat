import unittest
from unittest.mock import patch

from atlas_local.code_runner import (
    PythonGuiRuntimeManager,
    _inspect_python_gui_runtime,
    _python_declared_pip_packages,
    _python_gui_args,
)


class CodeRunnerSecurityTests(unittest.TestCase):
    def test_gui_argument_detection_uses_python_tokens(self) -> None:
        multiline_call = """
parser.add_argument(
    '--gui',
    action='store_true',
)
"""

        self.assertEqual(_python_gui_args(multiline_call), ["--gui"])
        self.assertEqual(_python_gui_args("# parser.add_argument('--gui')"), [])
        self.assertEqual(_python_gui_args("print(\"add_argument('--gui')\")"), [])

    def test_gui_argument_detection_handles_adversarial_unclosed_calls(self) -> None:
        code = "add_argument(" * 40_000 + "!"

        self.assertEqual(_python_gui_args(code), [])

    def test_dependency_hint_scanner_preserves_supported_forms(self) -> None:
        code = "\n".join(
            (
                "# pip install -q requests pandas>=2,<3 && echo ignored",
                "// %pip install pygame",
                "# Requirements: python3 -m pip install numpy, scipy to run the example",
                "# Dependencies: no external packages required",
            )
        )

        self.assertEqual(
            _python_declared_pip_packages(code),
            ["numpy", "pandas>=2,<3", "pygame", "requests", "scipy"],
        )

    def test_dependency_hint_scanner_handles_adversarial_whitespace(self) -> None:
        pip_hint = ",pip install " + (" " * 250_000) + "!"
        requirements_hint = "# Requirements:" + (" " * 250_000) + "!"

        self.assertEqual(_python_declared_pip_packages(pip_hint), [])
        self.assertEqual(_python_declared_pip_packages(requirements_hint), [])

    def test_runtime_inspection_does_not_return_exception_details(self) -> None:
        secret_detail = "/private/runtime/path: permission denied"
        with (
            patch(
                "atlas_local.code_runner.subprocess.run",
                side_effect=OSError(secret_detail),
            ),
            self.assertLogs("atlas_local.code_runner", level="WARNING") as captured,
        ):
            ready, size, error = _inspect_python_gui_runtime("docker")

        self.assertFalse(ready)
        self.assertIsNone(size)
        self.assertEqual(
            error, "Docker could not inspect the Python GUI runtime image."
        )
        self.assertNotIn(secret_detail, error)
        self.assertIn(secret_detail, "\n".join(captured.output))

    def test_runtime_build_status_does_not_return_exception_details(self) -> None:
        secret_detail = "/private/docker/socket: permission denied"
        manager = PythonGuiRuntimeManager()
        with (
            patch(
                "atlas_local.code_runner.subprocess.Popen",
                side_effect=OSError(secret_detail),
            ),
            self.assertLogs("atlas_local.code_runner", level="ERROR") as captured,
        ):
            manager._build("docker")
        with patch(
            "atlas_local.code_runner._inspect_python_gui_runtime",
            return_value=(False, None, None),
        ):
            status = manager.status("docker")

        self.assertEqual(status["state"], "failed")
        self.assertNotIn(secret_detail, status["error"])
        self.assertIn("Check Docker availability", status["error"])
        self.assertIn(secret_detail, "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
