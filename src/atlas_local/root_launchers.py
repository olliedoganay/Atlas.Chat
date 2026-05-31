from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

DEFAULT_QUESTION = "What should I build next?"
DEFAULT_SMOKE_PROMPT = "Summarize the local-first architecture of this project in three concise bullets."


def maybe_reexec_with_repo_venv(script_path: Path) -> None:
    repo_root = script_path.resolve().parent
    candidates = (
        (
            repo_root / ".venv" / "Scripts" / "python.exe",
            repo_root / ".venv" / "Scripts" / "python",
        )
        if sys.platform == "win32"
        else (
            repo_root / ".venv" / "bin" / "python",
            repo_root / ".venv" / "bin" / "python3",
        )
    )
    repo_python = next((candidate for candidate in candidates if candidate.exists()), None)

    if repo_python is None:
        return

    current_python = Path(sys.executable).resolve()
    target_python = repo_python.resolve()
    if current_python == target_python:
        return

    os.execv(str(target_python), [str(target_python), str(script_path), *sys.argv[1:]])


def run_chat_launcher(
    cli_main: Callable[[list[str]], int],
    *,
    description: str,
    argv: list[str] | None = None,
    default_question: str = DEFAULT_QUESTION,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--thread-id", default="main")
    parser.add_argument("--model", required=True, help="Local chat model to use.")
    parser.add_argument(
        "--ask",
        action="store_true",
        help="Run a one-shot local turn instead of chat.",
    )
    args = parser.parse_args(argv)

    if args.ask or args.question:
        cli_args = ["ask", args.question or default_question, "--thread-id", args.thread_id]
    else:
        cli_args = ["chat", "--thread-id", args.thread_id]

    cli_args.extend(["--user-id", args.user_id, "--model", args.model])
    return cli_main(cli_args)


def run_smoke_launcher(
    cli_main: Callable[[list[str]], int],
    *,
    description: str,
    argv: list[str] | None = None,
    default_prompt: str = DEFAULT_SMOKE_PROMPT,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("prompt", nargs="?", default=default_prompt)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--thread-id", default="smoke-test")
    parser.add_argument("--model", required=True, help="Local chat model to use.")
    args = parser.parse_args(argv)

    cli_args = ["ask", args.prompt, "--user-id", args.user_id, "--thread-id", args.thread_id, "--model", args.model]
    return cli_main(cli_args)
