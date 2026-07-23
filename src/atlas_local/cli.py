from __future__ import annotations

import argparse
import getpass
import os
import sys
import time

from .api_service import AtlasBackendService
from .config import load_config
from .runtime import configure_console

_PROFILE_PASSWORD_ENV = "ATLAS_PROFILE_PASSWORD"
_RUN_POLL_SECONDS = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Atlas local chat CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Run a single local turn through the graph.")
    ask_parser.add_argument("prompt", help="Prompt to send to the agent.")
    ask_parser.add_argument("--user-id", required=True)
    ask_parser.add_argument("--thread-id", default="default-thread")
    ask_parser.add_argument("--model", required=True, help="Local chat model to use.")
    ask_parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read the profile password from standard input instead of prompting securely.",
    )

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session.")
    chat_parser.add_argument("--user-id", required=True)
    chat_parser.add_argument("--thread-id", default="default-thread")
    chat_parser.add_argument("--model", required=True, help="Local chat model to use.")
    chat_parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read the profile password from standard input instead of prompting securely.",
    )

    memories_parser = subparsers.add_parser("memories", help="List stored memories for a user.")
    memories_parser.add_argument("--user-id", required=True)
    memories_parser.add_argument("--limit", type=int, default=20)
    memories_parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read the profile password from standard input instead of prompting securely.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_console()
    config = load_config()
    parser = build_parser()
    args = parser.parse_args(argv)

    service: AtlasBackendService | None = None
    try:
        service = AtlasBackendService.create(config=config)
        _unlock_cli_profile(
            service,
            user_id=args.user_id,
            password_stdin=bool(args.password_stdin),
        )
        try:
            if args.command == "ask":
                return _run_single_turn(
                    service,
                    prompt=args.prompt,
                    user_id=args.user_id,
                    thread_id=args.thread_id,
                    chat_model=args.model,
                )

            if args.command == "chat":
                return _run_chat(
                    service=service,
                    user_id=args.user_id,
                    thread_id=args.thread_id,
                    chat_model=args.model,
                )

            if args.command == "memories":
                for item in service.list_memories(user_id=args.user_id, limit=args.limit):
                    print(f"{item.get('memory_id', '')}\t{item.get('memory', '')}")
                return 0
        finally:
            service.lock_user(user_id=args.user_id)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if service is not None:
            service.close()

    return 1


def _run_single_turn(
    service: AtlasBackendService,
    *,
    prompt: str,
    user_id: str,
    thread_id: str,
    chat_model: str,
) -> int:
    started = service.start_chat(
        prompt=prompt,
        user_id=user_id,
        thread_id=thread_id,
        chat_model=chat_model,
    )
    result = _wait_for_run(service, str(started.get("run_id", "") or ""))
    print(result.get("answer", ""))
    return 0


def _run_chat(*, service: AtlasBackendService, user_id: str, thread_id: str, chat_model: str) -> int:
    print(f"Atlas chat started for user={user_id} thread={thread_id} model={chat_model}. Type 'exit' to stop.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            return 0

        started = service.start_chat(
            prompt=user_input,
            user_id=user_id,
            thread_id=thread_id,
            chat_model=chat_model,
        )
        result = _wait_for_run(service, str(started.get("run_id", "") or ""))
        print(f"Atlas: {result.get('answer', '')}")


def _unlock_cli_profile(
    service: AtlasBackendService,
    *,
    user_id: str,
    password_stdin: bool,
) -> None:
    profile = next(
        (
            item
            for item in service.list_users()
            if str(item.get("user_id", "") or "") == user_id
        ),
        None,
    )
    if profile is None:
        raise RuntimeError(f"User not found: {user_id}")

    password: str | None = None
    if bool(profile.get("locked")):
        if password_stdin:
            password = sys.stdin.readline().rstrip("\r\n")
        else:
            password = os.environ.get(_PROFILE_PASSWORD_ENV, "")
            if not password:
                password = getpass.getpass(f"Password for Atlas profile '{user_id}': ")
        if not password:
            raise RuntimeError("Password is required for this user.")
    service.unlock_user(user_id=user_id, password=password)


def _wait_for_run(service: AtlasBackendService, run_id: str) -> dict[str, object]:
    if not run_id:
        raise RuntimeError("Atlas did not return a run id.")
    while True:
        artifact = service.get_run(run_id)
        status = str(artifact.get("status", "") or "")
        if status == "completed":
            return artifact
        if status == "failed":
            raise RuntimeError(str(artifact.get("error", "") or "Atlas run failed."))
        time.sleep(_RUN_POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
