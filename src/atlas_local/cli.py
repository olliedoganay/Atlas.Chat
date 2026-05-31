from __future__ import annotations

import argparse
import sys

from .config import load_config
from .graph.builder import build_chat_application
from .runtime import configure_console


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Atlas local chat CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Run a single local turn through the graph.")
    ask_parser.add_argument("prompt", help="Prompt to send to the agent.")
    ask_parser.add_argument("--user-id", required=True)
    ask_parser.add_argument("--thread-id", default="default-thread")
    ask_parser.add_argument("--model", required=True, help="Local chat model to use.")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session.")
    chat_parser.add_argument("--user-id", required=True)
    chat_parser.add_argument("--thread-id", default="default-thread")
    chat_parser.add_argument("--model", required=True, help="Local chat model to use.")

    memories_parser = subparsers.add_parser("memories", help="List stored memories for a user.")
    memories_parser.add_argument("--user-id", required=True)
    memories_parser.add_argument("--limit", type=int, default=20)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_console()
    config = load_config()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        with build_chat_application(config) as app:
            if args.command == "ask":
                return _run_single_turn(
                    app,
                    prompt=args.prompt,
                    user_id=args.user_id,
                    thread_id=args.thread_id,
                    chat_model=args.model,
                )

            if args.command == "chat":
                return _run_chat(app=app, user_id=args.user_id, thread_id=args.thread_id, chat_model=args.model)

            if args.command == "memories":
                for item in app.list_memories(user_id=args.user_id, limit=args.limit):
                    print(f"{item.memory_id}\t{item.memory}")
                return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 1


def _run_single_turn(
    app,
    *,
    prompt: str,
    user_id: str,
    thread_id: str,
    chat_model: str,
) -> int:
    result = app.ask(prompt, user_id=user_id, thread_id=thread_id, chat_model=chat_model)
    print(result.get("answer", ""))
    return 0


def _run_chat(*, app, user_id: str, thread_id: str, chat_model: str) -> int:
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

        result = app.ask(user_input, user_id=user_id, thread_id=thread_id, chat_model=chat_model)
        print(f"Atlas: {result.get('answer', '')}")


if __name__ == "__main__":
    raise SystemExit(main())
