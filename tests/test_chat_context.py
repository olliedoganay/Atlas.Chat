import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from atlas_local.graph.context import GraphContext
from atlas_local.graph.nodes import _build_answer_messages, _latest_user_text


class ChatContextTests(unittest.TestCase):
    def test_cross_chat_memory_injects_minimal_context(self) -> None:
        state = {
            "messages": [HumanMessage(content="What is my name?")],
            "retrieved_memories": ["name: Atlas Tester"],
        }
        context = GraphContext(
            user_id="u1",
            thread_id="main",
            session_id="u1__main",
            chat_model="test-model",
            chat_temperature=0.2,
            cross_chat_memory=True,
        )

        messages = _build_answer_messages(state=state, runtime_context=context)

        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIn("Relevant persistent memories", str(messages[0].content))

    def test_cross_chat_memory_disabled_keeps_raw_messages(self) -> None:
        user_message = HumanMessage(content="What is my name?")
        state = {
            "messages": [user_message],
            "retrieved_memories": ["name: Atlas Tester"],
        }
        context = GraphContext(
            user_id="u1",
            thread_id="main",
            session_id="u1__main",
            chat_model="test-model",
            chat_temperature=0.2,
            cross_chat_memory=False,
        )

        messages = _build_answer_messages(state=state, runtime_context=context)

        self.assertEqual(messages, [user_message])

    def test_answer_prompt_template_can_be_injected(self) -> None:
        state = {
            "messages": [HumanMessage(content="What is my name?")],
            "retrieved_memories": ["name: Atlas Tester"],
        }
        context = GraphContext(
            user_id="u1",
            thread_id="main",
            session_id="u1__main",
            chat_model="test-model",
            chat_temperature=0.2,
            cross_chat_memory=True,
        )

        messages = _build_answer_messages(
            state=state,
            runtime_context=context,
            answer_prompt_template="User ID: {user_id}\nRetrieved memories:\n{memory_context}",
        )

        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIn("User ID: u1", str(messages[0].content))
        self.assertIn("name: Atlas Tester", str(messages[0].content))
        self.assertEqual(messages[-1].content, "What is my name?")

    def test_default_answer_prompt_describes_runtime_context_without_hard_denials(self) -> None:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "answer.md"
        prompt = prompt_path.read_text(encoding="utf-8")

        self.assertIn("Runtime note", prompt)
        self.assertIn("conversation history, summaries, retrieved memories, attachments, or tool/run outputs", prompt)
        self.assertIn("one self-contained HTML document", prompt)
        self.assertIn("immediate offline preview", prompt)
        self.assertIn("consistency pass for imports, names, and documented library APIs", prompt)
        self.assertIn("Do not describe code as tested, verified, or ready-to-run", prompt)
        self.assertNotIn("do not claim", prompt)
        self.assertNotIn("do not have direct", prompt)

    def test_thread_summary_replaces_compacted_prefix_in_prompt(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="first"),
                HumanMessage(content="second"),
                HumanMessage(content="latest question"),
            ],
            "thread_summary": "- user asked about earlier setup",
            "compacted_message_count": 2,
        }
        context = GraphContext(
            user_id="u1",
            thread_id="main",
            session_id="u1__main",
            chat_model="test-model",
            chat_temperature=0.2,
            cross_chat_memory=False,
            effective_context_window=512,
        )

        messages = _build_answer_messages(state=state, runtime_context=context)

        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIn("Conversation summary from earlier in this thread", str(messages[0].content))
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1].content, "latest question")

    def test_prompt_window_can_use_exact_message_token_counter(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="first"),
                HumanMessage(content="second"),
                HumanMessage(content="latest question"),
            ],
        }
        context = GraphContext(
            user_id="u1",
            thread_id="main",
            session_id="u1__main",
            chat_model="test-model",
            chat_temperature=0.2,
            cross_chat_memory=False,
            effective_context_window=1200,
        )

        messages = _build_answer_messages(
            state=state,
            runtime_context=context,
            token_counter=lambda batch: len(batch) * 400,
        )

        self.assertEqual([message.content for message in messages], ["second", "latest question"])

    def test_prompt_window_reserves_answer_prompt_memory_and_summary_together(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="older question"),
                HumanMessage(content="latest question"),
            ],
            "retrieved_memories": ["name: Atlas Tester"],
            "thread_summary": "- earlier decision",
        }
        context = GraphContext(
            user_id="u1",
            thread_id="main",
            session_id="u1__main",
            chat_model="test-model",
            chat_temperature=0.2,
            cross_chat_memory=True,
            effective_context_window=1600,
        )

        messages = _build_answer_messages(
            state=state,
            runtime_context=context,
            answer_prompt_template="Follow the Atlas answer rules.",
            token_counter=lambda batch: len(batch) * 250,
        )

        self.assertEqual(
            [message.content for message in messages[-1:]],
            ["latest question"],
        )
        self.assertEqual(len(messages), 4)

    def test_prompt_window_counts_fitting_history_in_one_batch(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="first"),
                HumanMessage(content="second"),
                HumanMessage(content="latest question"),
            ],
        }
        context = GraphContext(
            user_id="u1",
            thread_id="main",
            session_id="u1__main",
            chat_model="test-model",
            chat_temperature=0.2,
            cross_chat_memory=False,
            effective_context_window=4096,
        )
        counted_batch_sizes: list[int] = []

        messages = _build_answer_messages(
            state=state,
            runtime_context=context,
            token_counter=lambda batch: counted_batch_sizes.append(len(batch)) or len(batch) * 100,
        )

        self.assertEqual([message.content for message in messages], ["first", "second", "latest question"])
        self.assertEqual(counted_batch_sizes, [3])

    def test_prompt_window_skips_memories_that_do_not_fit_actual_budget(
        self,
    ) -> None:
        state = {
            "messages": [
                HumanMessage(content="older context " * 200),
                HumanMessage(content="latest question"),
            ],
            "retrieved_memories": [
                "oversized-memory " * 5_000,
                "short memory",
            ],
            "thread_summary": "- important earlier decision",
        }
        context = GraphContext(
            user_id="u1",
            thread_id="main",
            session_id="u1__main",
            chat_model="test-model",
            chat_temperature=0.2,
            cross_chat_memory=True,
            effective_context_window=2048,
        )

        def count_tokens(batch):
            return sum(len(str(message.content)) // 4 + 8 for message in batch)

        messages = _build_answer_messages(
            state=state,
            runtime_context=context,
            answer_prompt_template=(
                "Follow the Atlas answer rules.\n"
                "Relevant memories:\n{memory_context}"
            ),
            token_counter=count_tokens,
        )

        rendered = "\n".join(str(message.content) for message in messages)
        self.assertNotIn("oversized-memory", rendered)
        self.assertIn("short memory", rendered)
        self.assertIn("important earlier decision", rendered)
        self.assertEqual(messages[-1].content, "latest question")
        self.assertLessEqual(
            count_tokens(messages) + 64,
            int(2048 * 0.72),
        )

    def test_prompt_window_rejects_latest_request_that_cannot_fit(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="x" * 10_000),
            ],
        }
        context = GraphContext(
            user_id="u1",
            thread_id="main",
            session_id="u1__main",
            chat_model="test-model",
            chat_temperature=0.2,
            cross_chat_memory=False,
            effective_context_window=2048,
        )

        with self.assertRaisesRegex(RuntimeError, "latest request is too large"):
            _build_answer_messages(
                state=state,
                runtime_context=context,
                answer_prompt_template="Follow the Atlas answer rules.",
                token_counter=lambda batch: sum(
                    len(str(message.content)) // 4 + 8
                    for message in batch
                ),
            )

    def test_latest_user_text_ignores_image_blocks(self) -> None:
        state = {
            "messages": [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": "data:image/png;base64,AAAA"},
                    ]
                )
            ]
        }

        self.assertEqual(_latest_user_text(state), "Describe this image")


if __name__ == "__main__":
    unittest.main()
