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
