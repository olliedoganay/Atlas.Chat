from __future__ import annotations

import logging
import re
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from ..config import AppConfig
from ..llm import format_runtime_error
from ..memory.mem0_service import Mem0Service
from ..memory.models import MemoryCandidate, MemoryRecord
from ..memory.policy import fallback_local_memory_candidates_from_text
from ..providers.base import ChatModelProvider
from ..runtime import read_prompt
from .context import GraphContext
from .state import AgentState

LOGGER = logging.getLogger(__name__)
MAX_MEMORY_PERSISTENCE_ERROR_CHARS = 500


class GraphNodes:
    def __init__(
        self,
        config: AppConfig,
        llm_provider: ChatModelProvider,
        memory_service: Mem0Service,
    ):
        self.config = config
        self.llm_provider = llm_provider
        self.memory_service = memory_service
        self.answer_prompt_template = _read_optional_prompt(config, "answer.md")

    def retrieve_memories(
        self,
        state: AgentState,
        runtime: Runtime[GraphContext],
    ) -> dict[str, Any]:
        if not runtime.context.cross_chat_memory:
            return {"retrieved_memories": []}

        latest_user_message = _latest_user_text(state)
        if not latest_user_message:
            return {"retrieved_memories": []}

        retrieved: list[str] = []
        try:
            stored = self.memory_service.search(
                latest_user_message,
                user_id=runtime.context.user_id,
                limit=max(1, self.config.memory_top_k),
            )
            retrieved = [item.memory for item in stored if item.memory]
        except Exception as exc:  # pragma: no cover - integration path
            LOGGER.warning("Memory retrieval failed: %s", exc)
        return {"retrieved_memories": retrieved}

    def synthesize_answer(
        self,
        state: AgentState,
        runtime: Runtime[GraphContext],
    ) -> dict[str, Any]:
        messages = _build_answer_messages(
            state=state,
            runtime_context=runtime.context,
            token_counter=_provider_message_token_counter(self.llm_provider, runtime.context.chat_model),
            answer_prompt_template=self.answer_prompt_template,
        )
        try:
            response = self.llm_provider.chat(
                runtime.context.chat_model,
                temperature=runtime.context.chat_temperature,
                reasoning=runtime.context.reasoning_mode,
            ).invoke(messages)
        except Exception as exc:  # pragma: no cover - integration path
            raise format_runtime_error(self.config, exc, chat_model=runtime.context.chat_model) from exc

        answer = _finalize_answer_text(str(response.content))
        return {
            "messages": [AIMessage(content=answer)],
            "answer": answer,
        }

    def extract_updates(
        self,
        state: AgentState,
        runtime: Runtime[GraphContext],
    ) -> dict[str, Any]:
        if not runtime.context.cross_chat_memory:
            return {"update_candidates": []}

        latest_user_message = _latest_user_text(state)
        if not latest_user_message:
            return {"update_candidates": []}

        candidates = fallback_local_memory_candidates_from_text(latest_user_message)
        return {"update_candidates": [candidate.to_dict() for candidate in candidates]}

    def persist(
        self,
        state: AgentState,
        runtime: Runtime[GraphContext],
    ) -> dict[str, Any]:
        if not runtime.context.cross_chat_memory:
            return {
                "persisted_memories": [],
                "memory_persistence_warnings": [],
            }

        existing_memories: set[str] = set()
        try:
            existing_memories = {
                item.memory.strip().lower()
                for item in self.memory_service.list(user_id=runtime.context.user_id, limit=200)
                if item.memory.strip()
            }
        except Exception as exc:  # pragma: no cover - integration path
            LOGGER.warning("Could not load existing memories before persistence: %s", exc)

        persisted: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for payload in state.get("update_candidates", []):
            candidate = MemoryCandidate.from_dict(payload)
            storage_text = candidate.to_storage_text().strip()
            if not storage_text or storage_text.lower() in existing_memories:
                continue
            try:
                response = self.memory_service.add(
                    MemoryRecord(
                        claim_id=f"auto:{runtime.context.thread_id}",
                        text=storage_text,
                    ),
                    user_id=runtime.context.user_id,
                    metadata={
                        "source": "auto",
                        "kind": "extracted_memory",
                        "category": candidate.category,
                    },
                )
            except Exception as exc:  # pragma: no cover - integration path
                error_message = (
                    str(exc).strip() or "Atlas memory storage rejected the update."
                )[:MAX_MEMORY_PERSISTENCE_ERROR_CHARS]
                LOGGER.warning(
                    "Could not persist an extracted %s memory: %s",
                    candidate.category,
                    error_message,
                )
                warnings.append(
                    {
                        "category": candidate.category,
                        "error": error_message,
                    }
                )
                continue
            existing_memories.add(storage_text.lower())
            persisted.append(
                {
                    "memory": storage_text,
                    "category": candidate.category,
                    "response": response,
                }
            )
        return {
            "persisted_memories": persisted,
            "memory_persistence_warnings": warnings,
        }


def _latest_user_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            stored_prompt = str(message.additional_kwargs.get("atlas_user_prompt", "") or "").strip()
            if stored_prompt:
                return stored_prompt
            return _message_text(message.content)
    return ""


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(part for part in parts if part).strip()


MessageTokenCounter = Callable[[list[BaseMessage]], int]


def _build_answer_messages(
    *,
    state: AgentState,
    runtime_context: GraphContext,
    token_counter: MessageTokenCounter | None = None,
    answer_prompt_template: str | None = None,
) -> list[HumanMessage | AIMessage | SystemMessage]:
    messages = list(state.get("messages", []))
    compacted_count = max(
        0,
        min(
            int(state.get("compacted_message_count", 0) or 0),
            len(messages),
        ),
    )
    candidate_messages = messages[compacted_count:]
    effective_context_window = int(
        runtime_context.effective_context_window
        or state.get("detected_context_window")
        or 0
    )

    selected_memories = (
        [
            str(item)
            for item in state.get("retrieved_memories", [])
            if item
        ]
        if getattr(runtime_context, "cross_chat_memory", True)
        else []
    )
    summary_message = _thread_summary_message(state)

    def build_prefix(
        memories: list[str],
        summary: SystemMessage | None,
    ) -> list[SystemMessage]:
        answer_prompt_message = _answer_prompt_message(
            state=state,
            runtime_context=runtime_context,
            answer_prompt_template=answer_prompt_template,
            memories=memories,
        )
        memory_message = (
            None
            if answer_prompt_template
            and "{memory_context}" in answer_prompt_template
            else _memory_context_message(
                state=state,
                runtime_context=runtime_context,
                memories=memories,
            )
        )
        return [
            item
            for item in (
                answer_prompt_message,
                memory_message,
                summary,
            )
            if item is not None
        ]

    if effective_context_window <= 0:
        prefix = build_prefix(selected_memories, summary_message)
        return prefix + candidate_messages

    prompt_budget = max(256, int(effective_context_window * 0.72))
    latest_messages = candidate_messages[-1:] if candidate_messages else []
    prefix = build_prefix([], None)
    if prefix:
        minimum_tokens = (
            _count_messages_tokens(
                prefix + latest_messages,
                token_counter=token_counter,
            )
            + 64
        )
        if minimum_tokens > prompt_budget:
            raise RuntimeError(
                "The latest request is too large for the selected model's "
                "prompt budget. Shorten the message, remove or reduce "
                "attachments, or choose a model with a larger context window."
            )

    if summary_message is not None:
        summary_prefix = build_prefix([], summary_message)
        summary_tokens = (
            _count_messages_tokens(
                summary_prefix + latest_messages,
                token_counter=token_counter,
            )
            + 64
        )
        if summary_tokens > prompt_budget:
            summary_message = None

    bounded_memories: list[str] = []
    for memory in selected_memories:
        candidate_memories = [*bounded_memories, memory]
        candidate_prefix = build_prefix(
            candidate_memories,
            summary_message,
        )
        candidate_tokens = (
            _count_messages_tokens(
                candidate_prefix + latest_messages,
                token_counter=token_counter,
            )
            + 64
        )
        if candidate_tokens <= prompt_budget:
            bounded_memories.append(memory)

    prefix = build_prefix(bounded_memories, summary_message)
    recent_messages = _recent_prompt_messages(
        state=state,
        runtime_context=runtime_context,
        prefix_messages=prefix,
        token_counter=token_counter,
    )
    return prefix + recent_messages


def _read_optional_prompt(config: AppConfig, name: str) -> str:
    try:
        return read_prompt(config.prompt_dir, name)
    except OSError:
        return ""


def _answer_prompt_message(
    *,
    state: AgentState,
    runtime_context: GraphContext,
    answer_prompt_template: str | None,
    memories: list[str] | None = None,
) -> SystemMessage | None:
    template = (answer_prompt_template or "").strip()
    if not template:
        return None
    resolved_memories = (
        [item for item in state.get("retrieved_memories", []) if item]
        if memories is None
        else memories
    )
    values = {
        "user_id": runtime_context.user_id,
        "thread_id": runtime_context.thread_id,
        "memory_context": _format_list(resolved_memories),
        "world_context": "- none",
        "reasoning_context": "- none",
        "browser_context": "- none",
        "citation_context": "- none",
    }
    try:
        content = template.format_map(_PromptValues(values)).strip()
    except (KeyError, ValueError):
        content = template
    return SystemMessage(content=content) if content else None


class _PromptValues(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""


def _memory_context_message(
    *,
    state: AgentState,
    runtime_context: GraphContext,
    memories: list[str] | None = None,
) -> SystemMessage | None:
    if not getattr(runtime_context, "cross_chat_memory", True):
        return None

    resolved_memories = (
        [item for item in state.get("retrieved_memories", []) if item]
        if memories is None
        else memories
    )
    if not resolved_memories:
        return None
    return SystemMessage(
        content="Relevant persistent memories:\n"
        + _format_list(resolved_memories)
    )


def _thread_summary_message(state: AgentState) -> SystemMessage | None:
    summary = str(state.get("thread_summary", "") or "").strip()
    if not summary:
        return None
    return SystemMessage(content=f"Conversation summary from earlier in this thread:\n{summary}")


def _recent_prompt_messages(
    *,
    state: AgentState,
    runtime_context: GraphContext,
    prefix_messages: list[SystemMessage],
    token_counter: MessageTokenCounter | None,
) -> list[BaseMessage]:
    messages = list(state.get("messages", []))
    compacted_count = max(0, min(int(state.get("compacted_message_count", 0) or 0), len(messages)))
    candidate_messages = messages[compacted_count:]
    effective_context_window = int(
        runtime_context.effective_context_window
        or state.get("detected_context_window")
        or 0
    )
    if effective_context_window <= 0:
        return candidate_messages

    prompt_budget = max(256, int(effective_context_window * 0.72))
    reserved_tokens = (
        _count_messages_tokens(
            prefix_messages,
            token_counter=token_counter,
        )
        + 64
    )
    available_tokens = max(0, prompt_budget - reserved_tokens)

    if not candidate_messages:
        return []
    if _count_messages_tokens(candidate_messages, token_counter=token_counter) <= available_tokens:
        return candidate_messages
    latest_tokens = _count_messages_tokens(
        candidate_messages[-1:],
        token_counter=token_counter,
    )
    if latest_tokens > available_tokens:
        raise RuntimeError(
            "The latest request is too large for the selected model's prompt "
            "budget. Shorten the message, remove or reduce attachments, or "
            "choose a model with a larger context window."
        )

    earliest_fitting_index = len(candidate_messages) - 1
    low = 0
    high = len(candidate_messages) - 1
    while low <= high:
        midpoint = (low + high) // 2
        suffix_tokens = _count_messages_tokens(
            candidate_messages[midpoint:],
            token_counter=token_counter,
        )
        if suffix_tokens <= available_tokens:
            earliest_fitting_index = midpoint
            high = midpoint - 1
        else:
            low = midpoint + 1
    return candidate_messages[earliest_fitting_index:]


def _estimate_message_tokens(message: BaseMessage | None) -> int:
    if message is None:
        return 0
    return _estimate_content_tokens(message.content) + 8


def _count_messages_tokens(
    messages: list[BaseMessage | None],
    *,
    token_counter: MessageTokenCounter | None,
) -> int:
    items = [message for message in messages if message is not None]
    if not items:
        return 0
    if token_counter is not None:
        try:
            counted = int(token_counter(items))
            if counted >= 0:
                return counted
        except Exception:
            pass
    return sum(_estimate_message_tokens(message) for message in items)


def _estimate_content_tokens(content: Any) -> int:
    if isinstance(content, str):
        return max(1, len(content) // 4)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, str):
                total += max(1, len(item) // 4)
            elif isinstance(item, dict):
                item_type = str(item.get("type", "")).strip().lower()
                if item_type == "text":
                    total += max(1, len(str(item.get("text", ""))) // 4)
                elif item_type == "image_url":
                    total += 256
        return max(total, 1)
    return max(1, len(str(content)) // 4)


def _format_list(values: list[str]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- {item}" for item in values)


def _provider_message_token_counter(
    llm_provider: ChatModelProvider,
    model: str,
) -> MessageTokenCounter | None:
    counter = getattr(llm_provider, "count_message_tokens", None)
    if not callable(counter):
        return None
    return lambda messages: counter(model, messages)


def _strip_empty_sources_footer(answer: str) -> str:
    cleaned = re.sub(r"\n{0,2}Sources:\s*(-\s*)?(none|n/?a)\s*$", "", answer.strip(), flags=re.IGNORECASE)
    return cleaned.strip()


def _finalize_answer_text(answer: str) -> str:
    return _strip_empty_sources_footer(answer.strip())
