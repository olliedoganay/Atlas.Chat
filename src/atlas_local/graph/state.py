from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_summary: str
    compacted_message_count: int
    detected_context_window: int
    timeline_events: list[dict[str, Any]]
    retrieved_memories: list[str]
    update_candidates: list[dict[str, Any]]
    persisted_memories: list[dict[str, Any]]
    memory_persistence_warnings: list[dict[str, Any]]
    answer: str
