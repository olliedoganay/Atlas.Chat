from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import queue
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Path as ApiPath, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send
import uvicorn

from .api_service import AtlasBackendService
from .code_runner import (
    CLIENT_LANGUAGES,
    LANGUAGES,
    docker_status,
    get_runner,
    prepare_runner_runtime,
    python_gui_runtime_status,
    resolve_language,
    runner_activity_status,
    supported_languages,
)
from .local_provider import normalize_local_provider_base_url
from .runtime import configure_console
from .run_contract import TERMINAL_EVENT_TYPES
from .version import atlas_version


MAX_IDENTIFIER_LENGTH = 128
MAX_PASSWORD_LENGTH = 1024
MAX_PROMPT_LENGTH = 200_000
MAX_MEMORY_LENGTH = 20_000
MAX_THREAD_TITLE_LENGTH = 200
MAX_MODEL_NAME_LENGTH = 200
MAX_CODE_LENGTH = 1_000_000
MAX_ATTACHMENTS = 8
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_TOTAL_ATTACHMENT_BYTES = 25 * 1024 * 1024
MAX_HTTP_REQUEST_BYTES = 40 * 1024 * 1024
MAX_SEARCH_QUERY_LENGTH = 1000
IDENTIFIER_PATTERN = (
    r"^[^./\\\s\x00-\x1f\x7f]"
    r"(?:[^/\\\x00-\x1f\x7f]*[^/\\\s\x00-\x1f\x7f])?$"
)
MODEL_NAME_PATTERN = (
    r"^[^\s\x00-\x1f\x7f](?:[^\x00-\x1f\x7f]*[^\s\x00-\x1f\x7f])?$"
)
IdentifierPath = Annotated[
    str,
    ApiPath(
        min_length=1,
        max_length=MAX_IDENTIFIER_LENGTH,
        pattern=IDENTIFIER_PATTERN,
    ),
]


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max(1, int(max_bytes))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        received_bytes = 0
        buffered_messages: list[Message] = []
        while True:
            message = await receive()
            buffered_messages.append(message)
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content={"detail": "Atlas request body is too large."},
                    )
                    await response(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break

        message_index = 0

        async def replay_receive() -> Message:
            nonlocal message_index
            if message_index < len(buffered_messages):
                message = buffered_messages[message_index]
                message_index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)


class AttachmentRequest(BaseModel):
    name: str = Field(default="attachment", min_length=1, max_length=255)
    media_type: str = Field(default="application/octet-stream", min_length=1, max_length=128)
    kind: Literal["image", "file"] | None = None
    data_url: str | None = Field(default=None, max_length=14_000_000)
    text_content: str | None = Field(default=None, max_length=500_000)
    byte_size: int | None = Field(default=None, ge=0, le=MAX_ATTACHMENT_BYTES)


class PromptRequest(BaseModel):
    prompt: str = Field(default="", max_length=MAX_PROMPT_LENGTH)
    user_id: str = Field(
        ..., min_length=1, max_length=MAX_IDENTIFIER_LENGTH, pattern=IDENTIFIER_PATTERN
    )
    thread_id: str = Field(
        ..., min_length=1, max_length=MAX_IDENTIFIER_LENGTH, pattern=IDENTIFIER_PATTERN
    )
    chat_model: str | None = Field(default=None, max_length=MAX_MODEL_NAME_LENGTH)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_mode: Literal["off", "on", "low", "medium", "high"] | None = None
    thread_title: str | None = Field(default=None, max_length=MAX_THREAD_TITLE_LENGTH)
    cross_chat_memory: bool = True
    auto_compact_long_chats: bool = True
    attachments: list[AttachmentRequest] = Field(default_factory=list, max_length=MAX_ATTACHMENTS)
    images: list[AttachmentRequest] = Field(default_factory=list, max_length=MAX_ATTACHMENTS)


class UserRequest(BaseModel):
    user_id: str = Field(
        ..., min_length=1, max_length=MAX_IDENTIFIER_LENGTH, pattern=IDENTIFIER_PATTERN
    )
    password: str | None = Field(default=None, max_length=MAX_PASSWORD_LENGTH)


class UnlockUserRequest(BaseModel):
    password: str | None = Field(default=None, max_length=MAX_PASSWORD_LENGTH)


class MemoryCreateRequest(BaseModel):
    user_id: str = Field(
        ..., min_length=1, max_length=MAX_IDENTIFIER_LENGTH, pattern=IDENTIFIER_PATTERN
    )
    text: str = Field(..., min_length=1, max_length=MAX_MEMORY_LENGTH)


class ModelContextWindowRequest(BaseModel):
    context_window: int | None = Field(default=None, ge=1024, le=262144)


class ModelUnloadRequest(BaseModel):
    model: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MODEL_NAME_LENGTH,
        pattern=MODEL_NAME_PATTERN,
    )


class ModelPullRequest(BaseModel):
    model: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MODEL_NAME_LENGTH,
        pattern=MODEL_NAME_PATTERN,
    )


class ProviderSettingsRequest(BaseModel):
    provider: Literal[
        "ollama",
        "lmstudio",
        "llamacpp",
        "vllm",
        "localai",
        "openai-compatible",
    ]
    base_url: str = Field(default="", max_length=2048)
    api_key: str | None = Field(default=None, max_length=4096)
    preserve_existing_key: bool = True


class ThreadTitleRequest(BaseModel):
    user_id: str = Field(
        ..., min_length=1, max_length=MAX_IDENTIFIER_LENGTH, pattern=IDENTIFIER_PATTERN
    )
    title: str = Field(..., min_length=1, max_length=MAX_THREAD_TITLE_LENGTH)


class ThreadBranchRequest(BaseModel):
    user_id: str = Field(
        ..., min_length=1, max_length=MAX_IDENTIFIER_LENGTH, pattern=IDENTIFIER_PATTERN
    )
    after_message_count: int = Field(..., ge=0, le=1_000_000)


class ResetThreadRequest(BaseModel):
    thread_id: str = Field(
        ..., min_length=1, max_length=MAX_IDENTIFIER_LENGTH, pattern=IDENTIFIER_PATTERN
    )
    user_id: str = Field(
        ..., min_length=1, max_length=MAX_IDENTIFIER_LENGTH, pattern=IDENTIFIER_PATTERN
    )


class ResetAllRequest(BaseModel):
    confirmation: str = Field(..., max_length=64)


class RunnerExecRequest(BaseModel):
    language: str = Field(..., min_length=1, max_length=64)
    code: str = Field(..., min_length=1, max_length=MAX_CODE_LENGTH)


DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "tauri://localhost",
)


def create_api_app(
    service: AtlasBackendService | None = None,
    *,
    request_server_shutdown: Callable[[], None] | None = None,
) -> FastAPI:
    managed_service = service
    required_token = os.environ.get("ATLAS_INSTANCE_TOKEN", "").strip()
    allow_insecure_localhost = _allow_insecure_localhost()
    allowed_origins = _allowed_origins()
    shutdown_lock = threading.Lock()
    shutdown_started = False

    if not required_token and not allow_insecure_localhost and service is None:
        raise RuntimeError(
            "ATLAS_INSTANCE_TOKEN is required for the Atlas API. "
            "Set ATLAS_ALLOW_INSECURE_LOCALHOST=1 only for explicit local development overrides."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal managed_service
        if managed_service is None:
            managed_service = AtlasBackendService.create()
        app.state.service = managed_service

        try:
            yield
        finally:
            if service is None and managed_service is not None:
                managed_service.close()

    app = FastAPI(title="Atlas API", version=atlas_version(), lifespan=lifespan)
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_HTTP_REQUEST_BYTES)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Atlas-Instance-Token"],
    )
    if managed_service is not None:
        app.state.service = managed_service

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(_request: Request, exc: RuntimeError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.middleware("http")
    async def require_instance_token(request: Request, call_next):
        origin = request.headers.get("origin", "").strip()
        if origin and origin not in allowed_origins:
            return JSONResponse(status_code=403, content={"detail": "Atlas backend rejected this origin."})
        if request.method.upper() == "OPTIONS":
            return await call_next(request)
        if required_token:
            provided = request.headers.get("x-atlas-instance-token", "").strip()
            if not secrets.compare_digest(provided, required_token):
                return JSONResponse(status_code=401, content={"detail": "Atlas backend identity check failed."})
        content_length = request.headers.get("content-length", "").strip()
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
            if declared_length < 0:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
            if declared_length > MAX_HTTP_REQUEST_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Atlas request body is too large."})
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    def backend() -> AtlasBackendService:
        return app.state.service

    def begin_backend_shutdown_and_server() -> None:
        try:
            begin_shutdown = getattr(backend(), "begin_shutdown", None)
            if callable(begin_shutdown):
                begin_shutdown()
            else:
                # Preserve compatibility for injected/test services that only
                # expose the original close contract.
                backend().close()
        finally:
            if request_server_shutdown is not None:
                request_server_shutdown()

    @app.post("/admin/prepare-shutdown")
    def prepare_shutdown() -> dict[str, Any]:
        nonlocal shutdown_started
        with shutdown_lock:
            if shutdown_started:
                return {"status": "shutdown-already-scheduled", "scheduled": False}
            shutdown_started = True
        try:
            threading.Thread(
                target=begin_backend_shutdown_and_server,
                name="atlas-backend-shutdown",
                daemon=True,
            ).start()
        except Exception:
            with shutdown_lock:
                shutdown_started = False
            raise
        return {"status": "shutdown-scheduled", "scheduled": True}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return backend().health()

    @app.get("/status")
    def status() -> dict[str, Any]:
        return backend().status()

    @app.get("/models")
    def models() -> dict[str, Any]:
        return backend().list_models()

    @app.get("/settings/provider")
    def provider_settings() -> dict[str, Any]:
        return backend().get_provider_settings()

    @app.put("/settings/provider")
    def update_provider_settings(request: ProviderSettingsRequest) -> dict[str, Any]:
        base_url = _validate_provider_base_url(request.base_url)
        return _handle_runtime(
            lambda: backend().save_provider_settings(
                provider=request.provider,
                base_url=base_url,
                api_key=request.api_key,
                preserve_existing_key=request.preserve_existing_key,
            )
        )

    @app.delete("/settings/provider/api-key")
    def delete_provider_api_key() -> dict[str, Any]:
        return _handle_runtime(backend().clear_provider_api_key)

    @app.post("/models/pulls")
    def start_model_pull(request: ModelPullRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().start_model_pull(model=request.model))

    @app.get("/models/pulls")
    def model_pulls() -> list[dict[str, Any]]:
        return backend().list_model_pulls()

    @app.get("/models/pulls/{pull_id}")
    def model_pull(pull_id: IdentifierPath) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().get_model_pull(pull_id))

    @app.delete("/models/pulls/{pull_id}")
    def cancel_model_pull(pull_id: IdentifierPath) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().cancel_model_pull(pull_id))

    @app.patch("/models/context-window")
    def set_ollama_context_window(request: ModelContextWindowRequest) -> dict[str, Any]:
        return _handle_runtime(
            lambda: backend().set_ollama_context_window(context_window=request.context_window)
        )

    @app.post("/models/unload")
    def unload_ollama_model(request: ModelUnloadRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().unload_ollama_model(model=request.model))

    @app.get("/discovery")
    def discovery() -> dict[str, Any]:
        return backend().discovery()

    @app.get("/users")
    def users() -> list[dict[str, Any]]:
        return backend().list_users()

    @app.get("/memories")
    def memories(
        user_id: str = Query(
            ...,
            min_length=1,
            max_length=MAX_IDENTIFIER_LENGTH,
            pattern=IDENTIFIER_PATTERN,
        ),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        return backend().list_memories(user_id=user_id, limit=limit)

    @app.post("/memories")
    def create_memory(request: MemoryCreateRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().add_memory(user_id=request.user_id, text=request.text))

    @app.delete("/memories/{memory_id}")
    def delete_memory(
        memory_id: IdentifierPath,
        user_id: str = Query(
            ...,
            min_length=1,
            max_length=MAX_IDENTIFIER_LENGTH,
            pattern=IDENTIFIER_PATTERN,
        ),
    ) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().delete_memory(user_id=user_id, memory_id=memory_id))

    @app.post("/users")
    def create_user(request: UserRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().create_user(user_id=request.user_id, password=request.password))

    @app.post("/users/{user_id}/unlock")
    def unlock_user(user_id: IdentifierPath, request: UnlockUserRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().unlock_user(user_id=user_id, password=request.password))

    @app.post("/users/{user_id}/lock")
    def lock_user(user_id: IdentifierPath) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().lock_user(user_id=user_id))

    @app.delete("/users/{user_id}")
    def delete_user(
        user_id: IdentifierPath,
        confirmation_user_id: str = Query(
            ...,
            min_length=1,
            max_length=MAX_IDENTIFIER_LENGTH,
            pattern=IDENTIFIER_PATTERN,
        ),
    ) -> dict[str, Any]:
        return _handle_runtime(
            lambda: backend().reset_user(user_id=user_id, confirmation_user_id=confirmation_user_id)
        )

    @app.get("/threads")
    def threads(
        user_id: str | None = Query(
            default=None,
            min_length=1,
            max_length=MAX_IDENTIFIER_LENGTH,
            pattern=IDENTIFIER_PATTERN,
        ),
    ) -> list[dict[str, Any]]:
        return backend().list_threads(user_id=user_id)

    @app.get("/search")
    def search_threads(
        user_id: str = Query(
            ...,
            min_length=1,
            max_length=MAX_IDENTIFIER_LENGTH,
            pattern=IDENTIFIER_PATTERN,
        ),
        q: str = Query(..., min_length=1, max_length=MAX_SEARCH_QUERY_LENGTH),
        current_thread_id: str | None = Query(
            default=None,
            min_length=1,
            max_length=MAX_IDENTIFIER_LENGTH,
            pattern=IDENTIFIER_PATTERN,
        ),
        limit: int = Query(default=8, ge=1, le=25),
    ) -> dict[str, Any]:
        return _handle_runtime(
            lambda: backend().search_threads(
                user_id=user_id,
                query=q,
                current_thread_id=current_thread_id,
                limit=limit,
            )
        )

    @app.patch("/threads/{thread_id}/title")
    def rename_thread(thread_id: IdentifierPath, request: ThreadTitleRequest) -> dict[str, Any]:
        return _handle_runtime(
            lambda: backend().rename_thread(user_id=request.user_id, thread_id=thread_id, title=request.title)
        )

    @app.post("/threads/{thread_id}/duplicate")
    def duplicate_thread(thread_id: IdentifierPath, request: UserRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().duplicate_thread(user_id=request.user_id, thread_id=thread_id))

    @app.post("/threads/{thread_id}/branch")
    def branch_thread(thread_id: IdentifierPath, request: ThreadBranchRequest) -> dict[str, Any]:
        return _handle_runtime(
            lambda: backend().branch_thread(
                user_id=request.user_id,
                thread_id=thread_id,
                after_message_count=request.after_message_count,
            )
        )

    @app.post("/threads/{thread_id}/compact")
    def compact_thread(thread_id: IdentifierPath, request: UserRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().start_manual_compact(user_id=request.user_id, thread_id=thread_id))

    @app.get("/threads/{thread_id}/history")
    def thread_history(
        thread_id: IdentifierPath,
        user_id: str = Query(
            ...,
            min_length=1,
            max_length=MAX_IDENTIFIER_LENGTH,
            pattern=IDENTIFIER_PATTERN,
        ),
    ) -> list[dict[str, Any]]:
        return _handle_runtime(lambda: backend().get_thread_history(user_id=user_id, thread_id=thread_id))

    @app.get("/threads/{thread_id}/context")
    def thread_context_usage(
        thread_id: IdentifierPath,
        user_id: str = Query(
            ...,
            min_length=1,
            max_length=MAX_IDENTIFIER_LENGTH,
            pattern=IDENTIFIER_PATTERN,
        ),
    ) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().get_thread_context_usage(user_id=user_id, thread_id=thread_id))

    @app.get("/threads/{thread_id}/runs")
    def thread_runs(
        thread_id: IdentifierPath,
        user_id: str = Query(
            ...,
            min_length=1,
            max_length=MAX_IDENTIFIER_LENGTH,
            pattern=IDENTIFIER_PATTERN,
        ),
    ) -> list[dict[str, Any]]:
        return _handle_runtime(lambda: backend().list_thread_runs(user_id=user_id, thread_id=thread_id))

    @app.get("/runs/{run_id}")
    def run_details(run_id: IdentifierPath) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().get_run(run_id))

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: IdentifierPath) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().cancel_run(run_id))

    @app.post("/chat")
    def chat(request: PromptRequest) -> dict[str, Any]:
        attachments = request.attachments or request.images
        if not request.prompt.strip() and not attachments:
            raise HTTPException(status_code=400, detail="Prompt or attachment is required.")
        attachment_payloads = [item.model_dump(exclude_none=True) for item in attachments]
        _validate_attachment_budget(attachment_payloads)
        return _handle_runtime(
            lambda: backend().start_chat(
                prompt=request.prompt,
                user_id=request.user_id,
                thread_id=request.thread_id,
                chat_model=request.chat_model,
                temperature=request.temperature,
                reasoning_mode=request.reasoning_mode,
                thread_title=request.thread_title,
                cross_chat_memory=request.cross_chat_memory,
                auto_compact_long_chats=request.auto_compact_long_chats,
                attachments=attachment_payloads,
            )
        )

    @app.get("/chat/stream/{run_id}")
    async def chat_stream(run_id: IdentifierPath) -> StreamingResponse:
        return _streaming_response(backend(), run_id)

    @app.get("/compact/stream/{run_id}")
    async def compact_stream(run_id: IdentifierPath) -> StreamingResponse:
        return _streaming_response(backend(), run_id)

    @app.post("/admin/reset/thread")
    def reset_thread(request: ResetThreadRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().reset_thread(thread_id=request.thread_id, user_id=request.user_id))

    @app.post("/admin/reset/all")
    def reset_all(request: ResetAllRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().reset_all(confirmation=request.confirmation))

    @app.get("/runner/status")
    def runner_status() -> dict[str, Any]:
        status_payload = docker_status()
        status_payload.update(runner_activity_status())
        status_payload["supported_languages"] = supported_languages()
        status_payload["server_languages"] = sorted(LANGUAGES.keys())
        status_payload["client_languages"] = sorted(CLIENT_LANGUAGES)
        return status_payload

    @app.get("/runner/runtime/python-gui")
    def runner_python_gui_runtime() -> dict[str, Any]:
        return python_gui_runtime_status()

    @app.post("/runner/runtime/prepare")
    def runner_runtime_prepare(request: RunnerExecRequest) -> dict[str, Any]:
        try:
            return prepare_runner_runtime(request.language, request.code)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/runner/exec")
    def runner_exec(request: RunnerExecRequest) -> dict[str, Any]:
        resolved = resolve_language(request.language)
        if not resolved:
            raise HTTPException(status_code=400, detail=f"Language '{request.language}' is not supported.")
        if resolved in CLIENT_LANGUAGES:
            raise HTTPException(status_code=400, detail="HTML runs in the client sandbox, not via the backend runner.")
        docker_state = docker_status()
        if not docker_state.get("available"):
            raise HTTPException(status_code=503, detail=docker_state.get("reason", "Docker is unavailable."))
        try:
            return get_runner().start(language=resolved, code=request.code)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/runner/stream/{run_id}")
    async def runner_stream(run_id: IdentifierPath) -> StreamingResponse:
        return _runner_streaming_response(run_id)

    @app.post("/runner/stop/{run_id}")
    def runner_stop(run_id: IdentifierPath) -> dict[str, Any]:
        return get_runner().stop(run_id)

    @app.post("/runner/shutdown")
    def runner_shutdown() -> dict[str, str]:
        get_runner().shutdown()
        return {"status": "stopped"}

    return app


def _streaming_response(service: AtlasBackendService, run_id: str) -> StreamingResponse:
    try:
        service.get_run(run_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def generator():
        subscriber = service.subscribe(run_id)
        emitted_fingerprints: set[str] = set()
        emitted_stream_text = {"token": "", "thinking_token": ""}
        next_sequence = 1
        terminal_emitted = False

        def ordered_unseen_events(artifact: dict[str, Any]) -> list[dict[str, Any]]:
            nonlocal next_sequence, terminal_emitted
            unseen: list[dict[str, Any]] = []
            artifact_stream_text = {"token": "", "thinking_token": ""}
            for event in artifact.get("events", []):
                event_type = str(event.get("type", "") or "")
                stream_parts: list[tuple[str, str]] = []
                payload = event.get("payload", {})
                if event_type in emitted_stream_text and isinstance(payload, dict):
                    stream_parts.append(
                        (event_type, str(payload.get("text", "") or ""))
                    )
                elif event_type == "stream_snapshot" and isinstance(payload, dict):
                    stream_parts.extend(
                        (
                            (
                                "thinking_token",
                                str(payload.get("thinking_text", "") or ""),
                            ),
                            (
                                "token",
                                str(payload.get("answer_text", "") or ""),
                            ),
                        )
                    )
                for stream_type, text in stream_parts:
                    artifact_stream_text[stream_type] += text

                sequence = _event_sequence(event)
                sequence_end: int | None = None
                if sequence is not None:
                    sequence_end = _event_sequence_end(event) or sequence
                    if sequence_end < next_sequence:
                        continue
                    if sequence > next_sequence:
                        break
                    resolved_sequence = next_sequence
                    next_sequence = sequence_end + 1
                else:
                    fingerprint = _event_fingerprint(event)
                    if fingerprint in emitted_fingerprints:
                        continue
                    emitted_fingerprints.add(fingerprint)
                    resolved_sequence = None

                if stream_parts:
                    missing_parts: list[tuple[str, str]] = []
                    for stream_type, _text in stream_parts:
                        target_text = artifact_stream_text[stream_type]
                        already_emitted = emitted_stream_text[stream_type]
                        if not target_text.startswith(already_emitted):
                            continue
                        missing_text = target_text[len(already_emitted) :]
                        if not missing_text:
                            continue
                        missing_parts.append((stream_type, missing_text))

                    part_sequence = resolved_sequence
                    for part_index, (stream_type, missing_text) in enumerate(
                        missing_parts
                    ):
                        replay_event = {
                            "type": stream_type,
                            "timestamp": str(event.get("timestamp", "") or ""),
                            "payload": {"text": missing_text},
                        }
                        if part_sequence is not None and sequence_end is not None:
                            remaining_parts = len(missing_parts) - part_index
                            remaining_width = sequence_end - part_sequence + 1
                            if remaining_width >= remaining_parts:
                                part_end = (
                                    sequence_end
                                    if remaining_parts == 1
                                    else part_sequence
                                )
                                replay_event["sequence"] = part_sequence
                                if part_end > part_sequence:
                                    replay_event["sequence_end"] = part_end
                                part_sequence = part_end + 1
                            else:
                                # Legacy mixed snapshots could occupy only one
                                # logical sequence. Keep the additional channel
                                # as an unsequenced compatibility event rather
                                # than assigning two events the same sequence.
                                part_sequence = None
                        emitted_stream_text[stream_type] += missing_text
                        unseen.append(replay_event)
                    continue

                if event_type in TERMINAL_EVENT_TYPES:
                    terminal_emitted = True
                unseen.append(event)
            return unseen

        def missing_terminal_event(artifact: dict[str, Any]) -> dict[str, Any] | None:
            status = str(artifact.get("status", "") or "")
            if status == "completed":
                return {
                    "type": "run_completed",
                    "timestamp": str(
                        artifact.get("completed_at")
                        or artifact.get("started_at")
                        or ""
                    ),
                    "payload": {
                        "answer": str(artifact.get("answer", "") or ""),
                    },
                    "sequence": next_sequence,
                }
            if status == "failed":
                return {
                    "type": "run_failed",
                    "timestamp": str(
                        artifact.get("completed_at")
                        or artifact.get("started_at")
                        or ""
                    ),
                    "payload": {
                        "error": str(artifact.get("error", "") or ""),
                    },
                    "sequence": next_sequence,
                }
            return None

        try:
            # Subscribe before replaying history. Events appended between the
            # initial existence check and this subscription are then present in
            # the reconciled artifact, while later events are queued.
            artifact = service.get_run(run_id)
            for event in ordered_unseen_events(artifact):
                yield _format_sse(event)

            if terminal_emitted:
                return
            terminal_fallback = missing_terminal_event(artifact)
            if terminal_fallback is not None:
                yield _format_sse(terminal_fallback)
                return

            while True:
                try:
                    event = await asyncio.to_thread(subscriber.get, True, 5.0)
                    sequence = _event_sequence(event)
                    if sequence is not None and sequence > next_sequence:
                        refreshed = service.get_run(run_id)
                        for reconciled in ordered_unseen_events(refreshed):
                            yield _format_sse(reconciled)
                        if terminal_emitted:
                            return
                    if sequence is not None:
                        if sequence < next_sequence:
                            continue
                        if sequence > next_sequence:
                            continue
                        next_sequence += 1
                    else:
                        # Legacy/custom service events may not carry sequence
                        # metadata. Reconcile persisted history before emitting
                        # them so a bounded queue cannot reorder older events.
                        refreshed = service.get_run(run_id)
                        for reconciled in ordered_unseen_events(refreshed):
                            yield _format_sse(reconciled)
                        if terminal_emitted:
                            return
                        fingerprint = _event_fingerprint(event)
                        if fingerprint in emitted_fingerprints:
                            continue
                        emitted_fingerprints.add(fingerprint)
                    event_type = str(event.get("type", "") or "")
                    if event_type in emitted_stream_text:
                        payload = event.get("payload", {})
                        if isinstance(payload, dict):
                            emitted_stream_text[event_type] += str(
                                payload.get("text", "") or ""
                            )
                    yield _format_sse(event)
                    if event.get("type") in TERMINAL_EVENT_TYPES:
                        return
                except queue.Empty:
                    refreshed = service.get_run(run_id)
                    for event in ordered_unseen_events(refreshed):
                        yield _format_sse(event)
                    if terminal_emitted:
                        return
                    terminal_fallback = missing_terminal_event(refreshed)
                    if terminal_fallback is not None:
                        yield _format_sse(terminal_fallback)
                        return
        finally:
            service.unsubscribe(run_id, subscriber)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _format_sse(event: dict[str, Any]) -> str:
    return f"event: {event.get('type', 'message')}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def _event_fingerprint(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _event_sequence(event: dict[str, Any]) -> int | None:
    value = event.get("sequence")
    if type(value) is int and value >= 1:
        return value
    return None


def _event_sequence_end(event: dict[str, Any]) -> int | None:
    sequence = _event_sequence(event)
    if sequence is None:
        return None
    value = event.get("sequence_end")
    if type(value) is int and value >= sequence:
        return value
    return sequence


def _runner_streaming_response(run_id: str) -> StreamingResponse:
    runner = get_runner()
    try:
        history, subscriber, finished = runner.subscribe(run_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def generator():
        for event in history:
            yield _format_sse(event)
        if finished:
            return
        try:
            while True:
                try:
                    event = await asyncio.to_thread(subscriber.get, True, 5.0)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield _format_sse(event)
                if event.get("type") == "exit":
                    return
        finally:
            runner.unsubscribe(run_id, subscriber)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _handle_runtime(callback):
    try:
        return callback()
    except RuntimeError as exc:
        message = str(exc)
        status = 409 if "already running another task" in message.lower() else 400
        raise HTTPException(status_code=status, detail=message) from exc


def main(argv: list[str] | None = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Run the Atlas local backend API.")
    parser.add_argument("--host", default=os.environ.get("ATLAS_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ATLAS_API_PORT", "8765")))
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    if _allow_insecure_localhost() and not os.environ.get("ATLAS_INSTANCE_TOKEN", "").strip():
        if not _is_loopback_host(args.host):
            parser.error(
                "ATLAS_ALLOW_INSECURE_LOCALHOST can only be used with a loopback host."
            )
    server: uvicorn.Server | None = None

    def request_server_shutdown() -> None:
        if server is not None:
            server.should_exit = True

    app = create_api_app(request_server_shutdown=request_server_shutdown)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port))
    server.run()
    return 0


def _allow_insecure_localhost() -> bool:
    return os.environ.get("ATLAS_ALLOW_INSECURE_LOCALHOST", "").strip().lower() in {"1", "true", "yes", "on"}


def _allowed_origins() -> tuple[str, ...]:
    raw = os.environ.get("ATLAS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return DEFAULT_ALLOWED_ORIGINS
    values = tuple(origin.strip() for origin in raw.split(",") if origin.strip())
    return values or DEFAULT_ALLOWED_ORIGINS


def _is_loopback_host(host: str) -> bool:
    resolved = host.strip().strip("[]").lower()
    if resolved == "localhost":
        return True
    try:
        return ipaddress.ip_address(resolved).is_loopback
    except ValueError:
        return False


def _validate_provider_base_url(base_url: str) -> str:
    try:
        return normalize_local_provider_base_url(base_url, allow_empty=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_attachment_budget(attachments: list[dict[str, Any]]) -> None:
    total = 0
    for attachment in attachments:
        declared_size = int(attachment.get("byte_size", 0) or 0)
        data_url = str(attachment.get("data_url", "") or "")
        text_content = str(attachment.get("text_content", "") or "")
        estimated_size = declared_size
        if data_url:
            encoded = data_url.partition(",")[2] or data_url
            compact_encoded = "".join(encoded.split())
            padding = min(2, len(compact_encoded) - len(compact_encoded.rstrip("=")))
            decoded_size = max(0, (len(compact_encoded) * 3) // 4 - padding)
            estimated_size = max(estimated_size, decoded_size)
        if text_content:
            estimated_size = max(estimated_size, len(text_content.encode("utf-8")))
        if estimated_size > MAX_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Attachment '{attachment.get('name', 'attachment')}' exceeds the 10 MiB limit.",
            )
        total += estimated_size
    if total > MAX_TOTAL_ATTACHMENT_BYTES:
        raise HTTPException(status_code=413, detail="Attachments exceed the 25 MiB request limit.")


if __name__ == "__main__":
    raise SystemExit(main())
