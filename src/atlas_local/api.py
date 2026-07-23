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
from typing import Any, Callable, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from .api_service import AtlasBackendService
from .code_runner import (
    CLIENT_LANGUAGES,
    LANGUAGES,
    docker_status,
    get_runner,
    resolve_language,
    supported_languages,
)
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
IDENTIFIER_PATTERN = r"^[^\x00-\x1f\x7f]+$"


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
    model: str = Field(..., min_length=1, max_length=MAX_MODEL_NAME_LENGTH)


class ModelPullRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=MAX_MODEL_NAME_LENGTH)


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
        if origin and not allow_insecure_localhost and origin not in allowed_origins:
            return JSONResponse(status_code=403, content={"detail": "Atlas backend rejected this origin."})
        if request.method.upper() == "OPTIONS":
            return await call_next(request)
        if required_token:
            provided = request.headers.get("x-atlas-instance-token", "").strip()
            if not secrets.compare_digest(provided, required_token):
                return JSONResponse(status_code=401, content={"detail": "Atlas backend identity check failed."})
        return await call_next(request)

    def backend() -> AtlasBackendService:
        return app.state.service

    def close_backend_and_server() -> None:
        try:
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
                target=close_backend_and_server,
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
    def model_pull(pull_id: str) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().get_model_pull(pull_id))

    @app.delete("/models/pulls/{pull_id}")
    def cancel_model_pull(pull_id: str) -> dict[str, Any]:
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
    def delete_memory(memory_id: str, user_id: str = Query(..., min_length=1)) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().delete_memory(user_id=user_id, memory_id=memory_id))

    @app.post("/users")
    def create_user(request: UserRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().create_user(user_id=request.user_id, password=request.password))

    @app.post("/users/{user_id}/unlock")
    def unlock_user(user_id: str, request: UnlockUserRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().unlock_user(user_id=user_id, password=request.password))

    @app.post("/users/{user_id}/lock")
    def lock_user(user_id: str) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().lock_user(user_id=user_id))

    @app.delete("/users/{user_id}")
    def delete_user(user_id: str, confirmation_user_id: str = Query(..., min_length=1)) -> dict[str, Any]:
        return _handle_runtime(
            lambda: backend().reset_user(user_id=user_id, confirmation_user_id=confirmation_user_id)
        )

    @app.get("/threads")
    def threads(user_id: str | None = Query(default=None)) -> list[dict[str, Any]]:
        return backend().list_threads(user_id=user_id)

    @app.get("/search")
    def search_threads(
        user_id: str = Query(..., min_length=1),
        q: str = Query(..., min_length=1),
        current_thread_id: str | None = Query(default=None),
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
    def rename_thread(thread_id: str, request: ThreadTitleRequest) -> dict[str, Any]:
        return _handle_runtime(
            lambda: backend().rename_thread(user_id=request.user_id, thread_id=thread_id, title=request.title)
        )

    @app.post("/threads/{thread_id}/duplicate")
    def duplicate_thread(thread_id: str, request: UserRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().duplicate_thread(user_id=request.user_id, thread_id=thread_id))

    @app.post("/threads/{thread_id}/branch")
    def branch_thread(thread_id: str, request: ThreadBranchRequest) -> dict[str, Any]:
        return _handle_runtime(
            lambda: backend().branch_thread(
                user_id=request.user_id,
                thread_id=thread_id,
                after_message_count=request.after_message_count,
            )
        )

    @app.post("/threads/{thread_id}/compact")
    def compact_thread(thread_id: str, request: UserRequest) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().start_manual_compact(user_id=request.user_id, thread_id=thread_id))

    @app.get("/threads/{thread_id}/history")
    def thread_history(thread_id: str, user_id: str | None = Query(default=None)) -> list[dict[str, Any]]:
        return _handle_runtime(lambda: backend().get_thread_history(user_id=user_id, thread_id=thread_id))

    @app.get("/threads/{thread_id}/context")
    def thread_context_usage(thread_id: str, user_id: str | None = Query(default=None)) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().get_thread_context_usage(user_id=user_id, thread_id=thread_id))

    @app.get("/threads/{thread_id}/runs")
    def thread_runs(thread_id: str, user_id: str | None = Query(default=None)) -> list[dict[str, Any]]:
        return _handle_runtime(lambda: backend().list_thread_runs(user_id=user_id, thread_id=thread_id))

    @app.get("/runs/{run_id}")
    def run_details(run_id: str) -> dict[str, Any]:
        return _handle_runtime(lambda: backend().get_run(run_id))

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
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
    async def chat_stream(run_id: str) -> StreamingResponse:
        return _streaming_response(backend(), run_id)

    @app.get("/compact/stream/{run_id}")
    async def compact_stream(run_id: str) -> StreamingResponse:
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
        status_payload["supported_languages"] = supported_languages()
        status_payload["server_languages"] = sorted(LANGUAGES.keys())
        status_payload["client_languages"] = sorted(CLIENT_LANGUAGES)
        return status_payload

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
    async def runner_stream(run_id: str) -> StreamingResponse:
        return _runner_streaming_response(run_id)

    @app.post("/runner/stop/{run_id}")
    def runner_stop(run_id: str) -> dict[str, Any]:
        return get_runner().stop(run_id)

    @app.post("/runner/shutdown")
    def runner_shutdown() -> dict[str, str]:
        get_runner().shutdown()
        return {"status": "stopped"}

    return app


def _streaming_response(service: AtlasBackendService, run_id: str) -> StreamingResponse:
    try:
        artifact = service.get_run(run_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def generator():
        emitted = 0
        for event in artifact.get("events", []):
            emitted += 1
            yield _format_sse(event)

        if artifact.get("status") in {"completed", "failed"}:
            return

        subscriber = service.subscribe(run_id)
        try:
            while True:
                try:
                    event = await asyncio.to_thread(subscriber.get, True, 5.0)
                    emitted += 1
                    yield _format_sse(event)
                    if event.get("type") in TERMINAL_EVENT_TYPES:
                        break
                except queue.Empty:
                    refreshed = service.get_run(run_id)
                    for event in refreshed.get("events", [])[emitted:]:
                        emitted += 1
                        yield _format_sse(event)
                        if event.get("type") in TERMINAL_EVENT_TYPES:
                            return
        finally:
            service.unsubscribe(run_id, subscriber)

    return StreamingResponse(generator(), media_type="text/event-stream")


def _format_sse(event: dict[str, Any]) -> str:
    return f"event: {event.get('type', 'message')}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


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

    return StreamingResponse(generator(), media_type="text/event-stream")


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
    resolved = base_url.strip()
    if not resolved:
        return ""
    parsed = urlsplit(resolved)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise HTTPException(
            status_code=400,
            detail="Provider URL must be an HTTP(S) base URL without credentials, query, or fragment.",
        )
    return resolved.rstrip("/")


def _validate_attachment_budget(attachments: list[dict[str, Any]]) -> None:
    total = 0
    for attachment in attachments:
        declared_size = int(attachment.get("byte_size", 0) or 0)
        data_url = str(attachment.get("data_url", "") or "")
        text_content = str(attachment.get("text_content", "") or "")
        estimated_size = declared_size
        if data_url:
            encoded = data_url.partition(",")[2] or data_url
            estimated_size = max(estimated_size, (len(encoded) * 3) // 4)
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
