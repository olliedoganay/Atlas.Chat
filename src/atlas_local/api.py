from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
from contextlib import asynccontextmanager
from typing import Any

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


class PromptRequest(BaseModel):
    prompt: str = ""
    user_id: str = Field(..., min_length=1)
    thread_id: str = Field(..., min_length=1)
    chat_model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_mode: str | None = None
    thread_title: str | None = None
    cross_chat_memory: bool = True
    auto_compact_long_chats: bool = True
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    images: list[dict[str, str]] = Field(default_factory=list)


class UserRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    password: str | None = None


class UnlockUserRequest(BaseModel):
    password: str | None = None


class MemoryCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class ModelContextWindowRequest(BaseModel):
    context_window: int | None = Field(default=None, ge=1024, le=262144)


class ModelUnloadRequest(BaseModel):
    model: str = Field(..., min_length=1)


class ThreadTitleRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)


class ThreadBranchRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    after_message_count: int = Field(..., ge=0)


class ResetThreadRequest(BaseModel):
    thread_id: str
    user_id: str | None = None


class ResetAllRequest(BaseModel):
    confirmation: str


class RunnerExecRequest(BaseModel):
    language: str = Field(..., min_length=1)
    code: str = Field(..., min_length=1)


DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "tauri://localhost",
)


def create_api_app(service: AtlasBackendService | None = None) -> FastAPI:
    managed_service = service
    required_token = os.environ.get("ATLAS_INSTANCE_TOKEN", "").strip()
    allow_insecure_localhost = _allow_insecure_localhost()
    allowed_origins = _allowed_origins()

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
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Atlas-Instance-Token"],
    )
    if managed_service is not None:
        app.state.service = managed_service

    @app.middleware("http")
    async def require_instance_token(request: Request, call_next):
        origin = request.headers.get("origin", "").strip()
        if origin and not allow_insecure_localhost and origin not in allowed_origins:
            return JSONResponse(status_code=403, content={"detail": "Atlas backend rejected this origin."})
        if request.method.upper() == "OPTIONS":
            return await call_next(request)
        if required_token:
            provided = request.headers.get("x-atlas-instance-token", "").strip()
            if provided != required_token:
                return JSONResponse(status_code=401, content={"detail": "Atlas backend identity check failed."})
        return await call_next(request)

    def backend() -> AtlasBackendService:
        return app.state.service

    @app.get("/health")
    def health() -> dict[str, Any]:
        return backend().health()

    @app.get("/status")
    def status() -> dict[str, Any]:
        return backend().status()

    @app.get("/models")
    def models() -> dict[str, Any]:
        return backend().list_models()

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
    def memories(user_id: str = Query(..., min_length=1), limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, Any]]:
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
                attachments=attachments,
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
    uvicorn.run("atlas_local.api:create_api_app", host=args.host, port=args.port, factory=True)
    return 0


def _allow_insecure_localhost() -> bool:
    return os.environ.get("ATLAS_ALLOW_INSECURE_LOCALHOST", "").strip().lower() in {"1", "true", "yes", "on"}


def _allowed_origins() -> tuple[str, ...]:
    raw = os.environ.get("ATLAS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return DEFAULT_ALLOWED_ORIGINS
    values = tuple(origin.strip() for origin in raw.split(",") if origin.strip())
    return values or DEFAULT_ALLOWED_ORIGINS


if __name__ == "__main__":
    raise SystemExit(main())
