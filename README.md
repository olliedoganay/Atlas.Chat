# Atlas Chat

[![CI](https://github.com/olliedoganay/Atlas.Chat/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/olliedoganay/Atlas.Chat/actions/workflows/ci.yml)
[![Windows release](https://github.com/olliedoganay/Atlas.Chat/actions/workflows/release-windows.yml/badge.svg)](https://github.com/olliedoganay/Atlas.Chat/actions/workflows/release-windows.yml)
[![Linux release](https://github.com/olliedoganay/Atlas.Chat/actions/workflows/release-linux.yml/badge.svg)](https://github.com/olliedoganay/Atlas.Chat/actions/workflows/release-linux.yml)
[![Latest release](https://img.shields.io/github/v/release/olliedoganay/Atlas.Chat?label=latest%20release)](https://github.com/olliedoganay/Atlas.Chat/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Atlas Chat is a local-first desktop app for working with local Ollama models. It provides a multi-thread chat workspace, profile-scoped memory, hardware-aware model discovery, run inspection, and a built-in code runner while keeping Atlas-managed state on the local machine.

Current version: `1.1.0`

<p align="center">
  <img src="docs/assets/atlas-chat-workspace.png" alt="Atlas Chat workspace" style="max-width: 100%; height: auto;">
</p>

## Download

For normal desktop usage, install the packaged desktop release instead of running from source.

[Download the latest release](https://github.com/olliedoganay/Atlas.Chat/releases/latest)

## Highlights

- Multi-thread local chat workspace for long-running conversations
- Hardware-aware Discovery page with Ollama model recommendations and pull commands
- Per-user profiles with optional password protection
- Local search across the active profile's chats
- Thread rename, duplicate, branch, model-lock, and temperature-lock workflows
- Reasoning traces, token streaming, stop controls, and saved run diagnostics
- Automatic and manual context compaction for long threads
- Optional cross-chat memory with manual remember/forget controls
- Image and file attachments in the composer
- One-click execution for generated code snippets in isolated run windows

Atlas requires a local Ollama runtime. Docker is optional for chat, but required for server-side code execution.

## Install

1. Open `https://github.com/olliedoganay/Atlas.Chat/releases/latest`.
2. Download the installer for your platform:
   - Windows: current x64 MSI installer.
   - Linux: current x64 `.deb`, `.rpm`, or AppImage package.
3. Install and launch `Atlas Chat`.

Atlas Chat does not currently publish macOS installers. Use the source workflow on macOS.

## Requirements

- Python 3.14+
- Node.js 20+
- Rust stable toolchain with Cargo
- Ollama running locally
- At least one local chat model of your choice
- A local embedding model, for example `nomic-embed-text:latest`
- Tauri prerequisites for your platform: `https://v2.tauri.app/start/prerequisites/`

## Source Setup

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
# Pull any chat model you prefer.
ollama pull gpt-oss:20b
ollama pull nomic-embed-text:latest
Set-Location apps\atlas
npm install
Set-Location ..\..
```

macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
# Pull any chat model you prefer.
ollama pull gpt-oss:20b
ollama pull nomic-embed-text:latest
cd apps/atlas
npm install
cd ../..
```

## Run From Source

Windows:

```powershell
.\scripts\start_atlas_dev.ps1
```

The launcher reuses an existing Atlas dev session when possible. When an already-built `target\debug\atlas-desktop.exe` exists, it starts the Vite frontend and reuses that desktop executable to avoid rebuilding the Rust/Tauri target cache on every launch.

When you intentionally need to rebuild the Rust/Tauri desktop shell after native source changes:

```powershell
.\scripts\start_atlas_dev.ps1 -RebuildDesktop
```

First compile, or any explicit rebuild, can create a large Rust `src-tauri/target` cache.

macOS and Linux:

```bash
source .venv/bin/activate
cd apps/atlas
npm run tauri dev
```

First launch:

1. Open `Settings`.
2. Create or select a profile.
3. Return to `Workspace`.
4. Pick a model and start a chat.

## CLI and Backend

These commands are mainly for diagnostics and automation.

Windows:

```powershell
.venv\Scripts\atlas-backend.exe
.venv\Scripts\atlas.exe --user-id your_user --model your-model:tag
.venv\Scripts\atlas.exe --user-id your_user --model your-model:tag "Summarize this project in three bullets."
.venv\Scripts\python.exe -m atlas_local.cli ask "Summarize this project in three bullets." --user-id your_user --thread-id scratch --model your-model:tag
.venv\Scripts\python.exe -m atlas_local.api
```

macOS and Linux:

```bash
source .venv/bin/activate
atlas-backend
atlas --user-id your_user --model your-model:tag
atlas --user-id your_user --model your-model:tag "Summarize this project in three bullets."
python -m atlas_local.cli ask "Summarize this project in three bullets." --user-id your_user --thread-id scratch --model your-model:tag
python -m atlas_local.api
```

- `atlas-backend` or `python -m atlas_local.api` runs only the local API/backend.
- `atlas --user-id your_user --model <ollama-model>` starts the terminal chat CLI on thread `main`.
- `atlas --user-id your_user --model <ollama-model> "..."` runs a single turn through the top-level launcher.
- `python -m atlas_local.cli ...` exposes the raw `ask`, `chat`, and `memories` subcommands.

## Configuration

Copy `.env.example` to `.env` and adjust it if needed.

| Variable | Purpose | Default |
| --- | --- | --- |
| `OLLAMA_URL` | Local Ollama base URL | `http://127.0.0.1:11434` |
| `CHAT_TEMPERATURE` | Optional initial sampling temperature; blank uses the selected model behavior | blank |
| `EMBED_MODEL` | Embedding model used for memory retrieval | `nomic-embed-text:latest` |
| `QDRANT_PATH` | Local vector-store directory | `.data/qdrant` |
| `MEM0_COLLECTION` | Collection name for persistent memory | `atlas_local_memory` |
| `EMBED_DIM` | Embedding dimension expected by local memory storage | `768` |
| `LANGGRAPH_CHECKPOINT_DB` | SQLite checkpoint path for thread state | `.data/langgraph/checkpoints.sqlite` |
| `MEM0_HISTORY_DB` | SQLite history path for memory records | `.data/mem0_history.sqlite` |
| `MEMORY_TOP_K` | Number of recalled memories to inject into a turn | `5` |
| `ATLAS_COMPACTION_TIMEOUT_SECONDS` | Max seconds to wait for a model-backed context compaction before falling back | `25` |

Runtime overrides:

| Variable | Purpose | Default |
| --- | --- | --- |
| `ATLAS_PROJECT_ROOT` | Override the effective project root used by the backend | repo root |
| `ATLAS_PROMPT_DIR` | Override the prompt directory | `<project root>/prompts` |
| `ATLAS_DATA_DIR` | Override the main data directory | `<project root>/.data` |
| `ATLAS_API_HOST` | Host for direct backend runs | `127.0.0.1` |
| `ATLAS_API_PORT` | Port for direct backend runs | `8765` |
| `ATLAS_ALLOWED_ORIGINS` | Comma-separated explicit origin allowlist override | built-in Tauri/dev origins |
| `ATLAS_ALLOW_INSECURE_LOCALHOST` | Allow localhost-only direct backend development without the managed instance token | off |
| `ATLAS_DISCOVERY_MANIFEST` | Optional path to a versioned Discovery recommendation manifest | bundled manifest |
| `ATLAS_RUNNER_NETWORK` | Docker network mode for code runs; `none` or `bridge` | `none` |
| `ATLAS_RUNNER_TIMEOUT_SECONDS` | Non-GUI code-runner timeout in seconds | `120` |
| `ATLAS_RUNNER_GUI_TIMEOUT_SECONDS` | GUI code-runner timeout in seconds | `900` |
| `VITE_ATLAS_BACKEND_URL` | Frontend-only direct backend URL for plain browser/Vite development | unset |
| `VITE_ATLAS_BACKEND_TOKEN` | Optional token paired with `VITE_ATLAS_BACKEND_URL` | unset |

`ATLAS_INSTANCE_TOKEN` is managed automatically by the Tauri shell and direct launchers. Do not hardcode it in `.env` for normal usage.

## Code Runner

Runnable code blocks get a **Run** button next to **Copy**. Clicking it opens a separate Atlas Run window that executes the snippet and streams output live. Closing the run window stops the run.

Server-side languages run through Docker: Python, JavaScript, TypeScript, Go, Rust, C, C++, Java, Ruby, PHP, Bash, C#, Kotlin, Swift, Perl, Lua, R, Elixir, and Dart. The first run for a language may need its Docker image to be downloaded, which can use substantial disk space.

HTML renders in a sandboxed client-side preview and does not require Docker.

Runner behavior:

- Dependencies are installed on demand based on snippet imports when outbound network access is enabled for the run.
- Python GUI snippets can open a live embedded noVNC view for supported toolkits. GUI system packages are installed inside the disposable run container only when needed.
- Docker-backed runs use disposable containers with CPU, memory, PID, timeout, and network controls. Closing the run window stops the container and removes the run-installed dependencies with it; Docker may still keep the base language image.
- Non-GUI Docker runs default to `--network none`; set `ATLAS_RUNNER_NETWORK=bridge` only when snippets need outbound dependency resolution or downloads.
- `ATLAS_RUNNER_TIMEOUT_SECONDS` controls non-GUI run TTL and `ATLAS_RUNNER_GUI_TIMEOUT_SECONDS` controls GUI run TTL.
- If Docker is unavailable, Atlas shows a retry path in the run window.

## Architecture and Security

Atlas is built as a local desktop system:

- The Tauri shell starts and manages a loopback-only backend.
- The backend binds to `127.0.0.1` on a random local port.
- The frontend authenticates every request with a per-launch instance token.
- The backend rejects unexpected origins unless explicitly configured for direct localhost development.
- Ollama is expected to run locally on the same machine.
- Local state is stored under Atlas-managed directories, with additional at-rest protection where supported.

For source runs, local data is written under `.data/`. Packaged builds use the app data directory for the current user.

## Verify and Build

On Windows, the canonical verification command is:

```powershell
.\scripts\verify_repo.ps1
```

It runs version consistency checks, Python tests, frontend tests, the frontend release build, and `cargo check`.

Optional flags:

- `-SkipBackend`
- `-SkipFrontend`

Plain `pytest` from the repo root is also safe because test discovery is scoped to `tests/`.

Before cutting a release tag, bump every checked manifest together:

```powershell
.venv\Scripts\python.exe scripts\bump_atlas_version.py X.Y.Z
```

Omit the version argument to advance to the next patch version.

Build the Windows MSI release bundle:

```powershell
.\scripts\build_atlas_release.ps1
```

Artifacts are written under:

```text
apps\atlas\src-tauri\target\release\bundle\
```

Atlas builds MSI as the canonical Windows installer. The Windows release workflow publishes MSI artifacts only.

Linux release bundles are built on GitHub Actions with the `release-linux` workflow. It publishes `.deb`, `.rpm`, and AppImage artifacts for tagged releases.

## Repository Layout

- `src/atlas_local`: Python backend, API, graph execution, memory, discovery, security helpers, code runner, and CLI entrypoints
- `apps/atlas/src`: React and Vite desktop UI
- `apps/atlas/src-tauri`: Rust Tauri shell that launches and manages the backend
- `prompts/answer.md`: backend answer prompt template packaged with the desktop app
- `scripts`: development, verification, packaging, and cleanup helpers
- `tests`: backend and API tests
- `atlas.py`, `smoke_test.py`: source-tree CLI wrappers that reuse `.venv` when present
- `AI.md`: compact operational notes for AI/coding agents; release tooling keeps its version fields in sync

## Local Data

For source runs, Atlas writes runtime data under `.data/`:

- `langgraph/checkpoints.sqlite`: thread checkpoint state
- `mem0_history.sqlite`: memory history database
- `qdrant/`: local vector storage for semantic memory
- `runs/index.json`: thread, run, and user index
- `runs/<run-id>.json`: saved run artifacts
- `storage.key.json`: local storage key material for encrypted-at-rest storage where supported
- `logs/`: backend logs for source runs when launched through the desktop shell

These paths should remain untracked. The repo ignore rules cover them.

## Troubleshooting

If PowerShell blocks repo scripts, allow them for the current session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

If the desktop app opens but no models appear:

1. Make sure Ollama is running.
2. Make sure the models in `.env` have been pulled locally.
3. Restart Atlas.

If the backend shows offline after a Python code change, fully close and reopen the app. A frontend refresh is not enough for backend changes.

If Atlas only stays open while the terminal that launched it stays open, you are running the source build in dev mode. The packaged Windows release runs as a normal installed desktop app.
