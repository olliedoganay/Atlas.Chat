# Atlas Chat

[![CI](https://github.com/olliedoganay/AtlasChat/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/olliedoganay/AtlasChat/actions/workflows/ci.yml)
[![Release](https://github.com/olliedoganay/AtlasChat/actions/workflows/release.yml/badge.svg)](https://github.com/olliedoganay/AtlasChat/actions/workflows/release.yml)
[![Latest release](https://img.shields.io/github/v/release/olliedoganay/AtlasChat?label=latest%20release)](https://github.com/olliedoganay/AtlasChat/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Atlas Chat is a local-first desktop app for working with local AI models. It provides a multi-thread chat workspace, profile-scoped memory, hardware-aware model discovery, run inspection, and a built-in code runner while keeping Atlas-managed state on the local machine.

Current version: `1.3.1`

<p align="center">
  <img src="docs/assets/atlas-chat-workspace.png" alt="Atlas Chat workspace" style="max-width: 100%; height: auto;">
</p>

## Download

For normal desktop usage, install the packaged desktop release instead of running from source.

[Download the latest release](https://github.com/olliedoganay/AtlasChat/releases/latest)

## Highlights

- Multi-thread local chat workspace for long-running conversations
- Hardware-aware Discovery page with managed Ollama downloads, live progress, cancel/retry controls, and copyable fallback commands
- In-app local provider and endpoint setup under **Settings > Connections**
- Per-user profiles with optional password protection
- Local search across the active profile's chats
- Thread rename, duplicate, branch, model-lock, and temperature-lock workflows
- Reasoning traces, token streaming, stop controls, and saved run diagnostics
- Automatic and manual context compaction for long threads
- Optional cross-chat memory with manual remember/forget controls
- Third-party Mem0 telemetry disabled in the Atlas runtime
- Image and file attachments in the composer
- One-click execution for generated code snippets in isolated run windows
- Keyboard-friendly, responsive desktop navigation and dialogs with a lighter route-split frontend

Atlas requires a local model runtime. Ollama is the default path, and Atlas can also connect to local OpenAI-compatible runtimes such as LM Studio, llama.cpp server, vLLM, and LocalAI. Docker is optional for chat, but required for server-side code execution.

## Install

1. Open `https://github.com/olliedoganay/AtlasChat/releases/latest`.
2. Download the installer for your platform:
   - Windows: current x64 MSI installer.
   - Linux: current x64 `.deb`, `.rpm`, or AppImage package.
   - macOS 14 or newer: current unsigned `.dmg` package for Apple Silicon or Intel.
3. Install and launch `Atlas Chat`.

Atlas Chat packages community/open-source desktop builds without Apple notarization. On macOS, Gatekeeper may require opening the app manually from Finder the first time.

## Install a Local Model Provider

Atlas Chat needs a local model provider before sending your first message. Ollama remains the default setup and is still used by Atlas memory features today. In the desktop app, open **Settings > Connections** to select Ollama or a supported local OpenAI-compatible runtime, confirm its local API URL, and choose **Save & restart**. Environment variables remain available for source, CLI, and automated setups.

Official Ollama install docs:

- Windows: `https://docs.ollama.com/windows`
- macOS: `https://docs.ollama.com/macos`
- Linux: `https://docs.ollama.com/linux`

Windows:

1. Download and run the Ollama installer from `https://ollama.com/download`.
2. Start Ollama from the Start menu. It runs in the background and exposes the `ollama` command in PowerShell or Command Prompt.

macOS:

1. Download the Ollama DMG from `https://ollama.com/download`.
2. Mount it, drag `Ollama.app` to `Applications`, then start Ollama.
3. Allow Ollama to add the `ollama` command to your PATH if macOS asks.

Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
```

Verify Ollama is reachable:

```bash
ollama -v
ollama list
```

Pull models for Atlas:

```bash
ollama pull gpt-oss:20b
ollama pull nomic-embed-text:latest
```

Atlas connects to Ollama on `http://127.0.0.1:11434` by default. If Atlas says Ollama is unavailable, start Ollama and refresh the local model list.

Other local chat providers:

| Provider | `ATLAS_CHAT_PROVIDER` | Default local API URL |
| --- | --- | --- |
| LM Studio | `lmstudio` | `http://127.0.0.1:1234/v1` |
| llama.cpp server | `llamacpp` | `http://127.0.0.1:8080/v1` |
| vLLM | `vllm` | `http://127.0.0.1:8000/v1` |
| LocalAI | `localai` | `http://127.0.0.1:8080/v1` |
| Generic local OpenAI-compatible server | `openai-compatible` | `http://127.0.0.1:8000/v1` |

For these providers, start the provider's local server, make a chat model available in that runtime, then select it under **Settings > Connections**. A saved provider and API URL take precedence over the matching environment defaults. This is local-provider support only; Atlas does not add hosted cloud providers here.

If a local runtime requires an API key, Atlas stores it only when secure operating-system secret protection is available. The key is protected before it is written, the frontend receives only key-present/key-unavailable status, and Atlas refuses to persist a key instead of silently falling back to plaintext. `ATLAS_CHAT_API_KEY` remains available for intentionally managed headless or development environments.

With Ollama selected, the Discovery page can download recommended models directly. Atlas streams Ollama's download progress, allows the active download to be cancelled, exposes retry after a failure, and refreshes the installed-model list when the pull completes. Only one managed model download runs at a time; the copyable `ollama pull ...` command remains available as a manual fallback.

## Requirements

- Python 3.14+
- Node.js 22.12+
- Rust stable toolchain with Cargo
- A local chat runtime: Ollama, LM Studio, llama.cpp server, vLLM, LocalAI, or another local OpenAI-compatible endpoint
- At least one local chat model of your choice
- A local embedding model, for example `nomic-embed-text:latest`
- Tauri prerequisites for your platform: `https://v2.tauri.app/start/prerequisites/`
- macOS 14+ for packaged macOS builds

## Source Setup

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
Copy-Item .env.example .env
# Pull any chat model you prefer.
ollama pull gpt-oss:20b
ollama pull nomic-embed-text:latest
Set-Location apps\atlas
npm ci
Set-Location ..\..
```

macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
cp .env.example .env
# Pull any chat model you prefer.
ollama pull gpt-oss:20b
ollama pull nomic-embed-text:latest
cd apps/atlas
npm ci
cd ../..
```

`requirements.lock` is the cross-platform, hash-locked environment used for reproducible source, CI, and release installs. `requirements.txt` is the smaller direct dependency/tooling input used to regenerate that lock.

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
3. Under `Connections`, select the local provider and API URL.
4. For Ollama, use `Discovery` to download a model if needed.
5. Return to `Workspace`, pick a model, and start a chat.

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
- `atlas --user-id your_user --model <local-model>` starts the terminal chat CLI on thread `main`.
- `atlas --user-id your_user --model <local-model> "..."` runs a single turn through the top-level launcher.
- `python -m atlas_local.cli ...` exposes the raw `ask`, `chat`, and `memories` subcommands.

## Configuration

Copy `.env.example` to `.env` and adjust it if needed.

For desktop use, prefer **Settings > Connections** for the chat provider, local API URL, and optional key. Atlas stores non-secret provider settings under `ATLAS_DATA_DIR`; saved values override the equivalent environment defaults after the managed backend restarts. Environment variables are still useful for initial defaults, direct backend runs, and automation.

| Variable | Purpose | Default |
| --- | --- | --- |
| `OLLAMA_URL` | Local Ollama base URL for Ollama chat and current memory embedding | `http://127.0.0.1:11434` |
| `ATLAS_CHAT_PROVIDER` | Local chat provider: `ollama`, `lmstudio`, `llamacpp`, `vllm`, `localai`, or `openai-compatible` | `ollama` |
| `ATLAS_CHAT_BASE_URL` | Override the selected provider's local API URL | provider default |
| `ATLAS_CHAT_API_KEY` | Optional bearer token for local OpenAI-compatible runtimes that require one | blank |
| `CHAT_TEMPERATURE` | Optional initial sampling temperature; blank uses the selected model behavior | blank |
| `EMBED_MODEL` | Embedding model used for memory retrieval | `nomic-embed-text:latest` |
| `QDRANT_PATH` | Local vector-store directory | `.data/qdrant` |
| `MEM0_COLLECTION` | Collection name for persistent memory | `atlas_local_memory` |
| `EMBED_DIM` | Embedding dimension expected by local memory storage | `768` |
| `LANGGRAPH_CHECKPOINT_DB` | SQLite checkpoint path for thread state | `.data/langgraph/checkpoints.sqlite` |
| `MEM0_HISTORY_DB` | SQLite history path for memory records | `.data/mem0_history.sqlite` |
| `MEMORY_TOP_K` | Number of recalled memories to inject into a turn | `5` |
| `ATLAS_COMPACTION_TIMEOUT_SECONDS` | Max seconds to wait for a model-backed context compaction before leaving the existing context unchanged | `25` |

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
| `ATLAS_ALLOW_LEGACY_PLAINTEXT_MIGRATION` | Explicit one-launch opt-in to import pre-encryption run indexes/artifacts; remove immediately after migration | off |
| `ATLAS_DISCOVERY_MANIFEST` | Optional path to a versioned Discovery recommendation manifest | bundled manifest |
| `ATLAS_RUNNER_NETWORK` | Docker network mode for code runs; `none` or `bridge` | `none` |
| `ATLAS_RUNNER_TIMEOUT_SECONDS` | Non-GUI code-runner timeout in seconds | `120` |
| `ATLAS_RUNNER_GUI_TIMEOUT_SECONDS` | GUI code-runner timeout in seconds | `900` |
| `ATLAS_RUNNER_MAX_CONCURRENT` | Maximum simultaneous Docker-backed code runs, bounded from 1 to 8 | `2` |
| `ATLAS_RUNNER_HISTORY_LIMIT` | Maximum retained events per code run, bounded to 20,000 | `2000` |
| `ATLAS_RUNNER_SUBSCRIBER_QUEUE_SIZE` | Maximum queued events per code-run stream subscriber, bounded to 2,048 | `256` |
| `ATLAS_RUNNER_MAX_OUTPUT_BYTES` | Per-run streamed output budget, bounded to 16 MiB | `1048576` |
| `ATLAS_RUNNER_STORAGE_LIMIT` | Optional container writable-layer limit when supported by the container engine | unset |
| `VITE_ATLAS_BACKEND_URL` | Frontend-only direct backend URL for plain browser/Vite development | unset |
| `VITE_ATLAS_BACKEND_TOKEN` | Optional token paired with `VITE_ATLAS_BACKEND_URL` | unset |

`ATLAS_INSTANCE_TOKEN` is managed automatically by the Tauri shell and direct launchers. Do not hardcode it in `.env` for normal usage.

## Code Runner

Runnable code blocks get a **Run** button next to **Copy**. Clicking it opens a separate Atlas Run window that executes the snippet and streams output live. Closing the run window stops the run.

Server-side languages run through Docker: Python, JavaScript, TypeScript, Go, Rust, C, C++, Java, Ruby, PHP, Bash, C#, Kotlin, Swift, Perl, Lua, R, Elixir, and Dart. The first run for a language may need its Docker image to be downloaded, which can use substantial disk space.

HTML renders in a sandboxed, offline-by-default client-side preview and does not require Docker.

Runner behavior:

- Outbound access is never enabled implicitly. HTML previews block remote scripts, requests, media, and form submissions. `ATLAS_RUNNER_NETWORK=none` keeps Docker runs offline; dependency-aware runs warn instead of silently switching networks. Set `ATLAS_RUNNER_NETWORK=bridge` only when submitted Docker code may access the network.
- GUI previews use an Atlas-owned internal Docker network when outbound access is disabled. Server-generated web pages are not loaded into the host WebView in this mode; their browser preview is available only with the explicit `ATLAS_RUNNER_NETWORK=bridge` opt-in because browser-side requests are outside Docker's network boundary.
- Non-GUI dependency-aware runs attempt on-demand installation under the configured network policy. With `none`, Atlas warns and the download normally fails without widening access; `bridge` must be selected explicitly for package resolution.
- Python GUI snippets use the versioned Atlas Python GUI runtime. On first use, the runner can build it from the checked-in Dockerfile and hash-locked allowlist while showing preparation progress. This trusted preparation phase may download the pinned base image and dependencies, but it never mounts or executes submitted code and concurrent requests share one build.
- The prepared GUI runtime currently bundles `pygame==2.6.1`, its `numpy==2.5.2` optional array/audio support, and the fixed Tk, SDL, terminal, and noVNC system stack. GUI snippets that request any undeclared third-party package fail before execution with an actionable offline-runtime error; Atlas does not grant the snippet network access or run apt/pip for it.
- Submitted Python GUI code runs in a separate disposable container as UID 65534 with a read-only root filesystem and source mount, all capabilities dropped, `no-new-privileges`, and only Atlas's internal preview network. The prepared image remains cached for later offline GUI runs.
- Docker-backed runs use fully qualified, versioned images and disposable containers with CPU, memory, PID, file-size, timeout, output, queue, and concurrency controls. Containers drop capabilities, enable `no-new-privileges`, and use a read-only root filesystem plus an unprivileged user for compatible workloads.
- Closing the run window stops its owned container and removes any per-run installed dependencies with it; Atlas also cleans up owned containers during backend shutdown. Docker may keep base language images and the versioned prepared GUI runtime.
- `ATLAS_RUNNER_TIMEOUT_SECONDS` controls non-GUI run TTL and `ATLAS_RUNNER_GUI_TIMEOUT_SECONDS` controls GUI run TTL.
- If Docker is unavailable, Atlas shows a retry path in the run window.

## Architecture and Security

Atlas is built as a local desktop system:

- The Tauri shell starts and manages a loopback-only backend.
- The backend binds to `127.0.0.1` on a random local port.
- The frontend authenticates every request with a per-launch instance token.
- The backend rejects unexpected origins unless explicitly configured for direct localhost development.
- The selected model provider is expected to run locally on the same machine.
- Provider API credentials are never returned through the settings API; Atlas reports only whether a protected key is configured.
- Sensitive Atlas-managed state fails closed when secure operating-system key storage is unavailable; it is not silently written as plaintext.

Very old Atlas data from before encrypted run storage is rejected by default. To migrate it, launch Atlas once with `ATLAS_ALLOW_LEGACY_PLAINTEXT_MIGRATION=1`, verify the data, then remove the variable. The opt-in deliberately permits unauthenticated legacy input and must not remain enabled.

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

The cross-platform CI gate also runs Ruff, frontend tests and the production Vite build, Cargo formatting/checks/tests/Clippy, Python and npm dependency audits, and CodeQL analysis. Tagged release jobs build each platform package and publish SHA-256 checksum files beside the installers.

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

Atlas builds MSI as the canonical Windows installer. The consolidated GitHub Actions `release` workflow verifies the release candidate once, builds Windows MSI, Linux `.deb`/`.rpm`/AppImage, and unsigned macOS Intel/x64 and Apple Silicon/arm64 DMG packages, then publishes them together only after every platform succeeds.

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
- `provider-settings.json`: selected local provider, API URL, and protected API-key ciphertext when OS secret protection is available
- `storage.key.json`: local storage key material for encrypted-at-rest storage where supported
- `logs/`: backend logs for source runs when launched through the desktop shell

These paths should remain untracked. The repo ignore rules cover them.

## Troubleshooting

If PowerShell blocks repo scripts, allow them for the current session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

If the desktop app opens but no models appear:

1. Make sure your selected local model provider is running.
2. Check the provider and local API URL under **Settings > Connections**.
3. Make sure at least one chat model is installed or loaded in that provider.
4. For Ollama, open **Discovery** to download a recommended model or copy its manual pull command.
5. Restart Atlas.

If the backend shows offline after a Python code change, fully close and reopen the app. A frontend refresh is not enough for backend changes.

If Atlas only stays open while the terminal that launched it stays open, you are running the source build in dev mode. The packaged Windows release runs as a normal installed desktop app.
