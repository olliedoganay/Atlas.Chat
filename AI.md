# Atlas Repo Note

- Repo: `olliedoganay/AtlasChat`
- Current version: `1.3.1`
- Stack: Tauri 2 + React/Vite frontend, Rust desktop shell, Python FastAPI backend, local model providers, Docker-backed code runner.
- Runtime shape: local-first desktop app; Tauri launches a loopback backend protected by a per-launch instance token.

## Main Paths

- UI: `apps/atlas/src`
- Tauri shell/backend launcher: `apps/atlas/src-tauri/src/lib.rs`
- Backend API/service: `src/atlas_local/api.py`, `src/atlas_local/api_service.py`
- LLM/local provider layer: `src/atlas_local/llm.py`, `src/atlas_local/providers`
- Saved provider settings and managed Ollama pulls: `src/atlas_local/provider_settings.py`, `src/atlas_local/model_pulls.py`
- Config/version/context settings: `src/atlas_local/config.py`
- Code runner: `src/atlas_local/code_runner.py`
- Answer prompt: `prompts/answer.md`
- Tests: `tests`, `apps/atlas/src/**/*.test.ts*`
- Reproducible Python environment: `requirements.lock` (generated from `requirements.txt`)
- Cross-platform release workflow: `.github/workflows/release.yml`

## Setup

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
Copy-Item .env.example .env
cd apps\atlas
npm ci
cd ..\..
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
cp .env.example .env
cd apps/atlas
npm ci
cd ../..
```

Use `requirements.lock` for local verification, CI, and release reproduction. Change direct pins in `requirements.txt`, then regenerate the universal hash-locked file intentionally; do not hand-edit the generated lock.

## Common Commands

After activating the repo virtual environment:

- Start Windows source app: `.\scripts\start_atlas_dev.ps1`
- Force Windows desktop rebuild: `.\scripts\start_atlas_dev.ps1 -RebuildDesktop`
- Start Linux/macOS source app: `cd apps/atlas && npm run tauri dev`
- Version check: `python scripts/check_atlas_version.py --tag vX.Y.Z`
- Bump all release manifests: `python scripts/bump_atlas_version.py X.Y.Z`
- Backend tests: `python -m unittest discover -s tests`
- Focused runner tests: `python -m unittest tests.test_code_runner`
- Python lint: `python -m ruff check src tests scripts`
- Python dependency audit: `python -m pip_audit -r requirements.lock`
- Frontend tests: `cd apps/atlas && npm test`
- Frontend build: `cd apps/atlas && npm run build`
- Frontend dependency audit: `cd apps/atlas && npm audit --audit-level=high`
- Rust verification: `cd apps/atlas/src-tauri && cargo fmt --check && cargo check --locked && cargo test --locked && cargo clippy --all-targets -- -D warnings`
- Windows full verification: `.\scripts\verify_repo.ps1`
- Windows release bundle: `.\scripts\build_atlas_release.ps1`

Plain `pytest` from repo root is intended to be safe when pytest is installed, but `python -m unittest discover -s tests` is the backend baseline.

## Current Feature Notes

- CI and release workflows use Python `3.14` and Node `22`.
- **Settings > Connections** is the normal desktop path for choosing the local chat provider, API URL, and optional API key. Saved settings override equivalent environment defaults after a managed restart.
- Provider keys must use available operating-system secret protection. The settings API returns only key status, and persistence must fail closed when secure key storage is unavailable.
- Mem0 telemetry is disabled before importing Mem0; preserve this local-first boundary so memory use neither creates shared telemetry state outside `ATLAS_DATA_DIR` nor sends third-party events.
- The Discovery page performs real managed Ollama pulls with streamed progress, cancel, retry, and installed-model refresh. Managed downloads are Ollama-only and limited to one active pull.
- Settings > Models has a global Ollama context-window slider when the chat provider is Ollama. Atlas sends the selected value as `num_ctx`; `Auto` follows Ollama defaults.
- `ATLAS_CHAT_PROVIDER`, `ATLAS_CHAT_BASE_URL`, and `ATLAS_CHAT_API_KEY` remain fallback controls for fresh, direct-backend, and automated environments. Non-Ollama providers use local OpenAI-compatible `/v1` APIs.
- Context compaction is automatic/manual, uses the effective context window, and has visible compaction status/animation in the composer area.
- Only model outputs, thinking traces, and compaction summaries should be text-selectable in the app UI.
- The titlebar is custom and compact; keep it visually close to VS Code-style density.
- Offline local provider states should stay compact. Avoid large blocking banners when the empty-state already explains what is wrong.
- First-run and search dialogs use focus-trapped dialog primitives; preserve keyboard navigation, explicit labels, live status regions, and focus restoration when changing them.
- The desktop shell has a 900 px minimum width and compact/split-screen layouts below the wide-desktop breakpoint. Preserve the fixed desktop frame and test narrow-height as well as narrow-width behavior.
- Discovery, Advanced, Settings, the runner, Markdown, and syntax highlighting are split from the initial route where practical. Keep Latin-only font imports unless broader glyph coverage is intentionally required.

## Code Runner

Server-side Docker languages:

```text
python, javascript, typescript, go, rust, c, cpp, java, ruby, php, bash,
csharp, kotlin, swift, perl, lua, r, elixir, dart
```

Client-side browser language:

```text
html
```

Aliases include:

```text
py, python3, js, node, ts, golang, rs, c++, cxx, cc, rb, sh, shell,
zsh, cs, c#, kt, kts, pl, ex, exs, htm
```

Runner behavior:

- Docker image names are fully qualified to avoid short-name resolution ambiguity.
- `ATLAS_RUNNER_NETWORK=none` is the default and must never silently become outbound `bridge` networking. Dependency-aware runs warn when explicit outbound access is absent.
- `ATLAS_RUNNER_NETWORK=bridge` is the only mode that grants outbound access; use it deliberately for package resolution or code that needs the network.
- GUI and web-preview containers under `none` use the Atlas-owned internal network so loopback port publishing works without outbound routing. Atlas embeds the GUI/noVNC preview; it does not load server-generated web pages into the host WebView unless `bridge` was explicitly selected, because browser-side requests sit outside Docker's network boundary.
- Python GUI runs use the versioned runtime defined in `src/atlas_local/runner_images/python_gui`. Trusted preparation may fetch only its pinned base image, fixed apt list, and hash-locked Python allowlist; it never receives submitted code. Preparation is observable and deduplicated in the backend.
- The prepared GUI runtime currently allowlists `pygame==2.6.1` and `numpy==2.5.2` (required by Pygame's optional array/audio APIs). Reject undeclared third-party GUI dependencies before execution rather than enabling runtime apt/pip or outbound access.
- Execute GUI snippets in a separate disposable container as UID 65534 with a read-only root filesystem and source mount, all capabilities dropped, `no-new-privileges`, and the internal preview network only. Keep runtime image labels/version/definition hash checks aligned when changing its assets.
- Docker runs have concurrency, history, subscriber-queue, output, timeout, memory, CPU, PID, and file-size bounds. Compatible runs also use a read-only root filesystem and unprivileged user.
- Containers drop all capabilities by default, enable `no-new-privileges`, use ownership labels, and are cleaned up on stop and backend shutdown. Keep added capabilities narrowly limited to workloads that install system packages.
- HTML runs in the frontend sandbox and does not need Docker.
- Python has extra support for:
  - import-based pip detection and common import/package aliases,
  - model-provided hints such as `pip install ...` and `Requirements: ...`,
  - one safe retry-repair pass for missing-module errors in non-GUI runs,
  - offline GUI/terminal apps through the prepared noVNC/xterm runtime where appropriate,
  - common web previews for Flask/FastAPI/Streamlit/Gradio/etc.,
  - common native apt packages for GUI/media/database libraries.

When changing language specs, run focused `tests.test_code_runner` coverage and at least one live Atlas Run smoke for affected languages.

## Releases

- Current release tag: `v1.3.1`
- `scripts/bump_atlas_version.py` updates this file and `README.md` together with the release manifests.
- `scripts/check_atlas_version.py` validates this file and `README.md` together with the release manifests.
- The consolidated `release` workflow verifies once, then publishes Windows MSI, Linux `.deb`/`.rpm`/AppImage, and unsigned macOS x64/arm64 DMG artifacts only after every build succeeds.
- Packaged macOS builds declare macOS 14 as their minimum system version, matching the locked native Python wheels.
- Release jobs stage expected artifacts explicitly and publish SHA-256 checksum files; a missing expected installer must fail the job.
- CI covers Linux, Windows, and macOS verification, while the security workflow audits Python/npm dependencies and CodeQL scans Python/JavaScript.
- Every published version needs `docs/releases/vX.Y.Z.md`; `.github/workflows/release.yml` uses that exact file as the GitHub Release body.
- Release tags must match every checked manifest version:
  - `AI.md`
  - `README.md`
  - `pyproject.toml`
  - `apps/atlas/package.json`
  - `apps/atlas/package-lock.json`
  - `apps/atlas/src-tauri/tauri.conf.json`
  - `apps/atlas/src-tauri/Cargo.toml`
  - `apps/atlas/src-tauri/Cargo.lock`

Safe release sequence:

1. Finish and verify app changes.
2. Bump every manifest with `scripts/bump_atlas_version.py`.
3. Run `scripts/check_atlas_version.py --tag vX.Y.Z`.
4. Commit the release change.
5. Tag it: `git tag -a vX.Y.Z -m "Atlas Chat vX.Y.Z"`.
6. Push `main`, then push the tag.
7. Watch `ci`, `codeql`, and the consolidated `release` workflow on GitHub Actions.
