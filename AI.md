# Atlas Repo Note

- Repo: `olliedoganay/Atlas.Chat`
- Current version: `1.1.1`
- Stack: Tauri 2 + React/Vite frontend, Rust desktop shell, Python FastAPI backend, Ollama, Docker-backed code runner.
- Runtime shape: local-first desktop app; Tauri launches a loopback backend protected by a per-launch instance token.

## Main Paths

- UI: `apps/atlas/src`
- Tauri shell/backend launcher: `apps/atlas/src-tauri/src/lib.rs`
- Backend API/service: `src/atlas_local/api.py`, `src/atlas_local/api_service.py`
- LLM/Ollama provider: `src/atlas_local/llm.py`, `src/atlas_local/providers`
- Config/version/context settings: `src/atlas_local/config.py`
- Code runner: `src/atlas_local/code_runner.py`
- Answer prompt: `prompts/answer.md`
- Tests: `tests`, `apps/atlas/src/**/*.test.ts*`
- Release workflows: `.github/workflows/release-windows.yml`, `.github/workflows/release-linux.yml`

## Setup

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
cd apps\atlas
npm install
cd ..\..
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
cd apps/atlas
npm install
cd ../..
```

## Common Commands

After activating the repo virtual environment:

- Start Windows source app: `.\scripts\start_atlas_dev.ps1`
- Force Windows desktop rebuild: `.\scripts\start_atlas_dev.ps1 -RebuildDesktop`
- Start Linux/macOS source app: `cd apps/atlas && npm run tauri dev`
- Version check: `python scripts/check_atlas_version.py --tag vX.Y.Z`
- Bump all release manifests: `python scripts/bump_atlas_version.py X.Y.Z`
- Backend tests: `python -m unittest discover -s tests`
- Focused runner tests: `python -m unittest tests.test_code_runner`
- Frontend tests: `cd apps/atlas && npm test`
- Frontend build: `cd apps/atlas && npm run build`
- Windows full verification: `.\scripts\verify_repo.ps1`
- Windows release bundle: `.\scripts\build_atlas_release.ps1`

Plain `pytest` from repo root is intended to be safe when pytest is installed, but `python -m unittest discover -s tests` is the backend baseline.

## Current Feature Notes

- CI and release workflows use Python `3.14` and Node `20`.
- Settings > Models has a global Ollama context-window slider. Atlas sends the selected value as `num_ctx`; `Auto` follows Ollama defaults.
- Context compaction is automatic/manual, uses the effective context window, and has visible compaction status/animation in the composer area.
- Only model outputs, thinking traces, and compaction summaries should be text-selectable in the app UI.
- The titlebar is custom and compact; keep it visually close to VS Code-style density.
- Offline Ollama states should stay compact. Avoid large blocking banners when the empty-state already explains what is wrong.

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
- JS/TS/Go/Rust/Ruby/Perl/R/Dart request bridge networking because package managers may need outbound access.
- Non-GUI Docker runs otherwise default to isolated networking unless dependency resolution or port exposure requires bridge mode.
- HTML runs in the frontend sandbox and does not need Docker.
- Python has extra support for:
  - import-based pip detection and common import/package aliases,
  - model-provided hints such as `pip install ...` and `Requirements: ...`,
  - one safe retry-repair pass for missing-module errors,
  - GUI/terminal apps through noVNC/xterm where appropriate,
  - common web previews for Flask/FastAPI/Streamlit/Gradio/etc.,
  - common native apt packages for GUI/media/database libraries.

When changing language specs, run focused `tests.test_code_runner` coverage and at least one live Atlas Run smoke for affected languages.

## Releases

- Current release tag: `v1.1.1`
- `scripts/bump_atlas_version.py` updates this file and `README.md` together with the release manifests.
- `scripts/check_atlas_version.py` validates this file and `README.md` together with the release manifests.
- Windows release workflow publishes MSI artifacts.
- Linux release workflow publishes `.deb`, `.rpm`, and AppImage artifacts.
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
7. Watch `ci`, `release-windows`, and `release-linux` on GitHub Actions.
