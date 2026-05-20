# Atlas Desktop Shell

This folder contains the Tauri, React, and Vite desktop shell for Atlas.

Use the repo root [README](../../README.md) for install, release, and packaging instructions.

If you need to work inside this folder directly, these are the only commands that matter:

```powershell
npm run dev
npm test
npm run build
npm run build:release
npm run tauri dev
```

`npm run build` checks and builds only the frontend. `npm run build:release` also packages the Python backend resources used by Tauri release builds.

Generated folders such as `dist`, `output`, `src-tauri/resources/backend`, `src-tauri/resources/prompts`, and `src-tauri/target` are ignored and should not be committed.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/)
- [Tauri VS Code extension](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode)
- [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
