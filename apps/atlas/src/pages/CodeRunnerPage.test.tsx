import { act, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const tauriWindowMocks = vi.hoisted(() => ({
  getCurrentWindow: vi.fn(),
}));

vi.mock("@tauri-apps/api/window", () => ({
  getCurrentWindow: tauriWindowMocks.getCurrentWindow,
}));

vi.mock("react-router-dom", () => ({
  useParams: () => ({ token: "token-1" }),
}));

const apiMocks = vi.hoisted(() => ({
  getRunnerStatus: vi.fn(),
  getPythonGuiRuntimeStatus: vi.fn(),
  prepareRunnerRuntime: vi.fn(),
  execCode: vi.fn(),
  stopRunnerRun: vi.fn(),
  streamRunnerRun: vi.fn(),
}));

vi.mock("../lib/api", () => ({
  getRunnerStatus: apiMocks.getRunnerStatus,
  getPythonGuiRuntimeStatus: apiMocks.getPythonGuiRuntimeStatus,
  prepareRunnerRuntime: apiMocks.prepareRunnerRuntime,
  execCode: apiMocks.execCode,
  stopRunnerRun: apiMocks.stopRunnerRun,
  streamRunnerRun: apiMocks.streamRunnerRun,
}));

import {
  buildClientPreviewBlob,
  buildClientPreviewDocument,
  buildRunnerRepairDiagnostics,
  CLIENT_PREVIEW_CSP,
  CLIENT_PREVIEW_SANDBOX,
  CodeRunnerPage,
  MAX_CLIENT_PREVIEW_CONSOLE_CHARS,
  RUNNER_PREVIEW_PROBE_TIMEOUT_MS,
  SERVER_PREVIEW_SANDBOX,
} from "./CodeRunnerPage";
import {
  readRunnerRepairRequest,
  RUNNER_REPAIR_REQUEST_STORAGE_KEY,
} from "../lib/runner";

function readBlobText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read blob."));
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.readAsText(blob);
  });
}

describe("CodeRunnerPage client preview", () => {
  beforeEach(() => {
    window.localStorage.clear();
    tauriWindowMocks.getCurrentWindow.mockReturnValue({
      close: vi.fn().mockResolvedValue(undefined),
      onCloseRequested: vi.fn().mockResolvedValue(() => undefined),
      setTitle: vi.fn().mockResolvedValue(undefined),
    });
    apiMocks.getRunnerStatus.mockResolvedValue({ available: true });
    apiMocks.prepareRunnerRuntime.mockResolvedValue({ required: false, runtime: null, started: false });
    apiMocks.execCode.mockResolvedValue({ run_id: "run-1" });
    apiMocks.stopRunnerRun.mockResolvedValue({ run_id: "run-1", status: "stopping" });
    apiMocks.streamRunnerRun.mockReturnValue(vi.fn());
  });

  afterEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
  });

  it("keeps complete HTML documents intact while adding preview diagnostics", async () => {
    const code = [
      "<!DOCTYPE html>",
      "<html>",
      "<head><style>canvas{background:#111}</style></head>",
      "<body><canvas id=\"gameCanvas\"></canvas><script>window.atlasPreviewRan=true;</script></body>",
      "</html>",
    ].join("");

    const preview = buildClientPreviewDocument(code, "test-channel");

    expect(preview).toContain("<!DOCTYPE html>");
    expect(preview).toContain("<style>canvas{background:#111}</style>");
    expect(preview).toContain("<canvas id=\"gameCanvas\"></canvas>");
    expect(preview).toContain("<script>window.atlasPreviewRan=true;</script>");
    expect(preview).toContain('http-equiv="Content-Security-Policy"');
    expect(preview).toContain("style-src-elem");
    expect(preview).toContain("atlas-client-preview");
    expect(preview).toContain("test-channel");
  });

  it("inserts the offline policy into the real head despite decoy head markup", () => {
    const code = [
      "<!DOCTYPE html>",
      "<!-- <head id=\"decoy\"> -->",
      "<html><body><script id=\"user-script\">window.previewLoaded=true;</script></body></html>",
    ].join("");

    const preview = buildClientPreviewDocument(code, "decoy-channel");
    const parsed = new DOMParser().parseFromString(preview, "text/html");
    const policy = parsed.head.querySelector(
      'meta[http-equiv="Content-Security-Policy"]',
    );

    expect(policy).not.toBeNull();
    expect(policy?.getAttribute("content")).toBe(CLIENT_PREVIEW_CSP);
    expect(parsed.querySelector("#user-script")).not.toBeNull();
  });

  it("wraps HTML fragments in a runnable preview document", () => {
    const preview = buildClientPreviewDocument("<h1>Hello</h1>", "fragment-channel");
    const parsed = new DOMParser().parseFromString(preview, "text/html");

    expect(preview).toContain("<!DOCTYPE html>");
    expect(preview).toContain('<meta charset="utf-8">');
    expect(parsed.body.innerHTML).toContain("<h1>Hello</h1>");
    expect(preview).toContain("fragment-channel");
  });

  it("can still build raw HTML blobs for callers that need them", async () => {
    const code = "<!DOCTYPE html><html><body>raw</body></html>";
    const blob = buildClientPreviewBlob(code);

    expect(blob.type).toBe("text/html;charset=utf-8");
    await expect(readBlobText(blob)).resolves.toBe(code);
  });

  it("allows scripts without granting same-origin access", () => {
    const tokens = CLIENT_PREVIEW_SANDBOX.split(" ");

    expect(tokens).toContain("allow-scripts");
    expect(tokens).toContain("allow-pointer-lock");
    expect(tokens).not.toContain("allow-same-origin");
    expect(tokens).not.toContain("allow-popups");
    expect(tokens).not.toContain("allow-modals");
  });

  it("keeps client previews offline and bounds relayed console output", () => {
    const preview = buildClientPreviewDocument("<script>console.log('ready')</script>", "offline-channel");

    expect(CLIENT_PREVIEW_CSP).toContain("connect-src 'none'");
    expect(CLIENT_PREVIEW_CSP).toContain("form-action 'none'");
    expect(CLIENT_PREVIEW_CSP).not.toMatch(/\bhttps?:/);
    expect(CLIENT_PREVIEW_CSP).not.toMatch(/\bwss?:/);
    expect(preview).toContain(`const maxConsoleChars = ${MAX_CLIENT_PREVIEW_CONSOLE_CHARS};`);
    expect(preview).toContain("[atlas-preview] output truncated");
  });

  it("installs sandbox-safe storage fallbacks before user scripts", () => {
    const code =
      '<!DOCTYPE html><html><head></head><body><script>localStorage.setItem("score", "1");</script></body></html>';
    const preview = buildClientPreviewDocument(code, "storage-channel");

    expect(preview).toContain('installStorageFallback("localStorage")');
    expect(preview).toContain('installStorageFallback("sessionStorage")');
    expect(preview.indexOf('installStorageFallback("localStorage")')).toBeLessThan(
      preview.indexOf('localStorage.setItem("score", "1")'),
    );
    expect(CLIENT_PREVIEW_SANDBOX.split(" ")).not.toContain("allow-same-origin");
  });

  it("shows the runnable source beside the preview and lets the user hide it", async () => {
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "html", code: "<h1>Hello Atlas</h1>" }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();

    expect(container.querySelector(".runner-source-panel")?.textContent).toContain("<h1>Hello Atlas</h1>");
    expect(container.querySelector(".runner-iframe")).not.toBeNull();

    const sourceButton = Array.from(container.querySelectorAll("button")).find((button) =>
      button.textContent?.includes("Source"),
    );
    act(() => {
      sourceButton?.click();
    });

    expect(container.querySelector(".runner-source-panel")).toBeNull();
    expect(container.querySelector(".runner-iframe")).not.toBeNull();

    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("keeps the runner usable when the Tauri window API is unavailable", async () => {
    tauriWindowMocks.getCurrentWindow.mockImplementation(() => {
      throw new Error("Tauri window unavailable");
    });
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "html", code: "<h1>Browser preview</h1>" }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();

    expect(container.querySelector(".runner-source-panel")?.textContent).toContain("Browser preview");
    expect(container.querySelector(".runner-iframe")).not.toBeNull();

    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("stops an active server run when the run page unmounts", async () => {
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "python", code: "print('hello')" }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();
    expect(apiMocks.execCode).toHaveBeenCalledWith("python", "print('hello')");

    act(() => {
      root.unmount();
    });
    container.remove();

    expect(apiMocks.stopRunnerRun).toHaveBeenCalledWith("run-1");
  });

  it("consumes and starts a server run once under React StrictMode", async () => {
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "python", code: "print('strict')" }),
    );
    const { root, container } = renderRunnerPage(true);

    await flushEffects();

    expect(apiMocks.execCode).toHaveBeenCalledTimes(1);
    expect(apiMocks.execCode).toHaveBeenCalledWith("python", "print('strict')");
    expect(container.textContent).not.toContain("lost its payload");

    act(() => root.unmount());
    container.remove();
  });

  it("shows trusted runtime preparation progress before starting Python GUI code", async () => {
    vi.useFakeTimers();
    const preparingRuntime = {
      name: "python-gui",
      version: "1.0.1",
      image: "localhost/atlas-python-gui-runtime:1.0.1",
      state: "preparing",
      message: "STEP 3/8: installing pinned dependencies",
      error: null,
      progress: 0.35,
      bundled_packages: ["numpy==2.5.2", "pygame==2.6.1"],
      execution_network: "internal-preview-only",
      submitted_code_used_during_preparation: false,
      log_tail: ["STEP 3/8: installing pinned dependencies"],
    };
    apiMocks.prepareRunnerRuntime.mockResolvedValue({
      required: true,
      runtime: preparingRuntime,
      started: true,
    });
    apiMocks.getPythonGuiRuntimeStatus.mockResolvedValue({
      ...preparingRuntime,
      state: "ready",
      message: "Secure offline Python GUI runtime is ready.",
      progress: 1,
      log_tail: ["Successfully tagged localhost/atlas-python-gui-runtime:1.0.1"],
    });
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "python", code: "import pygame\npygame.init()" }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();
    expect(container.textContent).toContain("Preparing runtime...");
    expect(container.textContent).toContain("Offline runtime 35%");
    expect(container.textContent).toContain("submitted code is not mounted or executed");
    expect(apiMocks.execCode).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(750);
    });
    await flushEffects();

    expect(apiMocks.getPythonGuiRuntimeStatus).toHaveBeenCalledTimes(1);
    expect(apiMocks.execCode).toHaveBeenCalledWith("python", "import pygame\npygame.init()");
    act(() => root.unmount());
    container.remove();
    vi.useRealTimers();
  });

  it("waits for active-run cleanup before closing the runner window", async () => {
    let closeHandler: ((event: { preventDefault: () => void }) => Promise<void>) | undefined;
    const close = vi.fn().mockResolvedValue(undefined);
    tauriWindowMocks.getCurrentWindow.mockReturnValue({
      close,
      onCloseRequested: vi.fn().mockImplementation(async (handler) => {
        closeHandler = handler;
        return () => undefined;
      }),
      setTitle: vi.fn().mockResolvedValue(undefined),
    });
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "python", code: "print('close')" }),
    );
    const { root, container } = renderRunnerPage();
    await flushEffects();
    const preventDefault = vi.fn();

    await act(async () => {
      await closeHandler?.({ preventDefault });
    });

    expect(preventDefault).toHaveBeenCalledTimes(1);
    expect(apiMocks.stopRunnerRun).toHaveBeenCalledWith("run-1");
    expect(close).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
    container.remove();
  });

  it("stops a server run that finishes starting after the window unmounts", async () => {
    let resolveStart: ((value: { run_id: string }) => void) | null = null;
    apiMocks.execCode.mockReturnValue(
      new Promise((resolve) => {
        resolveStart = resolve;
      }),
    );
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "python", code: "print('late')" }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();
    expect(apiMocks.execCode).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
    container.remove();

    await act(async () => {
      resolveStart?.({ run_id: "run-late" });
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(apiMocks.stopRunnerRun).toHaveBeenCalledWith("run-late");
  });

  it("moves to an actionable error state when the runner stream disconnects", async () => {
    apiMocks.streamRunnerRun.mockImplementation((_runId, _onEvent, onError) => {
      onError("Runner stream disconnected.");
      return vi.fn();
    });
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "python", code: "print('hello')" }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();

    expect(container.textContent).toContain("Error");
    expect(container.textContent).toContain("Runner stream disconnected.");
    expect(container.textContent).toContain("Rerun");

    act(() => root.unmount());
    container.remove();
  });

  it("creates a scoped repair draft from a failed run without mutating or rerunning source", async () => {
    const source = "import pygame\npygame.math.sin(1)";
    const close = vi.fn().mockResolvedValue(undefined);
    tauriWindowMocks.getCurrentWindow.mockReturnValue({
      close,
      onCloseRequested: vi.fn().mockResolvedValue(() => undefined),
      setTitle: vi.fn().mockResolvedValue(undefined),
    });
    apiMocks.streamRunnerRun.mockImplementation((_runId, onEvent) => {
      onEvent({
        type: "output",
        stream: "stdout",
        chunk: "[atlas-runner] using prepared offline Python GUI runtime 1.0.1\n",
      });
      onEvent({
        type: "output",
        stream: "stderr",
        chunk: "AttributeError: module 'pygame.math' has no attribute 'sin'\n",
      });
      onEvent({ type: "exit", code: 1, duration_ms: 2500 });
      return vi.fn();
    });
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({
        language: "python",
        code: source,
        originUserId: "ollie",
        originThreadId: "snake-chat",
      }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();
    expect(container.textContent).toContain("Fix with Atlas");
    expect(container.textContent).toContain("Rerun same source");
    const fixButton = Array.from(container.querySelectorAll("button")).find((button) =>
      button.textContent?.includes("Fix with Atlas"),
    );
    await act(async () => {
      fixButton?.click();
      await Promise.resolve();
    });

    const request = readRunnerRepairRequest(
      window.localStorage.getItem(RUNNER_REPAIR_REQUEST_STORAGE_KEY),
    );
    expect(request).toMatchObject({
      code: source,
      originUserId: "ollie",
      originThreadId: "snake-chat",
    });
    expect(request?.diagnostics).toContain("AttributeError");
    expect(request?.diagnostics).not.toContain("using prepared offline");
    expect(apiMocks.execCode).toHaveBeenCalledTimes(1);
    expect(close).toHaveBeenCalledTimes(1);

    act(() => root.unmount());
    container.remove();
  });

  it("keeps useful tracebacks while removing benign runner setup lines", () => {
    expect(buildRunnerRepairDiagnostics({
      errorMessage: null,
      exitCode: 1,
      output: [
        { stream: "stdout", text: "[atlas-runner] GUI ready on port 6080\n" },
        { stream: "stderr", text: "Traceback:\nAttributeError: bad API\n" },
      ],
    })).toBe("Process exited with code 1.\nTraceback:\nAttributeError: bad API");
  });

  it("shows a retry path when Docker is unavailable", async () => {
    apiMocks.getRunnerStatus.mockResolvedValue({ available: false, reason: "Docker is stopped." });
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "python", code: "print('hello')" }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();

    expect(container.textContent).toContain("Docker Desktop isn't running");
    expect(container.textContent).toContain("Docker is stopped.");
    expect(container.textContent).toContain("Retry");

    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("hides the GUI viewer after a VNC-backed run exits", async () => {
    apiMocks.execCode.mockResolvedValue({ run_id: "run-1", vnc_url: "http://127.0.0.1:6080/vnc.html" });
    apiMocks.streamRunnerRun.mockImplementation((_runId, onEvent) => {
      onEvent({ type: "exit", code: 0, duration_ms: 1620 });
      return vi.fn();
    });
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "python", code: "import tkinter\nprint('done')" }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();

    expect(container.textContent).toContain("Done");
    expect(container.textContent).toContain("No output.");
    expect(container.querySelector(".runner-vnc-frame")).toBeNull();
    expect(container.querySelector(".runner-vnc-placeholder")).toBeNull();

    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("collapses VNC runner logs by default and lets the user open them", async () => {
    apiMocks.execCode.mockResolvedValue({ run_id: "run-1", vnc_url: "http://127.0.0.1:6080/vnc.html" });
    apiMocks.streamRunnerRun.mockImplementation((_runId, onEvent) => {
      onEvent({ type: "output", stream: "stdout", chunk: "[atlas-runner] GUI ready on port 6080\n" });
      onEvent({ type: "output", stream: "stderr", chunk: "DeprecationWarning: test warning\n" });
      return vi.fn();
    });
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "python", code: "import tkinter\nroot = tkinter.Tk()" }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();

    const logsButton = container.querySelector<HTMLButtonElement>(".runner-log-rail-button");
    expect(logsButton).not.toBeNull();
    expect(container.querySelector(".runner-output")).toBeNull();
    expect(container.textContent).not.toContain("DeprecationWarning");

    act(() => {
      logsButton?.click();
    });

    expect(container.querySelector(".runner-output")).not.toBeNull();
    expect(container.textContent).toContain("DeprecationWarning: test warning");

    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("waits for the GUI ready event before loading the VNC frame", async () => {
    apiMocks.execCode.mockResolvedValue({ run_id: "run-1", vnc_url: "http://127.0.0.1:6080/vnc.html" });
    apiMocks.streamRunnerRun.mockImplementation((_runId, onEvent) => {
      onEvent({ type: "output", stream: "stdout", chunk: "[atlas-runner] installing system dependencies\n" });
      return vi.fn();
    });
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "python", code: "import tkinter\nroot = tkinter.Tk()" }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();

    expect(container.textContent).toContain("Installing GUI dependencies...");
    expect(container.querySelector(".runner-vnc-frame")).toBeNull();
    expect(container.querySelector(".runner-output")).not.toBeNull();
    expect(container.textContent).toContain("installing system dependencies");

    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("shows a web preview for server runs with an exposed web URL", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("", { status: 200 }));
    apiMocks.execCode.mockResolvedValue({ run_id: "run-1", web_url: "http://127.0.0.1:43210/" });
    apiMocks.streamRunnerRun.mockImplementation((_runId, onEvent) => {
      onEvent({ type: "output", stream: "stdout", chunk: "[atlas-runner] web preview will use container port 5000\n" });
      return vi.fn();
    });
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({ language: "python", code: "from flask import Flask\napp = Flask(__name__)" }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();
    await flushEffects();

    expect(fetchSpy).toHaveBeenCalledWith(
      "http://127.0.0.1:43210/",
      expect.objectContaining({
        method: "GET",
        mode: "no-cors",
        signal: expect.any(AbortSignal),
      }),
    );
    expect(container.textContent).toContain("Starting web preview...");
    expect(container.querySelector('iframe[title="Atlas web preview"]')).not.toBeNull();
    const previewFrame = container.querySelector<HTMLIFrameElement>('iframe[title="Atlas web preview"]');
    expect(previewFrame?.getAttribute("sandbox")).toBe(SERVER_PREVIEW_SANDBOX);
    expect(previewFrame?.getAttribute("referrerpolicy")).toBe("no-referrer");
    const sandboxTokens = SERVER_PREVIEW_SANDBOX.split(" ");
    expect(sandboxTokens).toContain("allow-scripts");
    expect(sandboxTokens).toContain("allow-same-origin");
    expect(sandboxTokens).not.toContain("allow-popups");
    expect(sandboxTokens).not.toContain("allow-top-navigation");
    expect(sandboxTokens).not.toContain("allow-downloads");
    expect(container.querySelector(".runner-log-rail-button")).not.toBeNull();
    expect(container.querySelector(".runner-output")).toBeNull();

    act(() => {
      root.unmount();
    });
    container.remove();
    fetchSpy.mockRestore();
  });

  it("aborts a stalled server-preview readiness probe", async () => {
    vi.useFakeTimers();
    let probeSignal: AbortSignal | null = null;
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(
      (_input, init) =>
        new Promise((_resolve, reject) => {
          probeSignal = init?.signal ?? null;
          probeSignal?.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          );
        }),
    );
    apiMocks.execCode.mockResolvedValue({
      run_id: "run-1",
      web_url: "http://127.0.0.1:43210/",
    });
    window.localStorage.setItem(
      "atlas-runner:token-1",
      JSON.stringify({
        language: "python",
        code: "from flask import Flask\napp = Flask(__name__)",
      }),
    );
    const { root, container } = renderRunnerPage();

    await flushEffects();
    expect(probeSignal).not.toBeNull();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(RUNNER_PREVIEW_PROBE_TIMEOUT_MS);
    });

    expect((probeSignal as AbortSignal | null)?.aborted).toBe(true);
    act(() => root.unmount());
    container.remove();
    fetchSpy.mockRestore();
    vi.useRealTimers();
  });
});

function renderRunnerPage(strict = false): { root: Root; container: HTMLDivElement } {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  act(() => {
    root.render(strict ? <StrictMode><CodeRunnerPage /></StrictMode> : <CodeRunnerPage />);
  });
  return { root, container };
}

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}
