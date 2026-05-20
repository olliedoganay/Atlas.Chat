import { act } from "react";
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
  execCode: vi.fn(),
  stopRunnerRun: vi.fn(),
  streamRunnerRun: vi.fn(),
}));

vi.mock("../lib/api", () => ({
  getRunnerStatus: apiMocks.getRunnerStatus,
  execCode: apiMocks.execCode,
  stopRunnerRun: apiMocks.stopRunnerRun,
  streamRunnerRun: apiMocks.streamRunnerRun,
}));

import { buildClientPreviewBlob, buildClientPreviewDocument, CLIENT_PREVIEW_SANDBOX, CodeRunnerPage } from "./CodeRunnerPage";

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
      onCloseRequested: vi.fn().mockResolvedValue(() => undefined),
      setTitle: vi.fn().mockResolvedValue(undefined),
    });
    apiMocks.getRunnerStatus.mockResolvedValue({ available: true });
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

  it("wraps HTML fragments in a runnable preview document", () => {
    const preview = buildClientPreviewDocument("<h1>Hello</h1>", "fragment-channel");

    expect(preview).toContain("<!DOCTYPE html>");
    expect(preview).toContain('<meta charset="utf-8" />');
    expect(preview).toContain("<body><h1>Hello</h1></body>");
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

    expect(fetchSpy).toHaveBeenCalledWith("http://127.0.0.1:43210/", { method: "GET", mode: "no-cors" });
    expect(container.textContent).toContain("Starting web preview...");
    expect(container.querySelector('iframe[title="Atlas web preview"]')).not.toBeNull();
    expect(container.querySelector(".runner-log-rail-button")).not.toBeNull();
    expect(container.querySelector(".runner-output")).toBeNull();

    act(() => {
      root.unmount();
    });
    container.remove();
    fetchSpy.mockRestore();
  });
});

function renderRunnerPage(): { root: Root; container: HTMLDivElement } {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  act(() => {
    root.render(<CodeRunnerPage />);
  });
  return { root, container };
}

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}
