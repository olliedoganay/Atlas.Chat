import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/webviewWindow", () => ({
  WebviewWindow: class MockWebviewWindow {
    once() {
      return undefined;
    }
  },
}));

import {
  buildRunnerRepairPrompt,
  consumePendingRun,
  createRunnerRepairRequest,
  isClientLanguage,
  MAX_RUNNER_REPAIR_DIAGNOSTICS_LENGTH,
  readRunnerRepairRequest,
  resolveRunnableLanguage,
  RUNNER_REPAIR_REQUEST_STORAGE_KEY,
  stashRunnerRepairRequest,
  stashPendingRun,
} from "./runner";

describe("runner helpers", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it("normalizes runnable language aliases", () => {
    expect(resolveRunnableLanguage(" py ")).toBe("python");
    expect(resolveRunnableLanguage("JS")).toBe("javascript");
    expect(resolveRunnableLanguage("c#")).toBe("csharp");
  });

  it("returns null for unsupported languages", () => {
    expect(resolveRunnableLanguage("brainfuck")).toBeNull();
    expect(resolveRunnableLanguage("")).toBeNull();
  });

  it("detects client-only languages", () => {
    expect(isClientLanguage("html")).toBe(true);
    expect(isClientLanguage("python")).toBe(false);
  });

  it("round-trips pending runs through localStorage", () => {
    stashPendingRun("token-1", {
      language: "python",
      code: "print('hello')",
    });

    expect(consumePendingRun("token-1")).toEqual({
      language: "python",
      code: "print('hello')",
    });
    expect(consumePendingRun("token-1")).toBeNull();
  });

  it("returns null for malformed pending payloads", () => {
    window.localStorage.setItem("atlas-runner:broken", "{not-json");

    expect(consumePendingRun("broken")).toBeNull();
    expect(window.localStorage.getItem("atlas-runner:broken")).toBeNull();
  });

  it("rejects unsupported, empty, and oversized pending runs", () => {
    window.localStorage.setItem(
      "atlas-runner:unsupported",
      JSON.stringify({ language: "brainfuck", code: "++++" }),
    );
    window.localStorage.setItem(
      "atlas-runner:empty",
      JSON.stringify({ language: "python", code: "" }),
    );
    window.localStorage.setItem(
      "atlas-runner:oversized",
      JSON.stringify({ language: "python", code: "x".repeat(1_000_001) }),
    );

    expect(consumePendingRun("unsupported")).toBeNull();
    expect(consumePendingRun("empty")).toBeNull();
    expect(consumePendingRun("oversized")).toBeNull();
  });

  it("normalizes pending language aliases before execution", () => {
    stashPendingRun("alias", { language: "c++", code: "int main() {}" });

    expect(consumePendingRun("alias")).toEqual({
      language: "cpp",
      code: "int main() {}",
    });
  });

  it("preserves the originating profile and thread with a pending run", () => {
    stashPendingRun("scoped", {
      language: "python",
      code: "print('private')",
      originUserId: "ollie",
      originThreadId: "snake-chat",
    });

    expect(consumePendingRun("scoped")).toEqual({
      language: "python",
      code: "print('private')",
      originUserId: "ollie",
      originThreadId: "snake-chat",
    });
  });

  it("creates a bounded, scoped, local repair request without changing its source", () => {
    const code = "print(`broken`)";
    const request = createRunnerRepairRequest({
      language: "py",
      code,
      diagnostics: `early\n${"x".repeat(MAX_RUNNER_REPAIR_DIAGNOSTICS_LENGTH + 100)}`,
      originUserId: "ollie",
      originThreadId: "snake-chat",
      now: 10_000,
    });
    stashRunnerRepairRequest(request);

    const stored = readRunnerRepairRequest(
      window.localStorage.getItem(RUNNER_REPAIR_REQUEST_STORAGE_KEY),
      10_001,
    );
    expect(stored).toMatchObject({
      language: "python",
      code,
      originUserId: "ollie",
      originThreadId: "snake-chat",
    });
    expect(stored?.diagnostics.length).toBeLessThanOrEqual(MAX_RUNNER_REPAIR_DIAGNOSTICS_LENGTH);
    expect(stored?.diagnostics).toContain("Earlier runner output omitted");
    expect(buildRunnerRepairPrompt(stored!)).toContain(code);
    expect(buildRunnerRepairPrompt(stored!)).toContain("inspect the full startup path");
  });

  it("rejects stale repair requests instead of delivering them to a later chat", () => {
    const request = createRunnerRepairRequest({
      language: "python",
      code: "raise RuntimeError()",
      diagnostics: "RuntimeError",
      originUserId: "ollie",
      originThreadId: "snake-chat",
      now: 1_000,
    });

    expect(readRunnerRepairRequest(JSON.stringify(request), 1_000 + 11 * 60 * 1000)).toBeNull();
  });

  it("notifies another local Atlas window after durably storing the repair request", () => {
    const postMessageSpy = vi.fn();
    const closeSpy = vi.fn();
    vi.stubGlobal("BroadcastChannel", class {
      postMessage = postMessageSpy;
      close = closeSpy;
    });
    const request = createRunnerRepairRequest({
      language: "python",
      code: "print('repair me')",
      diagnostics: "exit 1",
      originUserId: "ollie",
      originThreadId: "snake-chat",
    });

    stashRunnerRepairRequest(request);

    expect(window.localStorage.getItem(RUNNER_REPAIR_REQUEST_STORAGE_KEY)).toContain(request.requestId);
    expect(postMessageSpy).toHaveBeenCalledWith({ requestId: request.requestId });
    expect(closeSpy).toHaveBeenCalledTimes(1);
  });
});
