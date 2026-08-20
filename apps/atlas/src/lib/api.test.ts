import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const coreMocks = vi.hoisted(() => ({
  invoke: vi.fn(),
  isTauri: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: coreMocks.invoke,
  isTauri: coreMocks.isTauri,
}));

import {
  createMemory,
  getStatus,
  invalidateBackendRuntime,
  openExternalUrl,
  parseLocalBackendRuntime,
  streamRun,
  streamRunnerRun,
  type RunStatusEvent,
} from "./api";

const runtime = {
  host: "127.0.0.1",
  port: 43123,
  token: "test-token",
};

function runEvent(
  type: string,
  timestamp: string,
  payload: Record<string, unknown> = {},
  sequence?: number,
): RunStatusEvent {
  return { type, timestamp, payload, ...(sequence === undefined ? {} : { sequence }) };
}

function eventStream(events: Array<RunStatusEvent | Record<string, unknown>>) {
  const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
  return new Response(body, {
    headers: { "content-type": "text/event-stream" },
    status: 200,
  });
}

describe("API transport safety", () => {
  beforeEach(() => {
    invalidateBackendRuntime();
    coreMocks.isTauri.mockReturnValue(false);
    coreMocks.invoke.mockImplementation(async (command: string) => {
      if (command === "backend_runtime") {
        return runtime;
      }
      return undefined;
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("reconnects chat streams and delivers replayed events in contiguous sequence order", async () => {
    const eventA = runEvent("stage", "2026-01-01T00:00:00.000Z", { stage: "queued" }, 1);
    const eventB = runEvent("token", "2026-01-01T00:00:00.001Z", { text: "B" }, 2);
    const eventC = runEvent("token", "2026-01-01T00:00:00.002Z", { text: "C" }, 3);
    const terminal = runEvent("run_completed", "2026-01-01T00:00:00.003Z", {}, 4);
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(eventStream([eventA, eventC]))
      .mockResolvedValueOnce(eventStream([eventA, eventB, eventC, terminal]));
    vi.stubGlobal("fetch", fetchMock);

    const delivered: RunStatusEvent[] = [];
    const errors: string[] = [];
    streamRun("chat", "run-1", (event) => delivered.push(event), (error) => errors.push(error));

    await vi.waitFor(() => expect(delivered.some((event) => event.type === "run_completed")).toBe(true), {
      timeout: 1500,
    });

    expect(delivered).toEqual([eventA, eventB, eventC, terminal]);
    expect(errors).toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("advances across compacted sequence ranges and discards covered buffered events", async () => {
    const coveredRaw = runEvent("token", "2026-01-01T00:00:00.001Z", { text: "covered" }, 2);
    const compacted: RunStatusEvent = {
      ...runEvent("token", "2026-01-01T00:00:00.000Z", { text: "ABC" }, 1),
      sequence_end: 3,
    };
    const terminal = runEvent("run_completed", "2026-01-01T00:00:00.003Z", {}, 4);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(eventStream([coveredRaw, terminal, compacted])));

    const delivered: RunStatusEvent[] = [];
    const errors: string[] = [];
    streamRun("chat", "run-1", (event) => delivered.push(event), (error) => errors.push(error));

    await vi.waitFor(() => expect(delivered.some((event) => event.type === "run_completed")).toBe(true));

    expect(delivered).toEqual([compacted, terminal]);
    expect(errors).toEqual([]);
  });

  it("buffers an early terminal event until every preceding event is replayed", async () => {
    const eventA = runEvent("stage", "2026-01-01T00:00:00.000Z", { stage: "queued" }, 1);
    const eventB = runEvent("token", "2026-01-01T00:00:00.001Z", { text: "B" }, 2);
    const eventC = runEvent("token", "2026-01-01T00:00:00.002Z", { text: "C" }, 3);
    const terminal = runEvent("run_completed", "2026-01-01T00:00:00.003Z", {}, 4);
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(eventStream([eventA, terminal]))
      .mockResolvedValueOnce(eventStream([eventA, eventB, eventC, terminal]));
    vi.stubGlobal("fetch", fetchMock);

    const delivered: RunStatusEvent[] = [];
    const errors: string[] = [];
    streamRun("chat", "run-1", (event) => delivered.push(event), (error) => errors.push(error));

    await vi.waitFor(() => expect(delivered.some((event) => event.type === "run_completed")).toBe(true), {
      timeout: 1500,
    });

    expect(delivered).toEqual([eventA, eventB, eventC, terminal]);
    expect(errors).toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("keeps distinct events that have identical timestamps and payloads", async () => {
    const first = runEvent("token", "2026-01-01T00:00:00.000Z", { text: "A" }, 1);
    const second = runEvent("token", "2026-01-01T00:00:00.000Z", { text: "A" }, 2);
    const terminal = runEvent("run_completed", "2026-01-01T00:00:00.001Z", {}, 3);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(eventStream([first, second, terminal])));

    const delivered: RunStatusEvent[] = [];
    streamRun("chat", "run-1", (event) => delivered.push(event), vi.fn());

    await vi.waitFor(() => expect(delivered.some((event) => event.type === "run_completed")).toBe(true));
    expect(delivered).toEqual([first, second, terminal]);
  });

  it("does not replay a failed mutation against a refreshed runtime", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error("connection reset"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(createMemory("main", "Remember this")).rejects.toThrow("connection reset");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(coreMocks.invoke).toHaveBeenCalledTimes(1);
  });

  it("reports runner stream EOF when no exit event arrived", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      eventStream([{ type: "output", stream: "stdout", chunk: "partial" }]),
    );
    vi.stubGlobal("fetch", fetchMock);
    const errors: string[] = [];

    streamRunnerRun("runner-1", vi.fn(), (error) => errors.push(error));

    await vi.waitFor(() => expect(errors).toEqual(["Runner stream disconnected."]));
  });

  it("never falls through to window.open after a Tauri command rejection", async () => {
    coreMocks.isTauri.mockReturnValue(true);
    coreMocks.invoke.mockRejectedValue(new Error("URL rejected"));
    const openSpy = vi.spyOn(window, "open").mockReturnValue(null);

    await expect(openExternalUrl("https://example.com")).rejects.toThrow("URL rejected");

    expect(openSpy).not.toHaveBeenCalled();
  });

  it("accepts only explicit loopback browser backend origins", () => {
    expect(parseLocalBackendRuntime("http://127.0.0.1:43123", " token ")).toEqual({
      host: "127.0.0.1",
      port: 43123,
      token: "token",
      baseUrl: "http://127.0.0.1:43123",
    });
    expect(parseLocalBackendRuntime("http://localhost:8765")).toMatchObject({
      host: "localhost",
      port: 8765,
    });
    expect(parseLocalBackendRuntime(" ")).toBeNull();
  });

  it.each([
    "https://127.0.0.1:43123",
    "http://example.com:43123",
    "http://127.0.0.1:43123/api",
    "http://user:secret@127.0.0.1:43123",
  ])("rejects unsafe browser backend override %s", (url) => {
    expect(() => parseLocalBackendRuntime(url)).toThrow(/loopback/);
  });

  it("times out an unresponsive backend request", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: RequestInfo | URL, init?: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener(
              "abort",
              () => reject(new DOMException("The operation was aborted.", "AbortError")),
              { once: true },
            );
          }),
      ),
    );

    const statusRequest = getStatus();
    const rejection = expect(statusRequest).rejects.toThrow("did not respond within 15 seconds");
    await vi.advanceTimersByTimeAsync(15_000);

    await rejection;
    vi.useRealTimers();
  });
});
