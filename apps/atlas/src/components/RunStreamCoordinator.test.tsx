import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { RunStatusEvent } from "../lib/api";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const apiMocks = vi.hoisted(() => ({
  getRun: vi.fn(),
  getThreadRuns: vi.fn(),
  streamRun: vi.fn(),
}));

vi.mock("../lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("../lib/api")>();
  return {
    ...original,
    getRun: apiMocks.getRun,
    getThreadRuns: apiMocks.getThreadRuns,
    streamRun: apiMocks.streamRun,
  };
});

import { useAtlasStore } from "../store/useAtlasStore";
import { RunStreamCoordinator } from "./RunStreamCoordinator";

const initialState = useAtlasStore.getInitialState();

describe("RunStreamCoordinator termination semantics", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useAtlasStore.setState(initialState, true);
    apiMocks.getRun.mockReset();
    apiMocks.getThreadRuns.mockReset();
    apiMocks.getThreadRuns.mockResolvedValue([]);
    apiMocks.streamRun.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    useAtlasStore.setState(initialState, true);
    document.body.innerHTML = "";
  });

  it("treats the backend user-stop terminal event as a neutral cancellation", async () => {
    let onEvent: ((event: RunStatusEvent) => void) | null = null;
    apiMocks.streamRun.mockImplementation((_mode, _runId, handleEvent) => {
      onEvent = handleEvent;
      return vi.fn();
    });
    useAtlasStore.getState().beginRun("run-1", "chat", "hello", "user-1", "main", []);
    const rendered = renderCoordinator();
    await flushEffects();

    act(() => {
      onEvent?.({
        type: "run_failed",
        timestamp: "2026-08-20T00:00:00Z",
        payload: { error: "Run stopped by user." },
      });
    });

    const state = useAtlasStore.getState();
    expect(state.isStreaming).toBe(false);
    expect(state.currentRunId).toBeNull();
    expect(state.currentStage).toBe("cancelled");
    expect(state.liveError).toBe("");
    unmountCoordinator(rendered);
  });

  it("keeps genuine backend failures in the error state", async () => {
    let onEvent: ((event: RunStatusEvent) => void) | null = null;
    apiMocks.streamRun.mockImplementation((_mode, _runId, handleEvent) => {
      onEvent = handleEvent;
      return vi.fn();
    });
    useAtlasStore.getState().beginRun("run-2", "chat", "hello", "user-1", "main", []);
    const rendered = renderCoordinator();
    await flushEffects();

    act(() => {
      onEvent?.({
        type: "run_failed",
        timestamp: "2026-08-20T00:00:00Z",
        payload: { error: "Provider disconnected." },
      });
    });

    const state = useAtlasStore.getState();
    expect(state.isStreaming).toBe(false);
    expect(state.currentStage).toBe("failed");
    expect(state.liveError).toBe("Provider disconnected.");
    unmountCoordinator(rendered);
  });

  it("also recognizes cancellation when recovering a disconnected stream from its artifact", async () => {
    let onStreamError: ((message: string) => void) | null = null;
    apiMocks.streamRun.mockImplementation((_mode, _runId, _handleEvent, handleError) => {
      onStreamError = handleError;
      return vi.fn();
    });
    apiMocks.getRun.mockResolvedValue({
      run_id: "run-3",
      status: "failed",
      error: "Run stopped by user.",
    });
    useAtlasStore.getState().beginRun("run-3", "chat", "hello", "user-1", "main", []);
    const rendered = renderCoordinator();
    await flushEffects();

    await act(async () => {
      onStreamError?.("Atlas stream disconnected.");
      await vi.waitFor(() => expect(useAtlasStore.getState().currentStage).toBe("cancelled"));
    });

    expect(useAtlasStore.getState().liveError).toBe("");
    expect(useAtlasStore.getState().isStreaming).toBe(false);
    unmountCoordinator(rendered);
  });

  it("reattaches and replays an active run after volatile store state is lost on reload", async () => {
    apiMocks.getThreadRuns.mockResolvedValue([
      {
        run_id: "run-recovered",
        mode: "chat",
        user_id: "user-1",
        thread_id: "main",
        prompt: "recover this exact prompt",
        status: "running",
        started_at: "2026-08-20T10:00:00Z",
        answer: "",
        events: [],
      },
    ]);
    apiMocks.streamRun.mockReturnValue(vi.fn());
    useAtlasStore.setState({ currentUserId: "user-1", currentThreadId: "main" });
    const rendered = renderCoordinator({ canRecoverRun: true });

    await act(async () => {
      await vi.waitFor(() => {
        expect(apiMocks.streamRun).toHaveBeenCalledWith(
          "chat",
          "run-recovered",
          expect.any(Function),
          expect.any(Function),
        );
      });
    });

    expect(useAtlasStore.getState()).toMatchObject({
      currentRunId: "run-recovered",
      currentRunMode: "chat",
      activeRunUserId: "user-1",
      activeRunThreadId: "main",
      pendingPrompt: "recover this exact prompt",
      isStreaming: true,
    });
    unmountCoordinator(rendered);
  });

  it("waits for the backend refetch before deciding that a cached thread has no recoverable run", async () => {
    let resolveRuns: ((runs: Array<Record<string, unknown>>) => void) | null = null;
    apiMocks.getThreadRuns.mockReturnValue(
      new Promise((resolve) => {
        resolveRuns = resolve;
      }),
    );
    useAtlasStore.setState({ currentUserId: "user-1", currentThreadId: "main" });
    const queryClient = new QueryClient({
      defaultOptions: {
        mutations: { retry: false },
        queries: { retry: false },
      },
    });
    queryClient.setQueryData(["thread-runs", "user-1", "main"], []);
    apiMocks.streamRun.mockReturnValue(vi.fn());
    const rendered = renderCoordinator({ canRecoverRun: true, queryClient });

    await flushEffects();
    expect(apiMocks.getThreadRuns).toHaveBeenCalledTimes(1);
    expect(apiMocks.streamRun).not.toHaveBeenCalled();

    await act(async () => {
      resolveRuns?.([
        {
          run_id: "run-from-refetch",
          mode: "chat",
          user_id: "user-1",
          thread_id: "main",
          prompt: "recover after refetch",
          status: "running",
          started_at: "2026-08-20T10:00:00Z",
          answer: "",
          events: [],
        },
      ]);
      await vi.waitFor(() => expect(apiMocks.streamRun).toHaveBeenCalledTimes(1));
    });

    expect(useAtlasStore.getState()).toMatchObject({
      currentRunId: "run-from-refetch",
      pendingPrompt: "recover after refetch",
      isStreaming: true,
    });
    unmountCoordinator(rendered);
  });

  it("does not resurrect a recovered run from stale active cache data after its terminal event", async () => {
    let onEvent: ((event: RunStatusEvent) => void) | null = null;
    apiMocks.getThreadRuns.mockResolvedValue([
      {
        run_id: "run-recovered",
        mode: "chat",
        user_id: "user-1",
        thread_id: "main",
        prompt: "recover once",
        status: "running",
        started_at: "2026-08-20T10:00:00Z",
        answer: "",
        events: [],
      },
    ]);
    apiMocks.streamRun.mockImplementation((_mode, _runId, handleEvent) => {
      onEvent = handleEvent;
      return vi.fn();
    });
    useAtlasStore.setState({ currentUserId: "user-1", currentThreadId: "main" });
    const rendered = renderCoordinator({ canRecoverRun: true });
    await act(async () => {
      await vi.waitFor(() => expect(apiMocks.streamRun).toHaveBeenCalledTimes(1));
    });

    await act(async () => {
      onEvent?.({
        type: "run_completed",
        timestamp: "2026-08-20T10:01:00Z",
        payload: { answer: "done" },
      });
      await vi.waitFor(() => expect(useAtlasStore.getState().currentStage).toBe("completed"));
    });
    await flushEffects();

    expect(useAtlasStore.getState().currentRunId).toBeNull();
    expect(useAtlasStore.getState().isStreaming).toBe(false);
    expect(apiMocks.streamRun).toHaveBeenCalledTimes(1);
    unmountCoordinator(rendered);
  });

  it("does not resurrect terminal artifacts and never queries a locked profile", async () => {
    apiMocks.getThreadRuns.mockResolvedValue([
      {
        run_id: "run-terminal",
        mode: "chat",
        user_id: "user-1",
        thread_id: "main",
        prompt: "finished",
        status: "completed",
        started_at: "2026-08-20T10:00:00Z",
        completed_at: "2026-08-20T10:01:00Z",
        answer: "done",
        events: [],
      },
    ]);
    useAtlasStore.setState({ currentUserId: "user-1", currentThreadId: "main" });
    const terminalRendered = renderCoordinator({ canRecoverRun: true });
    await act(async () => {
      await vi.waitFor(() => expect(apiMocks.getThreadRuns).toHaveBeenCalledTimes(1));
    });
    expect(useAtlasStore.getState().currentRunId).toBeNull();
    expect(apiMocks.streamRun).not.toHaveBeenCalled();
    unmountCoordinator(terminalRendered);

    apiMocks.getThreadRuns.mockClear();
    const lockedRendered = renderCoordinator({ canRecoverRun: false });
    await flushEffects();
    expect(apiMocks.getThreadRuns).not.toHaveBeenCalled();
    expect(useAtlasStore.getState().currentRunId).toBeNull();
    unmountCoordinator(lockedRendered);
  });
});

function renderCoordinator({
  canRecoverRun = true,
  queryClient = new QueryClient({
    defaultOptions: {
      mutations: { retry: false },
      queries: { retry: false },
    },
  }),
}: {
  canRecoverRun?: boolean;
  queryClient?: QueryClient;
} = {}) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(
      createElement(
        QueryClientProvider,
        { client: queryClient },
        createElement(RunStreamCoordinator, { canRecoverRun }),
      ),
    );
  });
  return { container, root };
}

function unmountCoordinator({ container, root }: { container: HTMLDivElement; root: Root }) {
  act(() => root.unmount());
  container.remove();
}

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}
