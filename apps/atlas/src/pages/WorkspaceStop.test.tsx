import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const apiMocks = vi.hoisted(() => ({
  cancelRun: vi.fn(),
  getModels: vi.fn(),
  getRun: vi.fn(),
  getStatus: vi.fn(),
  getThreadContextUsage: vi.fn(),
  getThreadHistory: vi.fn(),
  getThreadRuns: vi.fn(),
  getThreads: vi.fn(),
  getUsers: vi.fn(),
}));

vi.mock("../lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("../lib/api")>();
  return {
    ...original,
    ...apiMocks,
  };
});

import { useAtlasStore } from "../store/useAtlasStore";
import { WorkspacePage } from "./WorkspacePage";

const initialState = useAtlasStore.getInitialState();

describe("Workspace Stop failure recovery", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useAtlasStore.setState(initialState, true);
    apiMocks.getStatus.mockResolvedValue({
      busy: true,
      default_chat_temperature: null,
      chat_temperature: null,
      chat_provider: "ollama",
      chat_provider_label: "Ollama",
      ollama_url: "http://127.0.0.1:11434",
    });
    apiMocks.getModels.mockResolvedValue({
      default_temperature: null,
      ollama_online: true,
      has_local_models: true,
      provider: "ollama",
      provider_label: "Ollama",
      provider_online: true,
      has_chat_models: true,
      supports_context_window: true,
      supports_model_unload: true,
      catalog_source: "test",
      temperature_presets: [],
      context_window_presets: [],
      loaded_models: ["qwen:test"],
      ollama_context_window: {},
      models: ["qwen:test"],
      model_details: [{ name: "qwen:test", supports_reasoning: true }],
    });
    apiMocks.getUsers.mockResolvedValue([{ user_id: "user-1", locked: false }]);
    apiMocks.getThreads.mockResolvedValue([{
      user_id: "user-1",
      thread_id: "main",
      title: "Main",
      chat_model: "qwen:test",
      last_run_id: "run-1",
    }]);
    apiMocks.getThreadHistory.mockResolvedValue([]);
    apiMocks.getThreadContextUsage.mockResolvedValue({
      thread_id: "main",
      user_id: "user-1",
      chat_model: "qwen:test",
      context_window: 4096,
      auto_compact_ratio: 0.8,
      auto_compact_threshold: 3277,
      auto_compact_margin_tokens: 512,
      representation_tokens: 10,
      summary_tokens: 0,
      raw_message_tokens: 10,
      compacted_message_count: 0,
      recent_raw_message_count: 0,
      message_count: 0,
    });
    apiMocks.getThreadRuns.mockResolvedValue([]);
    apiMocks.getRun.mockResolvedValue({
      run_id: "run-1",
      mode: "chat",
      user_id: "user-1",
      thread_id: "main",
      prompt: "hello",
      status: "running",
      started_at: "2026-08-20T00:00:00Z",
      answer: "",
      events: [],
      chat_model: "qwen:test",
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    useAtlasStore.setState(initialState, true);
    document.body.innerHTML = "";
  });

  it("keeps the run attached, restores its prior stage, and reports a nonterminal action error", async () => {
    let rejectStop: ((error: Error) => void) | null = null;
    apiMocks.cancelRun.mockReturnValue(new Promise((_resolve, reject) => {
      rejectStop = reject;
    }));
    useAtlasStore.setState({
      currentUserId: "user-1",
      currentThreadId: "main",
      currentThreadTitle: "Main",
      draftThreadModel: "qwen:test",
    });
    useAtlasStore.getState().beginRun("run-1", "chat", "hello", "user-1", "main", []);
    useAtlasStore.getState().setStage("generation");
    const rendered = renderWorkspace();
    const stopButton = await waitForStopButton(rendered.container);

    act(() => stopButton.click());
    expect(useAtlasStore.getState()).toMatchObject({
      currentRunId: "run-1",
      currentStage: "stopping",
      isStreaming: true,
      liveError: "",
    });

    await act(async () => {
      rejectStop?.(new Error("Cancellation endpoint unavailable."));
      await Promise.resolve();
      await Promise.resolve();
    });
    await act(async () => {
      await vi.waitFor(() => {
        expect(rendered.container.textContent).toContain("Cancellation endpoint unavailable.");
      });
    });

    expect(useAtlasStore.getState()).toMatchObject({
      currentRunId: "run-1",
      currentStage: "generation",
      isStreaming: true,
      liveError: "",
    });
    unmountWorkspace(rendered);
  });
});

function renderWorkspace() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const queryClient = new QueryClient({
    defaultOptions: {
      mutations: { retry: false },
      queries: { retry: false },
    },
  });
  act(() => {
    root.render(
      createElement(
        QueryClientProvider,
        { client: queryClient },
        createElement(MemoryRouter, null, createElement(WorkspacePage)),
      ),
    );
  });
  return { container, root };
}

async function waitForStopButton(container: ParentNode) {
  let button: HTMLButtonElement | undefined;
  await act(async () => {
    await vi.waitFor(() => {
      button = Array.from(container.querySelectorAll("button")).find((item) => item.textContent?.trim() === "Stop");
      expect(button).toBeDefined();
    });
  });
  return button as HTMLButtonElement;
}

function unmountWorkspace({ container, root }: { container: HTMLDivElement; root: Root }) {
  act(() => root.unmount());
  container.remove();
}
