import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";

import { useAtlasStore } from "../store/useAtlasStore";
import { AdvancedPage } from "./AdvancedPage";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("AdvancedPage profile privacy", () => {
  let root: Root | null = null;
  let container: HTMLDivElement | null = null;

  afterEach(() => {
    act(() => root?.unmount());
    container?.remove();
    document.body.innerHTML = "";
    useAtlasStore.setState({ currentUserId: "", currentThreadId: "main" });
    root = null;
    container = null;
  });

  it("does not render cached prompts or run details for a locked profile", async () => {
    useAtlasStore.setState({ currentUserId: "alice", currentThreadId: "private-thread" });
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    queryClient.setQueryData(["users"], [
      { user_id: "alice", protection: "password", locked: true },
    ]);
    queryClient.setQueryData(["status"], {
      backend: "Atlas Chat local runtime",
      security: {
        sqlite_encrypted_at_rest: true,
        vector_store_encrypted_at_rest: true,
        profile_key_protection: "Password protected",
      },
    });
    queryClient.setQueryData(["models"], {
      provider_label: "Ollama",
      provider_online: true,
      has_chat_models: true,
      catalog_source: "ollama",
      models: ["qwen3.8:27b"],
    });
    queryClient.setQueryData(["threads", "alice"], [
      {
        user_id: "alice",
        thread_id: "private-thread",
        title: "Private plans",
        last_prompt: "CONFIDENTIAL CACHED PROMPT",
        last_run_id: "run-private",
      },
    ]);
    queryClient.setQueryData(["run", "run-private"], {
      run_id: "run-private",
      thread_id: "private-thread",
      thread_title: "Private plans",
      prompt: "CONFIDENTIAL CACHED RUN",
      status: "completed",
      started_at: "2026-08-20T00:00:00Z",
      answer: "secret",
      events: [],
      mode: "chat",
      user_id: "alice",
    });

    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    act(() => {
      root?.render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter>
            <AdvancedPage />
          </MemoryRouter>
        </QueryClientProvider>,
      );
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain("Profile locked");
    expect(document.body.textContent).not.toContain("CONFIDENTIAL CACHED PROMPT");
    expect(document.body.textContent).not.toContain("CONFIDENTIAL CACHED RUN");
    expect(document.body.textContent).not.toContain("Private plans");
  });
});
