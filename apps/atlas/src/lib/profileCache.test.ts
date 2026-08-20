import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";

import { removeAllProfileCaches, removeLockedProfileCaches } from "./profileCache";

describe("locked profile cache cleanup", () => {
  it("removes sensitive profile and run caches without touching another profile", () => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(["threads", "ollie"], [{ thread_id: "main", last_run_id: "run-1" }]);
    queryClient.setQueryData(["thread-history", "ollie", "main"], [{ role: "user", content: "secret" }]);
    queryClient.setQueryData(["thread-runs", "ollie", "main"], [{ run_id: "run-2", user_id: "ollie" }]);
    queryClient.setQueryData(["thread-context", "ollie", "main"], { tokens: 10 });
    queryClient.setQueryData(["memories", "ollie"], [{ memory: "secret" }]);
    queryClient.setQueryData(["chat-search", "ollie", "main", "secret"], {
      current_thread_results: [{ snippet: "secret" }],
    });
    queryClient.setQueryData(["run", "run-1"], { run_id: "run-1", user_id: "ollie" });
    queryClient.setQueryData(["run", "run-2"], { run_id: "run-2", user_id: "ollie" });
    queryClient.setQueryData(["threads", "other"], [{ thread_id: "main", last_run_id: "run-other" }]);
    queryClient.setQueryData(["run", "run-other"], { run_id: "run-other", user_id: "other" });

    removeLockedProfileCaches(queryClient, "ollie");

    expect(queryClient.getQueryData(["threads", "ollie"])).toBeUndefined();
    expect(queryClient.getQueryData(["thread-history", "ollie", "main"])).toBeUndefined();
    expect(queryClient.getQueryData(["thread-runs", "ollie", "main"])).toBeUndefined();
    expect(queryClient.getQueryData(["thread-context", "ollie", "main"])).toBeUndefined();
    expect(queryClient.getQueryData(["memories", "ollie"])).toBeUndefined();
    expect(queryClient.getQueryData(["chat-search", "ollie", "main", "secret"])).toBeUndefined();
    expect(queryClient.getQueryData(["run", "run-1"])).toBeUndefined();
    expect(queryClient.getQueryData(["run", "run-2"])).toBeUndefined();
    expect(queryClient.getQueryData(["threads", "other"])).toBeDefined();
    expect(queryClient.getQueryData(["run", "run-other"])).toBeDefined();
  });

  it("removes every profile-derived cache while preserving runtime bootstrap data", () => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(["users"], [{ user_id: "ollie" }]);
    queryClient.setQueryData(["threads", "ollie"], [{ thread_id: "main" }]);
    queryClient.setQueryData(["thread-history", "ollie", "main"], [{ content: "secret" }]);
    queryClient.setQueryData(["thread-runs", "ollie", "main"], [{ run_id: "run-1" }]);
    queryClient.setQueryData(["thread-context", "ollie", "main"], { tokens: 10 });
    queryClient.setQueryData(["memories", "ollie"], [{ memory: "secret" }]);
    queryClient.setQueryData(["chat-search", "ollie", "main", "secret"], {
      current_thread_results: [{ snippet: "secret" }],
    });
    queryClient.setQueryData(["run", "run-1"], { prompt: "secret" });
    queryClient.setQueryData(["status"], { status: "ok" });
    queryClient.setQueryData(["models"], { models: ["local-model"] });
    queryClient.setQueryData(["provider-settings"], { provider: "ollama" });

    removeAllProfileCaches(queryClient);

    for (const key of [
      ["users"],
      ["threads", "ollie"],
      ["thread-history", "ollie", "main"],
      ["thread-runs", "ollie", "main"],
      ["thread-context", "ollie", "main"],
      ["memories", "ollie"],
      ["chat-search", "ollie", "main", "secret"],
      ["run", "run-1"],
    ]) {
      expect(queryClient.getQueryData(key)).toBeUndefined();
    }
    expect(queryClient.getQueryData(["status"])).toEqual({ status: "ok" });
    expect(queryClient.getQueryData(["models"])).toEqual({ models: ["local-model"] });
    expect(queryClient.getQueryData(["provider-settings"])).toEqual({ provider: "ollama" });
  });
});
