import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { useAtlasStore } from "./useAtlasStore";

const initialState = useAtlasStore.getInitialState();

describe("useAtlasStore run state", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useAtlasStore.setState(initialState, true);
  });

  afterEach(() => {
    window.localStorage.clear();
    useAtlasStore.setState(initialState, true);
  });

  it("clears the active run mode when a run fails", () => {
    useAtlasStore.getState().beginRun("run-1", "compact", "", "user-1", "main", []);

    useAtlasStore.getState().failRun("Compaction failed.", "user-1", "main");

    const state = useAtlasStore.getState();
    expect(state.currentRunId).toBeNull();
    expect(state.currentRunMode).toBeNull();
    expect(state.isStreaming).toBe(false);
    expect(state.liveError).toBe("Compaction failed.");
  });

  it("ends a user-cancelled run without exposing a failure", () => {
    useAtlasStore.getState().beginRun("run-1", "chat", "hello", "user-1", "main", []);
    useAtlasStore.getState().appendToken("partial answer");

    useAtlasStore.getState().cancelRun();

    const state = useAtlasStore.getState();
    expect(state.currentRunId).toBeNull();
    expect(state.currentRunMode).toBeNull();
    expect(state.isStreaming).toBe(false);
    expect(state.liveError).toBe("");
    expect(state.currentStage).toBe("cancelled");
  });

  it("restores the prior stage after a failed stop request only while that run remains active", () => {
    useAtlasStore.getState().beginRun("run-1", "chat", "hello", "user-1", "main", []);
    useAtlasStore.getState().setStage("generation");
    useAtlasStore.getState().setStage("stopping");

    expect(useAtlasStore.getState().restoreRunStage("run-1", "generation")).toBe(true);
    expect(useAtlasStore.getState().currentStage).toBe("generation");
    expect(useAtlasStore.getState().currentRunId).toBe("run-1");
    expect(useAtlasStore.getState().isStreaming).toBe(true);
    expect(useAtlasStore.getState().liveError).toBe("");

    useAtlasStore.getState().completeRun();
    expect(useAtlasStore.getState().restoreRunStage("run-1", "generation")).toBe(false);
    expect(useAtlasStore.getState().currentStage).toBe("completed");
  });

  it("atomically recovers an authoritative run only for the persisted profile and thread", () => {
    useAtlasStore.setState({ currentUserId: "user-1", currentThreadId: "thread-1" });

    expect(useAtlasStore.getState().recoverRun({
      runId: "run-1",
      mode: "compact",
      userId: "user-1",
      threadId: "thread-1",
      prompt: "exact recovered prompt",
      stage: "compaction",
    })).toBe(true);

    expect(useAtlasStore.getState()).toMatchObject({
      currentRunId: "run-1",
      currentRunMode: "compact",
      activeRunUserId: "user-1",
      activeRunThreadId: "thread-1",
      pendingPrompt: "exact recovered prompt",
      currentStage: "compaction",
      isStreaming: true,
      liveError: "",
    });
  });

  it("refuses recovery for a different identity or over an existing live run", () => {
    useAtlasStore.setState({ currentUserId: "user-1", currentThreadId: "thread-1" });
    const wrongThread = {
      runId: "run-wrong",
      mode: "chat" as const,
      userId: "user-1",
      threadId: "thread-2",
      prompt: "wrong",
      stage: "running",
    };
    expect(useAtlasStore.getState().recoverRun(wrongThread)).toBe(false);
    expect(useAtlasStore.getState().currentRunId).toBeNull();

    useAtlasStore.getState().beginRun("run-live", "chat", "live", "user-1", "thread-1", []);
    expect(useAtlasStore.getState().recoverRun({ ...wrongThread, threadId: "thread-1" })).toBe(false);
    expect(useAtlasStore.getState().currentRunId).toBe("run-live");
  });

  it("toggles pinned chat keys per profile", () => {
    useAtlasStore.getState().togglePinnedThread("ollie", "main");
    expect(useAtlasStore.getState().pinnedThreadKeys).toEqual(["ollie::main"]);

    useAtlasStore.getState().togglePinnedThread("ollie", "main");
    expect(useAtlasStore.getState().pinnedThreadKeys).toEqual([]);
  });

  it("scrubs a locked current profile while preserving other profile and interface state", () => {
    useAtlasStore.setState({
      currentUserId: "ollie",
      currentThreadId: "secret-thread",
      currentThreadTitle: "Secret customer title",
      draftThreadModel: "private-model",
      draftThreadTemperature: 1.5,
      pinnedThreadKeys: ["ollie::secret-thread", "other::public-thread"],
      theme: "synthwave",
    });
    useAtlasStore.getState().beginRun(
      "run-1",
      "chat",
      "secret prompt",
      "ollie",
      "secret-thread",
      [],
    );
    useAtlasStore.getState().appendThinking("private reasoning");
    useAtlasStore.getState().appendToken("private answer");

    useAtlasStore.getState().clearProfileState("ollie");

    expect(useAtlasStore.getState()).toMatchObject({
      currentUserId: "",
      currentThreadId: "main",
      currentThreadTitle: "Main",
      draftThreadModel: "",
      draftThreadTemperature: null,
      currentRunId: null,
      activeRunUserId: null,
      activeRunThreadId: null,
      pendingPrompt: "",
      liveThinking: "",
      liveAnswer: "",
      liveError: "",
      pinnedThreadKeys: ["other::public-thread"],
      isStreaming: false,
      theme: "synthwave",
    });
  });

  it("clears erased profile and live-run state while preserving bootstrap and interface preferences", () => {
    const bootStartedAt = useAtlasStore.getState().backendStartupStartedAt;
    useAtlasStore.setState({
      currentUserId: "ollie",
      currentThreadId: "secret-thread",
      currentThreadTitle: "Secret title",
      draftThreadModel: "local-model",
      recentSearchQueries: ["secret query"],
      pinnedThreadKeys: ["ollie::secret-thread"],
      theme: "synthwave",
      crossChatMemoryEnabled: false,
    });
    useAtlasStore.getState().beginRun(
      "run-1",
      "chat",
      "secret prompt",
      "ollie",
      "secret-thread",
      [],
    );
    useAtlasStore.getState().appendThinking("private reasoning");
    useAtlasStore.getState().appendToken("private answer");

    useAtlasStore.getState().resetAfterDataWipe();

    expect(useAtlasStore.getState()).toMatchObject({
      currentUserId: "",
      currentThreadId: "main",
      currentThreadTitle: "Main",
      draftThreadModel: "",
      currentRunId: null,
      activeRunUserId: null,
      activeRunThreadId: null,
      pendingPrompt: "",
      liveThinking: "",
      liveAnswer: "",
      liveError: "",
      recentSearchQueries: [],
      pinnedThreadKeys: [],
      isStreaming: false,
      theme: "synthwave",
      crossChatMemoryEnabled: false,
      backendStartupStartedAt: bootStartedAt,
    });
  });
});
