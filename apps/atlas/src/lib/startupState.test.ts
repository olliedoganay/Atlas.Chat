import { describe, expect, it } from "vitest";

import { resolveStartupState } from "./startupState";

const baseOptions = {
  backendPhase: "online" as const,
  currentUserId: "atlas-user",
  currentUserLocked: false,
  modelCatalogLoaded: true,
  ollamaOnline: true,
  hasLocalModels: true,
  selectedModel: "gpt-oss:20b",
  selectedModelSupportsImages: false,
  threadHasHistory: false,
};

describe("resolveStartupState", () => {
  it("returns backend starting while the local runtime is booting", () => {
    const state = resolveStartupState({
      ...baseOptions,
      backendPhase: "starting",
    });

    expect(state.key).toBe("backend-starting");
    expect(state.tone).toBe("starting");
    expect(state.shellLabel).toBe("Starting backend");
    expect(state.canStartChat).toBe(false);
  });

  it("prioritizes profile selection before model readiness", () => {
    const state = resolveStartupState({
      ...baseOptions,
      currentUserId: "",
      modelCatalogLoaded: false,
      ollamaOnline: false,
    });

    expect(state.key).toBe("no-profile");
    expect(state.shellLabel).toBe("Choose a profile");
    expect(state.headerSummary).toContain("Choose a profile");
  });

  it("surfaces locked profiles as a first-class blocked state", () => {
    const state = resolveStartupState({
      ...baseOptions,
      currentUserLocked: true,
    });

    expect(state.key).toBe("profile-locked");
    expect(state.tone).toBe("warning");
    expect(state.composerPlaceholder).toContain("Unlock");
  });

  it("reports provider availability after model discovery starts", () => {
    const state = resolveStartupState({
      ...baseOptions,
      ollamaOnline: false,
      providerLabel: "LM Studio",
    });

    expect(state.key).toBe("ollama-offline");
    expect(state.idleTitle).toBe("LM Studio unavailable");
    expect(state.canStartChat).toBe(false);
  });

  it("reports missing local chat models separately from provider outage", () => {
    const state = resolveStartupState({
      ...baseOptions,
      hasLocalModels: false,
    });

    expect(state.key).toBe("no-local-models");
    expect(state.shellLabel).toBe("No local models");
  });

  it("requires an explicit model choice when no ready model is available", () => {
    const state = resolveStartupState({
      ...baseOptions,
      selectedModel: "",
    });

    expect(state.key).toBe("ready-no-model");
    expect(state.composerPlaceholder).toContain("Choose a local model");
    expect(state.canStartChat).toBe(false);
  });

  it("returns a ready state with image-aware copy for fresh threads", () => {
    const state = resolveStartupState({
      ...baseOptions,
      selectedModelSupportsImages: true,
    });

    expect(state.key).toBe("ready");
    expect(state.tone).toBe("online");
    expect(state.idleDescription).toContain("upload a photo");
    expect(state.canStartChat).toBe(true);
  });

  it("uses a locked-thread summary for chats that already have history", () => {
    const state = resolveStartupState({
      ...baseOptions,
      threadHasHistory: true,
    });

    expect(state.key).toBe("ready");
    expect(state.headerSummary).toBe("This chat is locked to its original model and temperature setting.");
  });
});
