import { describe, expect, it } from "vitest";

import { describeRunProgress } from "./runProgress";

describe("run progress descriptions", () => {
  it.each([
    ["queued", "Queued"],
    ["memory_retrieval", "Checking memory"],
    ["compaction", "Compacting context"],
    ["generation", "Generating response"],
    ["response-generation", "Generating response"],
    ["memory_persistence", "Saving memory"],
    ["stopping", "Stopping"],
    ["failed", "Failed"],
    ["completed", "Completed"],
  ])("maps %s to %s", (stage, label) => {
    expect(describeRunProgress({ stage }).label).toBe(label);
  });

  it.each(["synthesis", "model_startup", "model_loading", "first_token_wait"])(
    "treats %s as model startup until output arrives",
    (stage) => {
      const waiting = describeRunProgress({ stage, model: "qwen3.8:27b" });
      expect(waiting.label).toBe("Starting qwen3.8:27b");
      expect(waiting.detail).toContain("waiting for its first output");

      expect(describeRunProgress({ stage, model: "qwen3.8:27b", hasThinking: true }).label).toBe(
        "Generating response",
      );
      expect(describeRunProgress({ stage, model: "qwen3.8:27b", hasAnswer: true }).label).toBe(
        "Generating response",
      );
    },
  );

  it("keeps compact-run copy specific", () => {
    expect(describeRunProgress({ stage: "queued", mode: "compact" }).label).toBe("Compaction queued");
    expect(describeRunProgress({ stage: "compaction", mode: "compact" }).label).toBe(
      "Compacting older context",
    );
    expect(describeRunProgress({ stage: "stopping", mode: "compact" }).label).toBe(
      "Stopping compaction",
    );
  });

  it("shows an unknown backend stage instead of calling it deciding", () => {
    const progress = describeRunProgress({ stage: "provider_health_check" });

    expect(progress.label).toBe("Provider Health Check");
    expect(progress.detail).toContain("reported the current stage");
  });
});
