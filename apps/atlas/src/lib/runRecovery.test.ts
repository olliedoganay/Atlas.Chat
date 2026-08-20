import { describe, expect, it } from "vitest";

import type { RunSummary } from "./api";
import { selectRecoverableRun } from "./runRecovery";

describe("active run recovery selection", () => {
  it("selects the exact active profile/thread run and preserves its run contract", () => {
    const candidate = selectRecoverableRun(
      [
        run({ run_id: "queued", status: "queued", started_at: "2026-08-20T10:01:00Z" }),
        run({
          run_id: "running",
          mode: "compact",
          prompt: "exact prompt",
          status: "running",
          started_at: "2026-08-20T10:00:00Z",
          events: [event("stage_changed", { stage: "compaction" })],
        }),
        run({ run_id: "other-thread", thread_id: "other", status: "running" }),
        run({ run_id: "other-user", user_id: "other", status: "running" }),
      ],
      "user-1",
      "main",
    );

    expect(candidate).toEqual({
      runId: "running",
      mode: "compact",
      userId: "user-1",
      threadId: "main",
      prompt: "exact prompt",
      stage: "compaction",
    });
  });

  it("never resurrects terminal or internally terminal artifacts", () => {
    expect(
      selectRecoverableRun(
        [
          run({ run_id: "completed", status: "completed" }),
          run({ run_id: "failed", status: "failed" }),
          run({ run_id: "completed-at", status: "running", completed_at: "2026-08-20T10:02:00Z" }),
          run({
            run_id: "terminal-event",
            status: "running",
            events: [event("run_completed")],
          }),
          run({ run_id: "invalid-mode", mode: "tool", status: "running" }),
        ],
        "user-1",
        "main",
      ),
    ).toBeNull();
  });

  it.each([
    ["queued", "queued"],
    ["cancelling", "stopping"],
  ])("maps a %s artifact to the recoverable %s UI stage", (status, stage) => {
    expect(
      selectRecoverableRun([run({ status })], "user-1", "main"),
    ).toMatchObject({ runId: "run-1", stage });
  });
});

function run(overrides: Partial<RunSummary>): RunSummary {
  return {
    run_id: "run-1",
    mode: "chat",
    user_id: "user-1",
    thread_id: "main",
    prompt: "hello",
    status: "running",
    started_at: "2026-08-20T10:00:00Z",
    answer: "",
    events: [],
    ...overrides,
  };
}

function event(type: string, payload: Record<string, unknown> = {}) {
  return { type, timestamp: "2026-08-20T10:00:00Z", payload };
}
