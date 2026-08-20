import type { RunMode, RunSummary } from "./api";

const RECOVERABLE_RUN_STATUSES = new Set(["queued", "running", "cancelling"]);
const TERMINAL_RUN_STATUSES = new Set(["completed", "failed", "cancelled", "canceled"]);
const TERMINAL_RUN_EVENTS = new Set(["run_completed", "run_failed", "run_cancelled", "run_canceled"]);

export type RecoverableRun = {
  runId: string;
  mode: RunMode;
  userId: string;
  threadId: string;
  prompt: string;
  stage: string;
};

export function selectRecoverableRun(
  runs: RunSummary[],
  expectedUserId: string,
  expectedThreadId: string,
): RecoverableRun | null {
  if (!expectedUserId || !expectedThreadId) {
    return null;
  }
  const candidates = runs
    .filter((run) => isRecoverableRun(run, expectedUserId, expectedThreadId))
    .sort(compareRecoveryPriority);
  const run = candidates[0];
  if (!run) {
    return null;
  }
  return {
    runId: run.run_id,
    mode: run.mode as RunMode,
    userId: run.user_id,
    threadId: run.thread_id,
    prompt: run.prompt,
    stage: recoveryStage(run),
  };
}

function isRecoverableRun(run: RunSummary, expectedUserId: string, expectedThreadId: string) {
  const status = normalize(run.status);
  if (
    !run.run_id ||
    run.user_id !== expectedUserId ||
    run.thread_id !== expectedThreadId ||
    (run.mode !== "chat" && run.mode !== "compact") ||
    !RECOVERABLE_RUN_STATUSES.has(status) ||
    TERMINAL_RUN_STATUSES.has(status) ||
    Boolean(String(run.completed_at ?? "").trim())
  ) {
    return false;
  }
  return !(run.events ?? []).some((event) => TERMINAL_RUN_EVENTS.has(normalize(event.type)));
}

function compareRecoveryPriority(left: RunSummary, right: RunSummary) {
  const leftQueued = normalize(left.status) === "queued" ? 1 : 0;
  const rightQueued = normalize(right.status) === "queued" ? 1 : 0;
  if (leftQueued !== rightQueued) {
    return leftQueued - rightQueued;
  }
  return `${left.started_at}\u0000${left.run_id}`.localeCompare(`${right.started_at}\u0000${right.run_id}`);
}

function recoveryStage(run: RunSummary) {
  const status = normalize(run.status);
  if (status === "queued") {
    return "queued";
  }
  if (status === "cancelling") {
    return "stopping";
  }
  for (let index = (run.events ?? []).length - 1; index >= 0; index -= 1) {
    const event = run.events[index];
    if (normalize(event.type) !== "stage_changed") {
      continue;
    }
    const stage = String(event.payload?.stage ?? "").trim();
    if (stage) {
      return stage;
    }
  }
  return "running";
}

function normalize(value: unknown) {
  return String(value ?? "").trim().toLowerCase();
}
