export type RunProgressTone = "active" | "stopping" | "failed" | "complete";

export type RunProgressDescription = {
  label: string;
  detail: string;
  tone: RunProgressTone;
};

type RunProgressInput = {
  stage?: string | null;
  mode?: string | null;
  model?: string | null;
  hasThinking?: boolean;
  hasAnswer?: boolean;
};

const MEMORY_RETRIEVAL_STAGES = new Set(["memory_retrieval", "retrieving_memory", "memory_search"]);
const MODEL_STARTUP_STAGES = new Set([
  "synthesis",
  "model_startup",
  "model_loading",
  "loading_model",
  "waiting_for_model",
  "first_token_wait",
  "waiting_for_first_token",
]);
const GENERATION_STAGES = new Set(["generation", "generating", "response_generation", "streaming"]);
const MEMORY_PERSISTENCE_STAGES = new Set([
  "memory_persistence",
  "persisting_memory",
  "saving_memory",
  "persistence",
  "finalize",
  "finalizing",
]);

export function describeRunProgress({
  stage,
  mode,
  model,
  hasThinking = false,
  hasAnswer = false,
}: RunProgressInput): RunProgressDescription {
  const normalizedStage = normalizeStage(stage);
  const hasModelOutput = hasThinking || hasAnswer;

  if (mode === "compact") {
    if (isStoppingStage(normalizedStage)) {
      return {
        label: "Stopping compaction",
        detail: "Atlas is asking the local model to stop compacting this thread.",
        tone: "stopping",
      };
    }
    if (normalizedStage === "queued") {
      return {
        label: "Compaction queued",
        detail: "This compaction will start when the local run worker is available.",
        tone: "active",
      };
    }
    if (normalizedStage === "failed") {
      return failedProgress();
    }
    if (normalizedStage === "completed") {
      return completedProgress();
    }
    return {
      label: "Compacting older context",
      detail: "Atlas is folding older messages into a smaller working summary.",
      tone: "active",
    };
  }

  if (normalizedStage === "failed") {
    return failedProgress();
  }
  if (normalizedStage === "completed") {
    return completedProgress();
  }
  if (isStoppingStage(normalizedStage)) {
    return {
      label: "Stopping",
      detail: "Atlas is asking the local model to stop. This can take a moment.",
      tone: "stopping",
    };
  }
  if (normalizedStage === "queued") {
    return {
      label: "Queued",
      detail: "This response will start when the local run worker is available.",
      tone: "active",
    };
  }
  if (MEMORY_RETRIEVAL_STAGES.has(normalizedStage)) {
    return {
      label: "Checking memory",
      detail: "Atlas is searching saved context before starting the chat model.",
      tone: "active",
    };
  }
  if (normalizedStage === "compaction" || normalizedStage === "compacting") {
    return {
      label: "Compacting context",
      detail: "Atlas is reducing older context before continuing this response.",
      tone: "active",
    };
  }
  if (GENERATION_STAGES.has(normalizedStage) || (MODEL_STARTUP_STAGES.has(normalizedStage) && hasModelOutput)) {
    return {
      label: "Generating response",
      detail: "The model is producing this response.",
      tone: "active",
    };
  }
  if (MODEL_STARTUP_STAGES.has(normalizedStage)) {
    const modelLabel = String(model ?? "").trim();
    return {
      label: modelLabel ? `Starting ${modelLabel}` : "Starting model",
      detail: modelLabel
        ? `${modelLabel} is starting and Atlas is waiting for its first output. A cold start can take a minute or more.`
        : "The chat model is starting and Atlas is waiting for its first output. A cold start can take a minute or more.",
      tone: "active",
    };
  }
  if (MEMORY_PERSISTENCE_STAGES.has(normalizedStage)) {
    return {
      label: "Saving memory",
      detail: "The response is complete; Atlas is saving durable memory and conversation state.",
      tone: "active",
    };
  }
  if (!normalizedStage || normalizedStage === "idle" || normalizedStage === "starting") {
    return {
      label: "Starting run",
      detail: "Atlas is preparing this request for the local run worker.",
      tone: "active",
    };
  }

  const reportedStage = formatStage(normalizedStage);
  return {
    label: reportedStage,
    detail: `Atlas reported the current stage as ${reportedStage.toLocaleLowerCase()}.`,
    tone: "active",
  };
}

function normalizeStage(stage?: string | null) {
  return String(stage ?? "")
    .trim()
    .toLocaleLowerCase()
    .replace(/[\s-]+/g, "_");
}

function isStoppingStage(stage: string) {
  return stage === "stopping" || stage === "cancelling" || stage === "canceling";
}

function formatStage(stage: string) {
  return stage
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toLocaleUpperCase() + part.slice(1))
    .join(" ");
}

function failedProgress(): RunProgressDescription {
  return {
    label: "Failed",
    detail: "The run stopped because Atlas or the local model reported an error.",
    tone: "failed",
  };
}

function completedProgress(): RunProgressDescription {
  return {
    label: "Completed",
    detail: "The response and its conversation state were saved.",
    tone: "complete",
  };
}
