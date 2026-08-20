import { invoke, isTauri } from "@tauri-apps/api/core";

type BackendRuntime = {
  host: string;
  port: number;
  token: string;
  baseUrl?: string;
};

const API_READ_REQUEST_TIMEOUT_MS = 15_000;
const API_MUTATION_REQUEST_TIMEOUT_MS = 60_000;

let runtimePromise: Promise<BackendRuntime> | null = null;
let lastRuntime: BackendRuntime | null = null;

export type ThemeMode =
  | "light"
  | "dark"
  | "crt-green"
  | "crt-amber"
  | "synthwave"
  | "nasa";
export type RunMode = "chat" | "compact";
export type ReasoningMode = "off" | "on" | "low" | "medium" | "high";
export type AttachmentKind = "image" | "file";

export type ThreadSummary = {
  user_id: string;
  thread_id: string;
  title?: string;
  chat_model?: string;
  temperature?: number | null;
  last_mode?: string;
  updated_at?: string;
  last_prompt?: string;
  last_run_id?: string;
};

export type ChatAttachment = {
  name: string;
  media_type: string;
  kind?: AttachmentKind;
  data_url?: string;
  text_content?: string;
  byte_size?: number;
};
export type ImageAttachment = ChatAttachment;

export type UserSummary = {
  user_id: string;
  updated_at?: string;
  protection?: "passwordless" | "password";
  locked?: boolean;
};

export type StoredMemory = {
  memory: string;
  memory_id: string;
  score?: number | null;
  metadata?: Record<string, unknown> | null;
};

export type BackendStatus = {
  status: string;
  product_name: string;
  version?: string;
  backend: string;
  default_chat_temperature: number | null;
  chat_temperature: number | null;
  embed_model: string;
  ollama_url: string;
  chat_provider?: string;
  chat_provider_label?: string;
  chat_base_url?: string;
  runtime_mode: string;
  busy: boolean;
  security: {
    profile_key_protection: string;
    run_artifacts_encrypted_at_rest: boolean;
    run_index_encrypted_at_rest: boolean;
    packaged_logs_default: string;
    sqlite_encrypted_at_rest: boolean;
    sqlite_paths: string[];
    vector_store: string;
    vector_store_encrypted_at_rest: boolean;
    vector_store_path: string;
  };
};

export type AppDiagnostics = {
  platform: string;
  data_dir: string;
  log_dir: string;
  backend_log_path: string;
  packaged_logs_enabled: boolean;
};

export type TemperaturePreset = {
  label: string;
  value: number;
};

export type ModelCatalog = {
  default_temperature: number | null;
  ollama_online: boolean;
  has_local_models: boolean;
  provider?: string;
  provider_label?: string;
  provider_base_url?: string;
  provider_online?: boolean;
  has_chat_models?: boolean;
  supports_context_window?: boolean;
  supports_model_unload?: boolean;
  catalog_source: string;
  temperature_presets: TemperaturePreset[];
  context_window_presets: number[];
  loaded_models: string[];
  ollama_context_window: {
    configured_context_window?: number | null;
    effective_context_window?: number | null;
    source?: "configured" | "ollama" | "provider";
  };
  models: string[];
  model_details: Array<{
    name: string;
    size_bytes?: number | null;
    family?: string;
    families?: string[];
    capabilities?: string[];
    supports_images?: boolean;
    supports_reasoning?: boolean;
    reasoning_mode_strategy?: "none" | "boolean" | "levels";
  }>;
};

export type ProviderSettings = {
  provider: string;
  provider_label: string;
  base_url: string;
  has_api_key: boolean;
  api_key_unavailable?: boolean;
  base_url_invalid?: boolean;
  secure_key_storage_available: boolean;
  restart_required?: boolean;
  providers: Array<{
    id: string;
    label: string;
    default_base_url: string;
  }>;
};

export type ModelPull = {
  pull_id: string;
  model: string;
  status: "queued" | "pulling" | "completed" | "failed" | "cancelled";
  detail: string;
  completed: number;
  total: number;
  progress: number | null;
  started_at: string;
  updated_at: string;
  error?: string | null;
};

export type DiscoveryReport = {
  system: {
    os: string;
    platform: string;
    cpu: {
      model: string | null;
      logical_cores: number | null;
    };
    memory: {
      total_gb: number | null;
    };
    gpus: Array<{
      name: string;
      memory_gb: number | null;
      kind?: "dedicated" | "integrated" | "unknown";
      memory_source?: "nvidia-smi" | "adapterram" | "shared" | "unknown";
    }>;
    detection: {
      confidence: "minimal" | "partial" | "good" | "full";
      notes: string[];
    };
  };
  atlas: {
    status: "ready" | "memory-degraded" | "chat-blocked" | "runtime-unavailable";
    summary: string;
    notes: string[];
    ollama_url: string;
    ollama_online: boolean;
    provider?: string;
    provider_label?: string;
    provider_base_url?: string;
    provider_online?: boolean;
    has_local_chat_models: boolean;
    configured_embed_model: string;
    configured_embed_model_installed: boolean;
  };
  installed_models: Array<{
    name: string;
    atlas_role: "chat" | "embedding" | "vision" | "other";
    configured_embed_model: boolean;
    supports_images: boolean;
    supports_reasoning: boolean;
  }>;
  recommended_models: Array<{
    name: string;
    title: string;
    use_case: "chat" | "coding" | "vision" | "reasoning" | "embedding";
    atlas_role: "chat" | "embedding";
    installed: boolean;
    supports_images: boolean;
    supports_reasoning: boolean;
    model_size_gb?: number | null;
    fit: "good" | "tight" | "cpu-only" | "unavailable" | "too-large";
    runtime: "GPU" | "Hybrid" | "CPU" | "Unknown";
    reason: string;
    pull_command: string;
  }>;
};

export type RunStatusEvent = {
  type: string;
  timestamp: string;
  payload: Record<string, unknown>;
  sequence?: number;
  sequence_end?: number;
};

export type RunTraceItem = {
  timestamp: string;
  stage?: string;
  rationale?: string;
  inputs?: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  artifacts?: Record<string, unknown>;
};

export type RunSummary = {
  run_id: string;
  mode: string;
  user_id: string;
  thread_id: string;
  thread_title?: string;
  chat_model?: string;
  temperature?: number | null;
  prompt: string;
  status: string;
  started_at: string;
  completed_at?: string | null;
  answer: string;
  events: RunStatusEvent[];
  trace_items?: RunTraceItem[];
  error?: string | null;
  thread_summary?: string;
  compacted_message_count?: number;
  detected_context_window?: number;
  diagnostics?: {
    first_token_latency_ms?: number | null;
    total_duration_ms?: number | null;
    generation_duration_ms?: number | null;
    output_tokens_estimate?: number | null;
    output_tokens_per_second_estimate?: number | null;
    compaction_gain_tokens_estimate?: number | null;
    compaction_events_count?: number | null;
  };
};

export type ThreadMessage = {
  role: "user" | "assistant" | "system";
  content: string;
  attachments?: ImageAttachment[];
  history_index?: number | null;
  kind?: string;
  run_id?: string;
  timestamp?: string;
  thread_summary?: string;
  compacted_message_count?: number;
  newly_compacted_message_count?: number;
  detected_context_window?: number;
  history_representation_tokens_before_compaction?: number;
  history_representation_tokens_after_compaction?: number;
  compaction_reason?: string;
};

export type ChatSearchResult = {
  thread_id: string;
  thread_title: string;
  chat_model?: string;
  updated_at?: string;
  match_type: "thread" | "message";
  role?: "user" | "assistant" | null;
  history_index?: number | null;
  snippet: string;
};

export type ChatSearchResponse = {
  query: string;
  current_thread_id: string;
  current_thread_results: ChatSearchResult[];
  other_thread_results: ChatSearchResult[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const primaryRuntime = await getBackendRuntime();

  try {
    return await requestWithRuntime<T>(primaryRuntime, path, init);
  } catch (error) {
    invalidateBackendRuntime();
    if (!isReadOnlyRequest(init)) {
      throw error;
    }
    const refreshedRuntime = await getBackendRuntime();
    const changedRuntime =
      refreshedRuntime.host !== primaryRuntime.host ||
      refreshedRuntime.port !== primaryRuntime.port ||
      refreshedRuntime.token !== primaryRuntime.token;

    if (!changedRuntime) {
      throw error;
    }

    return requestWithRuntime<T>(refreshedRuntime, path, init);
  }
}

export function getStatus() {
  return request<BackendStatus>("/status");
}

export function getModels() {
  return request<ModelCatalog>("/models");
}

export function getProviderSettings() {
  return request<ProviderSettings>("/settings/provider");
}

export function saveProviderSettings(settings: {
  provider: string;
  base_url?: string;
  api_key?: string;
  preserve_existing_key?: boolean;
}) {
  return request<ProviderSettings>("/settings/provider", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}

export function clearProviderApiKey() {
  return request<ProviderSettings>("/settings/provider/api-key", {
    method: "DELETE",
  });
}

export function startModelPull(model: string) {
  return request<ModelPull>("/models/pulls", {
    method: "POST",
    body: JSON.stringify({ model }),
  });
}

export function listModelPulls() {
  return request<ModelPull[]>("/models/pulls");
}

export function getModelPull(pullId: string) {
  return request<ModelPull>(`/models/pulls/${encodeURIComponent(pullId)}`);
}

export function cancelModelPull(pullId: string) {
  return request<ModelPull>(`/models/pulls/${encodeURIComponent(pullId)}`, {
    method: "DELETE",
  });
}

export function setOllamaContextWindow(contextWindow: number | null) {
  return request<{
    configured_context_window?: number | null;
    ollama_context_window: ModelCatalog["ollama_context_window"];
  }>("/models/context-window", {
    method: "PATCH",
    body: JSON.stringify({ context_window: contextWindow }),
  });
}

export function unloadOllamaModel(model: string) {
  return request<{
    status: string;
    model: string;
    loaded_models?: string[];
  }>("/models/unload", {
    method: "POST",
    body: JSON.stringify({ model }),
  });
}

export function getDiscovery() {
  return request<DiscoveryReport>("/discovery");
}

export function getHealth() {
  return request<{ status: string; product: string }>("/health");
}

export function getThreads(userId?: string) {
  const query = userId ? `?user_id=${encodeURIComponent(userId)}` : "";
  return request<ThreadSummary[]>(`/threads${query}`);
}

export function getUsers() {
  return request<UserSummary[]>("/users");
}

export function createUser(userId: string, password?: string) {
  return request<UserSummary>("/users", {
    method: "POST",
    body: JSON.stringify({ user_id: userId, password: password || null }),
  });
}

export function unlockUser(userId: string, password?: string) {
  return request<UserSummary>(`/users/${encodeURIComponent(userId)}/unlock`, {
    method: "POST",
    body: JSON.stringify({ password: password || null }),
  });
}

export function lockUser(userId: string) {
  return request<UserSummary>(`/users/${encodeURIComponent(userId)}/lock`, {
    method: "POST",
  });
}

export function deleteUser(userId: string) {
  return request<{ status: string; user_id: string }>(
    `/users/${encodeURIComponent(userId)}?confirmation_user_id=${encodeURIComponent(userId)}`,
    {
      method: "DELETE",
    },
  );
}

export function getMemories(userId: string, limit = 50) {
  return request<StoredMemory[]>(`/memories?user_id=${encodeURIComponent(userId)}&limit=${limit}`);
}

export function createMemory(userId: string, text: string) {
  return request<{ status: string; user_id: string; memory_id: string; text: string }>("/memories", {
    method: "POST",
    body: JSON.stringify({ user_id: userId, text }),
  });
}

export function deleteMemory(userId: string, memoryId: string) {
  return request<{ status: string; user_id: string; memory_id: string }>(
    `/memories/${encodeURIComponent(memoryId)}?user_id=${encodeURIComponent(userId)}`,
    {
      method: "DELETE",
    },
  );
}

export function renameThread(threadId: string, userId: string, title: string) {
  return request<ThreadSummary>(`/threads/${encodeURIComponent(threadId)}/title`, {
    method: "PATCH",
    body: JSON.stringify({ user_id: userId, title }),
  });
}

export function duplicateThread(threadId: string, userId: string) {
  return request<ThreadSummary>(`/threads/${encodeURIComponent(threadId)}/duplicate`, {
    method: "POST",
    body: JSON.stringify({ user_id: userId }),
  });
}

export function branchThread(threadId: string, userId: string, afterMessageCount: number) {
  return request<ThreadSummary>(`/threads/${encodeURIComponent(threadId)}/branch`, {
    method: "POST",
    body: JSON.stringify({ user_id: userId, after_message_count: afterMessageCount }),
  });
}

export function getThreadHistory(threadId: string, userId?: string) {
  const query = userId ? `?user_id=${encodeURIComponent(userId)}` : "";
  return request<ThreadMessage[]>(`/threads/${encodeURIComponent(threadId)}/history${query}`);
}

export type ThreadContextUsage = {
  thread_id: string;
  user_id: string;
  chat_model: string;
  context_window: number;
  auto_compact_ratio: number;
  auto_compact_threshold: number;
  auto_compact_margin_tokens: number;
  representation_tokens: number;
  summary_tokens: number;
  raw_message_tokens: number;
  compacted_message_count: number;
  recent_raw_message_count: number;
  message_count: number;
};

export function getThreadContextUsage(threadId: string, userId?: string) {
  const query = userId ? `?user_id=${encodeURIComponent(userId)}` : "";
  return request<ThreadContextUsage>(`/threads/${encodeURIComponent(threadId)}/context${query}`);
}

export function getThreadRuns(threadId: string, userId?: string) {
  const query = userId ? `?user_id=${encodeURIComponent(userId)}` : "";
  return request<RunSummary[]>(`/threads/${encodeURIComponent(threadId)}/runs${query}`);
}

export function searchChats(query: string, userId: string, currentThreadId?: string, limit = 8) {
  const params = new URLSearchParams({
    user_id: userId,
    q: query,
    limit: String(limit),
  });
  if (currentThreadId) {
    params.set("current_thread_id", currentThreadId);
  }
  return request<ChatSearchResponse>(`/search?${params.toString()}`);
}

export function getRun(runId: string) {
  return request<RunSummary>(`/runs/${encodeURIComponent(runId)}`);
}

export function cancelRun(runId: string) {
  return request<{ status: string; run_id: string; detail?: string }>(`/runs/${encodeURIComponent(runId)}/cancel`, {
    method: "POST",
  });
}

export function startChat(
  prompt: string,
  userId: string,
  threadId: string,
  chatModel?: string,
  temperature?: number | null,
  reasoningMode?: ReasoningMode,
  threadTitle?: string,
  crossChatMemory = true,
  autoCompactLongChats = true,
  attachments: ImageAttachment[] = [],
) {
  return request<{ run_id: string; status: string; mode: RunMode; chat_model: string; temperature: number | null }>("/chat", {
    method: "POST",
    body: JSON.stringify({
      prompt,
      user_id: userId,
      thread_id: threadId,
      chat_model: chatModel,
      temperature,
      reasoning_mode: reasoningMode,
      thread_title: threadTitle,
      cross_chat_memory: crossChatMemory,
      auto_compact_long_chats: autoCompactLongChats,
      attachments,
    }),
  });
}

export function startCompact(threadId: string, userId: string) {
  return request<{ run_id: string; status: string; mode: RunMode; chat_model: string; temperature: number | null }>(
    `/threads/${encodeURIComponent(threadId)}/compact`,
    {
      method: "POST",
      body: JSON.stringify({ user_id: userId }),
    },
  );
}

export function resetThread(threadId: string, userId?: string) {
  return request<Record<string, unknown>>("/admin/reset/thread", {
    method: "POST",
    body: JSON.stringify({ thread_id: threadId, user_id: userId }),
  });
}

export function resetAll() {
  return request<Record<string, unknown>>("/admin/reset/all", {
    method: "POST",
    body: JSON.stringify({ confirmation: "RESET ATLAS" }),
  });
}

export type RunnerStatus = {
  available: boolean;
  busy: boolean;
  active_runs: number;
  runtime_preparing: boolean;
  reason?: string;
  server_version?: string;
  supported_languages: string[];
  server_languages: string[];
  client_languages: string[];
};

export type RunnerStartResponse = {
  run_id: string;
  language: string;
  container: string;
  network?: string;
  timeout_seconds?: number;
  vnc_url?: string;
  web_url?: string;
};

export type RunnerRuntimeStatus = {
  name: "python-gui";
  version: string;
  image: string;
  state: "missing" | "preparing" | "ready" | "failed" | "unavailable";
  message: string;
  error?: string | null;
  progress: number;
  started_at?: number | null;
  completed_at?: number | null;
  image_size_bytes?: number | null;
  bundled_packages: string[];
  execution_network: "internal-preview-only";
  submitted_code_used_during_preparation: false;
  log_tail: string[];
};

export type RunnerRuntimePreparation = {
  required: boolean;
  started: boolean;
  runtime: RunnerRuntimeStatus | null;
};

export type RunnerEvent =
  | { type: "output"; stream: "stdout" | "stderr"; chunk: string }
  | { type: "exit"; code: number; duration_ms: number };

export function getRunnerStatus() {
  return request<RunnerStatus>("/runner/status");
}

export function getPythonGuiRuntimeStatus() {
  return request<RunnerRuntimeStatus>("/runner/runtime/python-gui");
}

export function prepareRunnerRuntime(language: string, code: string) {
  return request<RunnerRuntimePreparation>("/runner/runtime/prepare", {
    method: "POST",
    body: JSON.stringify({ language, code }),
  });
}

export function execCode(language: string, code: string) {
  return request<RunnerStartResponse>("/runner/exec", {
    method: "POST",
    body: JSON.stringify({ language, code }),
  });
}

export function stopRunnerRun(runId: string) {
  return request<{ run_id: string; status: string }>(`/runner/stop/${encodeURIComponent(runId)}`, {
    method: "POST",
  });
}

export function streamRunnerRun(
  runId: string,
  onEvent: (event: RunnerEvent) => void,
  onError: (message: string) => void,
): () => void {
  const controller = new AbortController();
  let closed = false;
  let sawExitEvent = false;

  void getBackendRuntime()
    .then((runtime) => {
      if (closed) {
        return;
      }

      void fetch(`${buildApiUrl(runtime)}/runner/stream/${encodeURIComponent(runId)}`, {
        method: "GET",
        headers: {
          Accept: "text/event-stream",
          ...(runtime.token ? { "X-Atlas-Instance-Token": runtime.token } : {}),
        },
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok) {
            const contentType = response.headers.get("content-type") ?? "";
            const payload =
              contentType.includes("application/json")
                ? await response.json()
                : { detail: await response.text() };
            throw new Error(extractResponseErrorMessage(payload, response.statusText));
          }

          const body = response.body;
          if (!body) {
            throw new Error("Runner stream body was empty.");
          }

          const reader = body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          const dispatch = (block: string) => {
            const dataLines: string[] = [];
            for (const line of block.split(/\r?\n/)) {
              if (!line || line.startsWith(":") || line.startsWith("event:")) {
                continue;
              }
              if (line.startsWith("data:")) {
                dataLines.push(line.slice(5).trimStart());
              }
            }
            if (!dataLines.length) {
              return;
            }
            const parsed = JSON.parse(dataLines.join("\n")) as RunnerEvent;
            if (parsed.type === "exit") {
              sawExitEvent = true;
            }
            onEvent(parsed);
          };

          while (!closed) {
            const { value, done } = await reader.read();
            buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
            let idx = buffer.search(/\r?\n\r?\n/);
            while (idx !== -1) {
              const match = buffer.slice(idx).match(/^\r?\n\r?\n/);
              const sepLen = match?.[0].length ?? 2;
              const block = buffer.slice(0, idx);
              buffer = buffer.slice(idx + sepLen);
              dispatch(block);
              idx = buffer.search(/\r?\n\r?\n/);
            }
            if (done) {
              if (buffer.trim()) {
                dispatch(buffer);
              }
              break;
            }
          }

          if (!closed && !sawExitEvent) {
            throw new Error("Runner stream disconnected.");
          }
        })
        .catch((error) => {
          if (closed || controller.signal.aborted) {
            return;
          }
          onError(error instanceof Error ? error.message : "Runner stream disconnected.");
        });
    })
    .catch((error) => {
      if (closed || controller.signal.aborted) {
        return;
      }
      onError(error instanceof Error ? error.message : "Atlas runtime is unavailable.");
    });

  return () => {
    closed = true;
    controller.abort();
  };
}

export function streamRun(
  mode: RunMode,
  runId: string,
  onEvent: (event: RunStatusEvent) => void,
  onError: (message: string) => void,
): () => void {
  const controller = new AbortController();
  let closed = false;
  let sawTerminalEvent = false;
  const deliveredLegacyEventFingerprints = new Set<string>();
  const pendingSequencedEvents = new Map<number, RunStatusEvent>();
  let nextExpectedSequence = 1;
  let replayDedupeCapacityReached = false;
  const maxReconnectAttempts = 3;
  const maxReplayDedupeEvents = 20_000;

  const consumeStream = async () => {
    const runtime = await getBackendRuntime();
    if (closed) {
      return;
    }

    const response = await fetch(
      `${buildApiUrl(runtime)}/${mode}/stream/${encodeURIComponent(runId)}`,
      {
        method: "GET",
        headers: {
          Accept: "text/event-stream",
          ...(runtime.token ? { "X-Atlas-Instance-Token": runtime.token } : {}),
        },
        signal: controller.signal,
      },
    );
    if (!response.ok) {
      const contentType = response.headers.get("content-type") ?? "";
      const payload =
        contentType.includes("application/json")
          ? await response.json()
          : { detail: await response.text() };
      throw new Error(extractResponseErrorMessage(payload, response.statusText));
    }

    const body = response.body;
    if (!body) {
      throw new Error("Atlas stream body was empty.");
    }

    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const deliverEvent = (event: RunStatusEvent) => {
      if (closed || sawTerminalEvent) {
        return;
      }
      if (event.type === "run_completed" || event.type === "run_failed") {
        sawTerminalEvent = true;
      }
      onEvent(event);
    };

    const drainSequencedEvents = (event: RunStatusEvent) => {
      deliverEvent(event);
      const eventRangeEnd = runEventSequenceEnd(event);
      for (const pendingSequence of pendingSequencedEvents.keys()) {
        if (pendingSequence <= eventRangeEnd) {
          pendingSequencedEvents.delete(pendingSequence);
        }
      }
      nextExpectedSequence = eventRangeEnd + 1;
      while (!closed && !sawTerminalEvent) {
        const pending = pendingSequencedEvents.get(nextExpectedSequence);
        if (!pending) {
          break;
        }
        pendingSequencedEvents.delete(nextExpectedSequence);
        deliverEvent(pending);
        const pendingRangeEnd = runEventSequenceEnd(pending);
        for (const pendingSequence of pendingSequencedEvents.keys()) {
          if (pendingSequence <= pendingRangeEnd) {
            pendingSequencedEvents.delete(pendingSequence);
          }
        }
        nextExpectedSequence = pendingRangeEnd + 1;
      }
      if (sawTerminalEvent) {
        pendingSequencedEvents.clear();
      }
    };

    const dispatchEventBlock = (block: string) => {
      if (closed || sawTerminalEvent) {
        return;
      }
      const dataLines: string[] = [];
      for (const line of block.split(/\r?\n/)) {
        if (!line || line.startsWith(":") || line.startsWith("event:")) {
          continue;
        }
        if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trimStart());
        }
      }
      if (!dataLines.length) {
        return;
      }

      const parsed = JSON.parse(dataLines.join("\n")) as RunStatusEvent;
      const sequence = parsed.sequence;
      if (typeof sequence === "number" && Number.isSafeInteger(sequence) && sequence > 0) {
        if (sequence < nextExpectedSequence || pendingSequencedEvents.has(sequence)) {
          return;
        }
        if (sequence > nextExpectedSequence) {
          if (pendingSequencedEvents.size >= maxReplayDedupeEvents) {
            replayDedupeCapacityReached = true;
            throw new Error("Atlas stream exceeded the safe out-of-order event limit.");
          }
          pendingSequencedEvents.set(sequence, parsed);
          return;
        }
        drainSequencedEvents(parsed);
        return;
      }

      const fingerprint = runEventFingerprint(parsed);
      if (deliveredLegacyEventFingerprints.has(fingerprint)) {
        return;
      }
      if (deliveredLegacyEventFingerprints.size < maxReplayDedupeEvents) {
        deliveredLegacyEventFingerprints.add(fingerprint);
      } else {
        replayDedupeCapacityReached = true;
      }
      deliverEvent(parsed);
    };

    while (!closed) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

      let separatorIndex = buffer.search(/\r?\n\r?\n/);
      while (separatorIndex !== -1) {
        const separatorMatch = buffer.slice(separatorIndex).match(/^\r?\n\r?\n/);
        const separatorLength = separatorMatch?.[0].length ?? 2;
        const block = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + separatorLength);
        dispatchEventBlock(block);
        separatorIndex = buffer.search(/\r?\n\r?\n/);
      }

      if (done) {
        if (buffer.trim()) {
          dispatchEventBlock(buffer);
        }
        break;
      }
    }

    if (!closed && !sawTerminalEvent) {
      throw new Error("Atlas stream disconnected.");
    }
  };

  void (async () => {
    let reconnectAttempt = 0;
    while (!closed && !sawTerminalEvent) {
      try {
        await consumeStream();
      } catch (error) {
        if (closed || controller.signal.aborted) {
          return;
        }
        if (replayDedupeCapacityReached) {
          onError("Atlas stream disconnected after exceeding the safe replay limit.");
          return;
        }
        if (reconnectAttempt >= maxReconnectAttempts) {
          onError(error instanceof Error ? error.message : "Atlas stream disconnected.");
          return;
        }
        reconnectAttempt += 1;
        invalidateBackendRuntime();
        await sleep(Math.min(2000, 250 * 2 ** (reconnectAttempt - 1)));
      }
    }
  })();

  return () => {
    closed = true;
    controller.abort();
  };
}

export function invalidateBackendRuntime() {
  runtimePromise = null;
}

export async function restartManagedBackend(options?: { attempts?: number; delayMs?: number }) {
  await invoke("restart_backend");
  invalidateBackendRuntime();
  return waitForBackendReady(options);
}

export async function openExternalUrl(url: string) {
  try {
    await invoke("open_external_url", { url });
  } catch (error) {
    if (isTauri()) {
      throw error instanceof Error ? error : new Error("Could not open external URL.");
    }
    const opened = window.open(url, "_blank", "noopener,noreferrer");
    if (!opened) {
      throw error instanceof Error ? error : new Error("Could not open external URL.");
    }
  }
}

export async function getAppDiagnostics() {
  return invoke<AppDiagnostics>("app_diagnostics");
}

export async function openAppLocation(location: "data" | "logs") {
  return invoke("open_app_location", { location });
}

export async function waitForBackendReady(options?: { attempts?: number; delayMs?: number }) {
  const attempts = Math.max(1, options?.attempts ?? 30);
  const delayMs = Math.max(50, options?.delayMs ?? 250);
  let lastError: unknown = null;

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await getStatus();
    } catch (error) {
      lastError = error;
      invalidateBackendRuntime();
      if (attempt < attempts - 1) {
        await sleep(delayMs);
      }
    }
  }

  if (lastError instanceof Error) {
    throw lastError;
  }
  throw new Error("Atlas backend did not become ready.");
}

async function getBackendRuntime(): Promise<BackendRuntime> {
  if (!runtimePromise) {
    runtimePromise = resolveBackendRuntime();
  }
  return runtimePromise;
}

async function resolveBackendRuntime(): Promise<BackendRuntime> {
  const configuredRuntime = configuredBrowserBackendRuntime();
  if (configuredRuntime) {
    lastRuntime = configuredRuntime;
    return configuredRuntime;
  }
  try {
    const runtime = await invoke<BackendRuntime>("backend_runtime");
    lastRuntime = runtime;
    return runtime;
  } catch (error) {
    if (lastRuntime) {
      return lastRuntime;
    }
    throw error instanceof Error ? error : new Error("Atlas backend runtime is unavailable.");
  }
}

function buildApiUrl(runtime: BackendRuntime): string {
  if (runtime.baseUrl) {
    return runtime.baseUrl;
  }
  return `http://${runtime.host}:${runtime.port}`;
}

function configuredBrowserBackendRuntime(): BackendRuntime | null {
  const rawUrl = String(import.meta.env.VITE_ATLAS_BACKEND_URL ?? "").trim();
  const token = String(import.meta.env.VITE_ATLAS_BACKEND_TOKEN ?? "").trim();
  return parseLocalBackendRuntime(rawUrl, token);
}

export function parseLocalBackendRuntime(rawUrl: string, token = ""): BackendRuntime | null {
  const configuredUrl = rawUrl.trim();
  if (!configuredUrl) {
    return null;
  }
  try {
    const url = new URL(configuredUrl);
    const hostname = url.hostname.toLowerCase();
    const isLoopback =
      hostname === "localhost" ||
      hostname === "[::1]" ||
      hostname === "::1" ||
      /^127(?:\.\d{1,3}){3}$/.test(hostname);
    const hasRootPathOnly = url.pathname === "/" && !url.search && !url.hash;
    if (
      url.protocol !== "http:" ||
      !isLoopback ||
      url.username ||
      url.password ||
      !hasRootPathOnly
    ) {
      throw new Error(
        "VITE_ATLAS_BACKEND_URL must be an HTTP loopback origin without credentials, a path, query, or fragment.",
      );
    }
    const port = url.port ? Number(url.port) : 80;
    return {
      host: url.hostname,
      port,
      token: token.trim(),
      baseUrl: url.origin,
    };
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("VITE_ATLAS_BACKEND_URL")) {
      throw error;
    }
    throw new Error("VITE_ATLAS_BACKEND_URL must be a valid absolute loopback URL.");
  }
}

async function requestWithRuntime<T>(runtime: BackendRuntime, path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeoutMs = isReadOnlyRequest(init)
    ? API_READ_REQUEST_TIMEOUT_MS
    : API_MUTATION_REQUEST_TIMEOUT_MS;
  const timeout = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  const callerSignal = init?.signal;
  const abortFromCaller = () => controller.abort();
  if (callerSignal?.aborted) {
    controller.abort();
  } else {
    callerSignal?.addEventListener("abort", abortFromCaller, { once: true });
  }
  try {
    const response = await fetch(`${buildApiUrl(runtime)}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(runtime.token ? { "X-Atlas-Instance-Token": runtime.token } : {}),
        ...(init?.headers ?? {}),
      },
    });
    if (!response.ok) {
      const contentType = response.headers.get("content-type") ?? "";
      const payload =
        contentType.includes("application/json") ? await response.json() : { detail: await response.text() };
      throw new Error(extractResponseErrorMessage(payload, response.statusText));
    }

    return (await response.json()) as T;
  } catch (error) {
    if (controller.signal.aborted && !callerSignal?.aborted) {
      throw new Error(`Atlas backend did not respond within ${timeoutMs / 1000} seconds.`);
    }
    throw error;
  } finally {
    globalThis.clearTimeout(timeout);
    callerSignal?.removeEventListener("abort", abortFromCaller);
  }
}

function isReadOnlyRequest(init?: RequestInit) {
  const method = String(init?.method ?? "GET").trim().toUpperCase();
  return method === "GET" || method === "HEAD" || method === "OPTIONS";
}

function runEventFingerprint(event: RunStatusEvent) {
  if (
    Number.isSafeInteger(event.sequence)
    && Number(event.sequence) > 0
  ) {
    return `seq:${event.sequence}`;
  }
  return `${event.timestamp}\u0000${event.type}\u0000${JSON.stringify(event.payload)}`;
}

function runEventSequenceEnd(event: RunStatusEvent) {
  const sequence = event.sequence;
  if (
    typeof sequence !== "number"
    || !Number.isSafeInteger(sequence)
    || sequence < 1
  ) {
    return 0;
  }
  const sequenceEnd = event.sequence_end;
  if (
    typeof sequenceEnd === "number"
    && Number.isSafeInteger(sequenceEnd)
    && sequenceEnd >= sequence
  ) {
    return sequenceEnd;
  }
  return sequence;
}

function extractResponseErrorMessage(payload: unknown, fallback: string) {
  if (!payload || typeof payload !== "object") {
    return fallback || "The request did not complete.";
  }
  const record = payload as Record<string, unknown>;
  return formatErrorDetail(record.detail ?? record.error, fallback || "The request did not complete.");
}

function formatErrorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string") {
    return detail.trim() || fallback;
  }
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => formatValidationDetail(item))
      .filter((item) => item.length > 0);
    return parts.join("; ") || fallback;
  }
  if (detail && typeof detail === "object") {
    const record = detail as Record<string, unknown>;
    const nested = record.detail ?? record.error ?? record.message ?? record.msg;
    if (nested !== undefined && nested !== detail) {
      return formatErrorDetail(nested, fallback);
    }
    try {
      return JSON.stringify(record);
    } catch {
      return fallback;
    }
  }
  return fallback;
}

function formatValidationDetail(detail: unknown): string {
  if (!detail || typeof detail !== "object") {
    return formatErrorDetail(detail, "");
  }
  const record = detail as Record<string, unknown>;
  const message = record.msg ?? record.message;
  const location = Array.isArray(record.loc)
    ? record.loc.filter((item) => item !== "body").join(".")
    : "";
  if (typeof message === "string" && message.trim()) {
    return location ? `${location}: ${message}` : message;
  }
  return formatErrorDetail(record, "");
}

function sleep(delayMs: number) {
  return new Promise<void>((resolve) => {
    window.setTimeout(resolve, delayMs);
  });
}
