import { WebviewWindow } from "@tauri-apps/api/webviewWindow";

const SERVER_LANGUAGES = [
  "python",
  "javascript",
  "typescript",
  "go",
  "rust",
  "c",
  "cpp",
  "java",
  "ruby",
  "php",
  "bash",
  "csharp",
  "kotlin",
  "swift",
  "perl",
  "lua",
  "r",
  "elixir",
  "dart",
] as const;

const CLIENT_LANGUAGES = ["html"] as const;

const LANGUAGE_ALIASES: Record<string, string> = {
  py: "python",
  python3: "python",
  js: "javascript",
  node: "javascript",
  ts: "typescript",
  golang: "go",
  rs: "rust",
  "c++": "cpp",
  cxx: "cpp",
  cc: "cpp",
  rb: "ruby",
  sh: "bash",
  shell: "bash",
  zsh: "bash",
  cs: "csharp",
  "c#": "csharp",
  kt: "kotlin",
  kts: "kotlin",
  pl: "perl",
  ex: "elixir",
  exs: "elixir",
  htm: "html",
};

export const RUNNABLE_LANGUAGES: readonly string[] = [...SERVER_LANGUAGES, ...CLIENT_LANGUAGES];

export function resolveRunnableLanguage(language: string): string | null {
  const normalized = (language || "").trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if ((RUNNABLE_LANGUAGES as readonly string[]).includes(normalized)) {
    return normalized;
  }
  if (normalized in LANGUAGE_ALIASES) {
    return LANGUAGE_ALIASES[normalized];
  }
  return null;
}

export function isClientLanguage(language: string): boolean {
  return (CLIENT_LANGUAGES as readonly string[]).includes(language);
}

export type PendingRun = {
  language: string;
  code: string;
  originUserId?: string;
  originThreadId?: string;
};

export type RunnerRepairRequest = {
  version: 1;
  requestId: string;
  language: string;
  code: string;
  diagnostics: string;
  originUserId: string;
  originThreadId: string;
  createdAt: number;
};

const PENDING_PREFIX = "atlas-runner:";
const MAX_PENDING_CODE_LENGTH = 1_000_000;
const MAX_RUNNER_SCOPE_ID_LENGTH = 512;
const RUNNER_REPAIR_REQUEST_TTL_MS = 10 * 60 * 1000;
export const RUNNER_REPAIR_REQUEST_STORAGE_KEY = "atlas-runner:repair-request";
export const RUNNER_REPAIR_BROADCAST_CHANNEL = "atlas-runner-repair";
export const MAX_RUNNER_REPAIR_SOURCE_LENGTH = 170_000;
export const MAX_RUNNER_REPAIR_DIAGNOSTICS_LENGTH = 24_000;

function storageKey(token: string) {
  return `${PENDING_PREFIX}${token}`;
}

export function stashPendingRun(token: string, payload: PendingRun) {
  try {
    window.localStorage.setItem(storageKey(token), JSON.stringify(payload));
  } catch {
    // Ignore quota or privacy failures; the consumer will show an error.
  }
}

export function discardPendingRun(token: string) {
  try {
    window.localStorage.removeItem(storageKey(token));
  } catch {
    // Best-effort cleanup.
  }
}

export function consumePendingRun(token: string): PendingRun | null {
  const key = storageKey(token);
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(key);
  } catch {
    raw = null;
  }
  if (!raw) {
    return null;
  }
  discardPendingRun(token);
  try {
    return normalizePendingRun(JSON.parse(raw));
  } catch {
    return null;
  }
}

function normalizePendingRun(value: unknown): PendingRun | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const candidate = value as {
    language?: unknown;
    code?: unknown;
    originUserId?: unknown;
    originThreadId?: unknown;
  };
  if (typeof candidate.language !== "string" || typeof candidate.code !== "string") {
    return null;
  }
  const language = resolveRunnableLanguage(candidate.language);
  if (!language || candidate.code.length === 0 || candidate.code.length > MAX_PENDING_CODE_LENGTH) {
    return null;
  }
  const originUserId = normalizeScopeId(candidate.originUserId);
  const originThreadId = normalizeScopeId(candidate.originThreadId);
  return {
    language,
    code: candidate.code,
    ...(originUserId && originThreadId ? { originUserId, originThreadId } : {}),
  };
}

function makeToken(): string {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export async function openRunnerWindow(payload: PendingRun) {
  const normalizedPayload = normalizePendingRun(payload);
  if (!normalizedPayload) {
    throw new Error("The runner request is invalid.");
  }
  const token = makeToken();
  stashPendingRun(token, normalizedPayload);

  const label = `runner-${token}`;
  const title = `Atlas Run - ${normalizedPayload.language}`;
  const url = `index.html#/runner/${token}`;

  const runner = new WebviewWindow(label, {
    url,
    title,
    width: 960,
    height: 680,
    resizable: true,
    focus: true,
  });

  try {
    await new Promise<void>((resolve, reject) => {
      runner.once("tauri://created", () => resolve());
      runner.once("tauri://error", (event) => {
        reject(
          new Error(
            String(
              (event.payload as { message?: string } | undefined)?.message ??
                event.payload ??
                "unknown",
            ),
          ),
        );
      });
    });
  } catch (error) {
    discardPendingRun(token);
    throw error;
  }

  return runner;
}

export function createRunnerRepairRequest({
  language,
  code,
  diagnostics,
  originUserId,
  originThreadId,
  now = Date.now(),
}: {
  language: string;
  code: string;
  diagnostics: string;
  originUserId: string;
  originThreadId: string;
  now?: number;
}): RunnerRepairRequest {
  const normalizedLanguage = resolveRunnableLanguage(language);
  const normalizedUserId = normalizeScopeId(originUserId);
  const normalizedThreadId = normalizeScopeId(originThreadId);
  if (!normalizedLanguage || !normalizedUserId || !normalizedThreadId) {
    throw new Error("This run is not linked to an active Atlas chat. Copy the repair request instead.");
  }
  if (!code || code.length > MAX_RUNNER_REPAIR_SOURCE_LENGTH) {
    throw new Error(
      code
        ? "This source is too large for an Atlas repair draft. Copy the source and diagnostics manually."
        : "The failed run has no source to repair.",
    );
  }
  return {
    version: 1,
    requestId: makeRepairRequestId(),
    language: normalizedLanguage,
    code,
    diagnostics: boundRepairDiagnostics(diagnostics),
    originUserId: normalizedUserId,
    originThreadId: normalizedThreadId,
    createdAt: now,
  };
}

export function stashRunnerRepairRequest(request: RunnerRepairRequest) {
  window.localStorage.setItem(RUNNER_REPAIR_REQUEST_STORAGE_KEY, JSON.stringify(request));
  if (typeof BroadcastChannel === "function") {
    try {
      const channel = new BroadcastChannel(RUNNER_REPAIR_BROADCAST_CHANNEL);
      channel.postMessage({ requestId: request.requestId });
      channel.close();
    } catch {
      // The storage event and durable stored request remain available as fallbacks.
    }
  }
}

export function readRunnerRepairRequest(
  raw: string | null,
  now = Date.now(),
): RunnerRepairRequest | null {
  if (!raw) {
    return null;
  }
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!value || typeof value !== "object") {
    return null;
  }
  const candidate = value as Partial<RunnerRepairRequest>;
  const language = typeof candidate.language === "string"
    ? resolveRunnableLanguage(candidate.language)
    : null;
  const originUserId = normalizeScopeId(candidate.originUserId);
  const originThreadId = normalizeScopeId(candidate.originThreadId);
  if (
    candidate.version !== 1 ||
    typeof candidate.requestId !== "string" ||
    !candidate.requestId ||
    candidate.requestId.length > 128 ||
    !language ||
    typeof candidate.code !== "string" ||
    !candidate.code ||
    candidate.code.length > MAX_RUNNER_REPAIR_SOURCE_LENGTH ||
    typeof candidate.diagnostics !== "string" ||
    candidate.diagnostics.length > MAX_RUNNER_REPAIR_DIAGNOSTICS_LENGTH ||
    !originUserId ||
    !originThreadId ||
    typeof candidate.createdAt !== "number" ||
    !Number.isFinite(candidate.createdAt) ||
    candidate.createdAt > now + 60_000 ||
    now - candidate.createdAt > RUNNER_REPAIR_REQUEST_TTL_MS
  ) {
    return null;
  }
  return {
    version: 1,
    requestId: candidate.requestId,
    language,
    code: candidate.code,
    diagnostics: candidate.diagnostics,
    originUserId,
    originThreadId,
    createdAt: candidate.createdAt,
  };
}

export function discardRunnerRepairRequest(requestId?: string) {
  try {
    if (requestId) {
      const current = readRunnerRepairRequest(
        window.localStorage.getItem(RUNNER_REPAIR_REQUEST_STORAGE_KEY),
      );
      if (current?.requestId !== requestId) {
        return;
      }
    }
    window.localStorage.removeItem(RUNNER_REPAIR_REQUEST_STORAGE_KEY);
  } catch {
    // Best-effort cleanup; an expired request is ignored by the reader.
  }
}

export function buildRunnerRepairPrompt(request: RunnerRepairRequest): string {
  const fence = markdownFenceFor(`${request.diagnostics}\n${request.code}`);
  return [
    "The following program failed in Atlas's local runner. Diagnose the source-code defect and provide a corrected version.",
    "",
    "Requirements:",
    "- Preserve the intended behavior and UX unless a change is required to fix the failure.",
    `- Return the complete corrected program in one ${request.language} code block; do not omit unchanged sections.`,
    "- Do not patch only the reported line; inspect the full startup path, data-shape assumptions, and adjacent APIs for the next immediate failure.",
    "- Check imports, names, and APIs before answering.",
    "- Keep Atlas's offline runner isolation intact; do not suggest weakening its security or enabling network access.",
    "",
    "Runner diagnostics:",
    `${fence}text`,
    request.diagnostics || "The process failed without diagnostic output.",
    fence,
    "",
    `Original ${request.language} source:`,
    `${fence}${request.language}`,
    request.code,
    fence,
  ].join("\n");
}

function normalizeScopeId(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim();
  if (!normalized || normalized.length > MAX_RUNNER_SCOPE_ID_LENGTH) {
    return null;
  }
  return normalized;
}

function boundRepairDiagnostics(value: string): string {
  const normalized = value.replace(/\r\n?/g, "\n").trim();
  if (normalized.length <= MAX_RUNNER_REPAIR_DIAGNOSTICS_LENGTH) {
    return normalized;
  }
  const omission = "[Earlier runner output omitted]\n";
  return `${omission}${normalized.slice(-(MAX_RUNNER_REPAIR_DIAGNOSTICS_LENGTH - omission.length))}`;
}

function markdownFenceFor(value: string): string {
  const longest = Math.max(0, ...Array.from(value.matchAll(/`+/g), (match) => match[0].length));
  return "`".repeat(Math.max(3, longest + 1));
}

function makeRepairRequestId(): string {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${makeToken()}`;
}
