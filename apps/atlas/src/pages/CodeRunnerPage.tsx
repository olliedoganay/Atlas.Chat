import { getCurrentWindow } from "@tauri-apps/api/window";
import { Check, Code2, Copy, PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen, Play, RotateCcw, Square, Terminal, Wrench } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";

import {
  execCode,
  getPythonGuiRuntimeStatus,
  getRunnerStatus,
  prepareRunnerRuntime,
  stopRunnerRun,
  streamRunnerRun,
  type RunnerEvent,
  type RunnerRuntimeStatus,
  type RunnerStatus,
} from "../lib/api";
import {
  consumePendingRun,
  createRunnerRepairRequest,
  isClientLanguage,
  stashRunnerRepairRequest,
} from "../lib/runner";
import { useAtlasStore } from "../store/useAtlasStore";

type OutputLine = {
  stream: "stdout" | "stderr";
  text: string;
};

type Phase = "loading" | "preparing" | "docker-down" | "running" | "finished" | "error" | "idle";
type RepairState = "idle" | "drafting" | "ready" | "error";

export const CLIENT_PREVIEW_SANDBOX = "allow-scripts allow-forms allow-pointer-lock";
export const SERVER_PREVIEW_SANDBOX =
  "allow-scripts allow-forms allow-pointer-lock allow-same-origin";
export const CLIENT_PREVIEW_MESSAGE_SOURCE = "atlas-client-preview";
export const MAX_CLIENT_PREVIEW_CONSOLE_CHARS = 8_192;
export const RUNNER_PREVIEW_PROBE_TIMEOUT_MS = 3_000;
export const RUNNER_CLOSE_STOP_TIMEOUT_MS = 3_000;
export const RUNNER_RUNTIME_POLL_INTERVAL_MS = 750;
export const CLIENT_PREVIEW_CSP = [
  "default-src 'none'",
  "script-src 'unsafe-inline' 'unsafe-eval' data: blob:",
  "script-src-elem 'unsafe-inline' 'unsafe-eval' data: blob:",
  "script-src-attr 'unsafe-inline'",
  "style-src 'unsafe-inline' data: blob:",
  "style-src-elem 'unsafe-inline' data: blob:",
  "style-src-attr 'unsafe-inline'",
  "img-src data: blob:",
  "font-src data: blob:",
  "media-src data: blob:",
  "connect-src 'none'",
  "worker-src data: blob:",
  "child-src data: blob:",
  "frame-src data: blob:",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
].join("; ");

type ClientPreviewConsoleLevel = "log" | "warn" | "error";

type ClientPreviewEvent = {
  source: typeof CLIENT_PREVIEW_MESSAGE_SOURCE;
  channel: string;
  type: "ready" | "console" | "error";
  level?: ClientPreviewConsoleLevel;
  text?: string;
};

type ClientPreviewLine = {
  level: ClientPreviewConsoleLevel;
  text: string;
};

export function buildClientPreviewDocument(code: string, channel: string): string {
  // Trusted policy and diagnostics must precede every byte of user HTML.
  // Regex insertion can be redirected into a fake <head>, while parsing the
  // source in the host document can itself fetch attacker-controlled resources.
  // Leaving our head open lets the browser naturally merge either a complete
  // document or a fragment after the policy has already taken effect.
  return [
    "<!DOCTYPE html><html><head>",
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1">',
    buildClientPreviewCspMeta(),
    `<script>${buildClientPreviewBootstrapScript(channel)}</script>`,
    code,
    "</html>",
  ].join("");
}

export function buildClientPreviewBlob(code: string, channel = ""): Blob {
  return new Blob([channel ? buildClientPreviewDocument(code, channel) : code], { type: "text/html;charset=utf-8" });
}

function buildClientPreviewCspMeta(): string {
  const escaped = CLIENT_PREVIEW_CSP.replace(/&/g, "&amp;").replace(/"/g, "&quot;");
  return `<meta http-equiv="Content-Security-Policy" content="${escaped}">`;
}

function buildClientPreviewBootstrapScript(channel: string): string {
  const source = JSON.stringify(CLIENT_PREVIEW_MESSAGE_SOURCE);
  const channelValue = JSON.stringify(channel);
  const maxConsoleChars = JSON.stringify(MAX_CLIENT_PREVIEW_CONSOLE_CHARS);
  const script = [
    "(() => {",
    `  const source = ${source};`,
    `  const channel = ${channelValue};`,
    `  const maxConsoleChars = ${maxConsoleChars};`,
    '  const send = (payload) => { try { parent.postMessage({ source, channel, ...payload }, "*"); } catch {} };',
    "  const installStorageFallback = (name) => {",
    "    try {",
    "      const existing = window[name];",
    '      const probe = "__atlas_preview_storage_probe__";',
    '      existing.setItem(probe, "1");',
    "      existing.removeItem(probe);",
    "      return;",
    "    } catch {}",
    "    const data = new Map();",
    "    const fallback = {",
    "      get length() { return data.size; },",
    "      key(index) { return Array.from(data.keys())[Number(index)] ?? null; },",
    "      getItem(key) { key = String(key); return data.has(key) ? data.get(key) : null; },",
    "      setItem(key, value) { data.set(String(key), String(value)); },",
    "      removeItem(key) { data.delete(String(key)); },",
    "      clear() { data.clear(); },",
    "    };",
    "    try { Object.defineProperty(window, name, { value: fallback, configurable: true }); } catch {}",
    "  };",
    '  installStorageFallback("localStorage");',
    '  installStorageFallback("sessionStorage");',
    "  const stringify = (value) => {",
    '    try {',
    '      if (typeof value === "string") return value;',
    "      if (value instanceof Error) return value.stack || value.message;",
    "      const json = JSON.stringify(value);",
    "      return json === undefined ? String(value) : json;",
    "    } catch {",
    "      return String(value);",
    "    }",
    "  };",
    "  const boundedText = (value) => {",
    "    const text = stringify(value);",
    '    return text.length > maxConsoleChars ? `${text.slice(0, maxConsoleChars)}\\n[atlas-preview] output truncated` : text;',
    "  };",
    '  ["log", "warn", "error"].forEach((level) => {',
    "    const original = console[level];",
    "    console[level] = (...args) => {",
    '      send({ type: "console", level, text: boundedText(args.map(boundedText).join(" ")) });',
    "      original.apply(console, args);",
    "    };",
    "  });",
    '  window.addEventListener("error", (event) => {',
    "    const text = [event.message, event.filename || \"\", event.lineno ? String(event.lineno) : \"\", event.colno ? String(event.colno) : \"\"].filter(Boolean).join(\":\");",
    '    send({ type: "error", level: "error", text });',
    "  });",
    '  window.addEventListener("unhandledrejection", (event) => {',
    '    send({ type: "error", level: "error", text: boundedText(event.reason) });',
    "  });",
    '  window.addEventListener("securitypolicyviolation", (event) => {',
    "    const blocked = event.blockedURI || \"inline\";",
    '    send({ type: "error", level: "error", text: `CSP blocked ${event.violatedDirective}: ${blocked}` });',
    "  });",
    '  window.addEventListener("load", () => {',
    '    send({ type: "ready" });',
    "    setTimeout(() => {",
    "      try {",
    "        window.focus();",
    "        document.body?.focus?.();",
    "      } catch {}",
    "    }, 0);",
    "  });",
    "})();",
  ].join("\n");
  return script;
}

export function CodeRunnerPage() {
  const { token = "" } = useParams();
  const launchUserId = useAtlasStore((state) => state.currentUserId);
  const launchThreadId = useAtlasStore((state) => state.currentThreadId);
  const [phase, setPhase] = useState<Phase>("loading");
  const [language, setLanguage] = useState<string>("");
  const [code, setCode] = useState<string>("");
  const [runId, setRunId] = useState<string | null>(null);
  const [output, setOutput] = useState<OutputLine[]>([]);
  const [exitCode, setExitCode] = useState<number | null>(null);
  const [durationMs, setDurationMs] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [dockerReason, setDockerReason] = useState<string>("");
  const [vncUrl, setVncUrl] = useState<string | null>(null);
  const [webUrl, setWebUrl] = useState<string | null>(null);
  const [vncReady, setVncReady] = useState(false);
  const [clientPreviewNonce, setClientPreviewNonce] = useState(0);
  const [serverLogsOpen, setServerLogsOpen] = useState(false);
  const [stopPending, setStopPending] = useState(false);
  const [sourceOpen, setSourceOpen] = useState(true);
  const [sourceCopied, setSourceCopied] = useState(false);
  const [runtimeStatus, setRuntimeStatus] = useState<RunnerRuntimeStatus | null>(null);
  const [repairState, setRepairState] = useState<RepairState>("idle");
  const [repairMessage, setRepairMessage] = useState("");
  const streamDisposer = useRef<(() => void) | null>(null);
  const outputRef = useRef<HTMLDivElement | null>(null);
  const currentRunId = useRef<string | null>(null);
  const runAttemptRef = useRef(0);
  const initialPayloadRef = useRef<ReturnType<typeof consumePendingRun> | undefined>(undefined);
  const closeInProgressRef = useRef(false);
  const copyResetTimerRef = useRef<number | null>(null);

  const clientLang = useMemo(() => (language ? isClientLanguage(language) : false), [language]);
  const showVncPane = Boolean(vncUrl && vncReady && phase !== "finished" && phase !== "error");
  const showWebPane = Boolean(webUrl && phase !== "finished" && phase !== "error");
  const showServerPreview = showVncPane || showWebPane;
  const outputLineCount = output.length + (errorMessage ? 1 : 0);
  const activityLabel = useMemo(
    () => runnerActivityLabel({ clientLang, output, phase, runtimeStatus, vncReady, vncUrl, webUrl }),
    [clientLang, output, phase, runtimeStatus, vncReady, vncUrl, webUrl],
  );
  const failedRun = phase === "error" || (phase === "finished" && exitCode !== null && exitCode !== 0);

  useEffect(() => {
    if (!token) {
      setPhase("error");
      setErrorMessage("Runner token missing from URL.");
      return;
    }
    if (initialPayloadRef.current === undefined) {
      initialPayloadRef.current = consumePendingRun(token);
    }
    const payload = initialPayloadRef.current;
    if (!payload) {
      setPhase("error");
      setErrorMessage("This run window lost its payload. Close and try again.");
      return;
    }
    setLanguage(payload.language);
    setCode(payload.code);
  }, [token]);

  useEffect(() => {
    if (!language) {
      return;
    }
    const appWindow = getSafeCurrentWindow();
    if (!appWindow) {
      return;
    }
    void appWindow.setTitle(`Atlas Run - ${language}`).catch(() => undefined);
  }, [language]);

  const scrollToEnd = useCallback(() => {
    requestAnimationFrame(() => {
      const node = outputRef.current;
      if (node) {
        node.scrollTop = node.scrollHeight;
      }
    });
  }, []);

  const handleEvent = useCallback(
    (event: RunnerEvent) => {
      if (event.type === "output") {
        if (event.chunk.includes("GUI ready on port")) {
          setVncReady(true);
          setServerLogsOpen(false);
        }
        setOutput((prev) => [...prev, { stream: event.stream, text: event.chunk }]);
        scrollToEnd();
      } else if (event.type === "exit") {
        currentRunId.current = null;
        setExitCode(event.code);
        setDurationMs(event.duration_ms);
        setPhase("finished");
      }
    },
    [scrollToEnd],
  );

  const beginServerRun = useCallback(async () => {
    if (!language || !code) {
      return;
    }
    const runAttempt = runAttemptRef.current + 1;
    runAttemptRef.current = runAttempt;
    setPhase("loading");
    setOutput([]);
    setExitCode(null);
    setDurationMs(null);
    setErrorMessage(null);
    setVncUrl(null);
    setWebUrl(null);
    setVncReady(false);
    setServerLogsOpen(false);
    setStopPending(false);
    setRuntimeStatus(null);
    setRepairState("idle");
    setRepairMessage("");

    let status: RunnerStatus;
    try {
      status = await getRunnerStatus();
    } catch (error) {
      if (runAttempt !== runAttemptRef.current) {
        return;
      }
      setPhase("error");
      setErrorMessage(error instanceof Error ? error.message : "Failed to contact backend.");
      return;
    }

    if (runAttempt !== runAttemptRef.current) {
      return;
    }
    if (!status.available) {
      setPhase("docker-down");
      setDockerReason(status.reason ?? "Docker Desktop is not running.");
      return;
    }

    try {
      const preparation = await prepareRunnerRuntime(language, code);
      if (runAttempt !== runAttemptRef.current) {
        return;
      }
      if (preparation.required) {
        if (!preparation.runtime) {
          throw new Error("Atlas did not return Python GUI runtime preparation status.");
        }
        setPhase("preparing");
        const prepared = await waitForRunnerRuntime(
          preparation.runtime,
          (runtime) => {
            if (runAttempt !== runAttemptRef.current) {
              return;
            }
            setRuntimeStatus(runtime);
            setOutput(runtimeLogLines(runtime));
          },
          () => runAttempt === runAttemptRef.current,
        );
        if (!prepared || runAttempt !== runAttemptRef.current) {
          return;
        }
      }
      const started = await execCode(language, code);
      if (runAttempt !== runAttemptRef.current) {
        void stopRunnerRun(started.run_id).catch(() => undefined);
        return;
      }
      currentRunId.current = started.run_id;
      setRunId(started.run_id);
      setVncUrl(started.vnc_url ?? null);
      setWebUrl(started.web_url ?? null);
      setServerLogsOpen(false);
      setPhase("running");
      streamDisposer.current?.();
      streamDisposer.current = streamRunnerRun(
        started.run_id,
        (event) => {
          if (runAttempt === runAttemptRef.current) {
            handleEvent(event);
          }
        },
        (message) => {
          if (runAttempt !== runAttemptRef.current) {
            return;
          }
          setErrorMessage(message);
          setPhase("error");
        },
      );
    } catch (error) {
      if (runAttempt !== runAttemptRef.current) {
        return;
      }
      setPhase("error");
      setErrorMessage(error instanceof Error ? error.message : "Failed to start run.");
    }
  }, [language, code, handleEvent]);

  useEffect(() => {
    if (!language) {
      return;
    }
    if (clientLang) {
      setPhase("idle");
      return;
    }
    void beginServerRun();
  }, [language, clientLang, beginServerRun]);

  useEffect(() => {
    return () => {
      runAttemptRef.current += 1;
      streamDisposer.current?.();
      streamDisposer.current = null;
      const id = currentRunId.current;
      if (id) {
        void stopRunnerRun(id).catch(() => undefined);
      }
    };
  }, []);

  useEffect(() => {
    let unlisten: (() => void) | null = null;
    let cancelled = false;
    const appWindow = getSafeCurrentWindow();
    if (!appWindow) {
      return undefined;
    }
    void appWindow
      .onCloseRequested(async (event) => {
        if (closeInProgressRef.current) {
          return;
        }
        streamDisposer.current?.();
        streamDisposer.current = null;
        const id = currentRunId.current;
        if (!id) {
          return;
        }
        event.preventDefault();
        closeInProgressRef.current = true;
        currentRunId.current = null;
        let stopTimeout: number | null = null;
        try {
          await Promise.race([
            stopRunnerRun(id).catch(() => undefined),
            new Promise<void>((resolve) => {
              stopTimeout = window.setTimeout(resolve, RUNNER_CLOSE_STOP_TIMEOUT_MS);
            }),
          ]);
        } finally {
          if (stopTimeout !== null) {
            window.clearTimeout(stopTimeout);
          }
          await appWindow.close().catch(() => undefined);
        }
      })
      .then((fn) => {
        if (cancelled) {
          fn();
        } else {
          unlisten = fn;
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  const stopRun = useCallback(async () => {
    const id = currentRunId.current;
    if (!id || stopPending) {
      return;
    }
    setStopPending(true);
    try {
      await stopRunnerRun(id);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to stop run.");
    } finally {
      setStopPending(false);
    }
  }, [stopPending]);

  const rerun = useCallback(async () => {
    if (clientLang) {
      setPhase("idle");
      setErrorMessage(null);
      setClientPreviewNonce((current) => current + 1);
      return;
    }
    runAttemptRef.current += 1;
    streamDisposer.current?.();
    streamDisposer.current = null;
    const previousRunId = currentRunId.current;
    currentRunId.current = null;
    setRunId(null);
    if (previousRunId) {
      await stopRunnerRun(previousRunId).catch(() => undefined);
    }
    await beginServerRun();
  }, [beginServerRun, clientLang]);

  const copySource = useCallback(async () => {
    try {
      if (!navigator.clipboard?.writeText) {
        return;
      }
      await navigator.clipboard.writeText(code);
      setSourceCopied(true);
      if (copyResetTimerRef.current !== null) {
        window.clearTimeout(copyResetTimerRef.current);
      }
      copyResetTimerRef.current = window.setTimeout(() => {
        copyResetTimerRef.current = null;
        setSourceCopied(false);
      }, 1200);
    } catch {
      // Clipboard access can be denied in browser preview or locked-down desktop contexts.
    }
  }, [code]);

  const fixWithAtlas = useCallback(async () => {
    if (!failedRun || repairState === "drafting") {
      return;
    }
    const payload = initialPayloadRef.current;
    setRepairState("drafting");
    setRepairMessage("");
    try {
      const request = createRunnerRepairRequest({
        language,
        code,
        diagnostics: buildRunnerRepairDiagnostics({ errorMessage, exitCode, output }),
        originUserId: payload?.originUserId ?? launchUserId,
        originThreadId: payload?.originThreadId ?? launchThreadId,
      });
      stashRunnerRepairRequest(request);
      setRepairState("ready");
      setRepairMessage("Repair draft ready in the original Atlas chat. Review it before sending.");
      const appWindow = getSafeCurrentWindow();
      if (appWindow && typeof appWindow.close === "function") {
        await appWindow.close().catch(() => undefined);
      }
    } catch (error) {
      setRepairState("error");
      setRepairMessage(
        error instanceof Error ? error.message : "Atlas could not create the repair draft.",
      );
    }
  }, [code, errorMessage, exitCode, failedRun, language, launchThreadId, launchUserId, output, repairState]);

  useEffect(() => {
    setSourceCopied(false);
  }, [code, language]);

  useEffect(
    () => () => {
      if (copyResetTimerRef.current !== null) {
        window.clearTimeout(copyResetTimerRef.current);
      }
    },
    [],
  );

  if (!language) {
    return (
      <div className="runner-shell">
        <div className="runner-center">
          <p>{errorMessage ?? "Preparing run..."}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="runner-shell">
      <header className="runner-header">
        <div className="runner-title">
          <span className="runner-lang-pill">{language}</span>
          <RunnerStatusBadge phase={phase} exitCode={exitCode} />
          {activityLabel ? <span className="runner-activity-label">{activityLabel}</span> : null}
        </div>
        <div className="runner-actions">
          <button
            aria-pressed={sourceOpen}
            className="ghost-button compact-button"
            onClick={() => setSourceOpen((current) => !current)}
            title={sourceOpen ? "Hide source" : "Show source"}
            type="button"
          >
            {sourceOpen ? <PanelLeftClose size={14} /> : <PanelLeftOpen size={14} />}
            Source
          </button>
          {phase === "running" ? (
            <button
              className="ghost-button compact-button runner-stop"
              disabled={stopPending}
              onClick={() => void stopRun()}
              type="button"
            >
              <Square size={14} /> {stopPending ? "Stopping..." : "Stop"}
            </button>
          ) : null}
          {failedRun ? (
            <button
              className="primary-button compact-button runner-fix"
              disabled={repairState === "drafting"}
              onClick={() => void fixWithAtlas()}
              title="Create a reviewable repair draft in the original Atlas chat"
              type="button"
            >
              <Wrench size={14} />
              {repairState === "drafting" ? "Preparing draft..." : "Fix with Atlas"}
            </button>
          ) : null}
          {clientLang || phase === "finished" || phase === "error" || phase === "docker-down" ? (
            <button
              className="ghost-button compact-button"
              onClick={() => void rerun()}
              title={clientLang ? "Reload the preview" : "Run the same source again without changes"}
              type="button"
            >
              {clientLang ? <Play size={14} /> : <RotateCcw size={14} />}
              {clientLang ? "Reload" : "Rerun same source"}
            </button>
          ) : null}
        </div>
      </header>

      {repairMessage ? (
        <div
          aria-live="polite"
          className={`runner-repair-message${repairState === "error" ? " error" : ""}`}
          role={repairState === "error" ? "alert" : "status"}
        >
          {repairMessage}
        </div>
      ) : null}

      <main className={`runner-body${showServerPreview ? " with-vnc" : ""}${sourceOpen ? " with-source" : ""}`}>
        {sourceOpen ? (
          <RunnerSourcePanel
            code={code}
            copied={sourceCopied}
            language={language}
            onCopy={() => void copySource()}
          />
        ) : null}
        {clientLang ? (
          <ClientPreview code={code} key={clientPreviewNonce} />
        ) : phase === "docker-down" ? (
          <DockerDownPanel reason={dockerReason} onRetry={rerun} />
        ) : showVncPane && vncUrl ? (
          <>
            <VncPane url={vncUrl} />
            <ServerLogDrawer
              errorMessage={errorMessage}
              isOpen={serverLogsOpen}
              onToggle={() => setServerLogsOpen((current) => !current)}
              output={output}
              outputLineCount={outputLineCount}
              outputRef={outputRef}
              phase={phase}
            />
          </>
        ) : showWebPane && webUrl ? (
          <>
            <WebPreviewPane url={webUrl} />
            <ServerLogDrawer
              errorMessage={errorMessage}
              isOpen={serverLogsOpen}
              onToggle={() => setServerLogsOpen((current) => !current)}
              output={output}
              outputLineCount={outputLineCount}
              outputRef={outputRef}
              phase={phase}
            />
          </>
        ) : (
          <ServerOutputPanel
            errorMessage={errorMessage}
            output={output}
            outputRef={outputRef}
            phase={phase}
          />
        )}
      </main>

      <footer className="runner-footer">
        <span className="runner-meta">
          {runId
            ? `Run ${runId.slice(0, 8)}`
            : clientLang
              ? "Client sandbox"
              : phase === "preparing"
                ? `Offline runtime ${Math.round((runtimeStatus?.progress ?? 0) * 100)}%`
                : "Awaiting Docker"}
        </span>
        {durationMs != null ? <span className="runner-meta">{(durationMs / 1000).toFixed(2)}s</span> : null}
        {exitCode != null ? <span className="runner-meta">exit {exitCode}</span> : null}
      </footer>
    </div>
  );
}

function RunnerSourcePanel({
  code,
  copied,
  language,
  onCopy,
}: {
  code: string;
  copied: boolean;
  language: string;
  onCopy: () => void;
}) {
  return (
    <aside className="runner-source-panel" aria-label="Run source">
      <div className="runner-source-header">
        <span className="runner-source-title">
          <Code2 size={15} />
          Source
        </span>
        <span className="runner-source-language">{language}</span>
        <button className="ghost-button compact-button runner-source-copy" onClick={onCopy} type="button">
          {copied ? <Check size={14} /> : <Copy size={14} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="runner-source-code">
        <code>{code}</code>
      </pre>
    </aside>
  );
}

function getSafeCurrentWindow() {
  try {
    return getCurrentWindow();
  } catch {
    return null;
  }
}

const BENIGN_RUNNER_DIAGNOSTIC = /^\[atlas-runner\]\s+(?:trusted preparation|secure offline python gui runtime|using prepared offline python gui runtime|gui ready on port)/i;

export function buildRunnerRepairDiagnostics({
  errorMessage,
  exitCode,
  output,
}: {
  errorMessage: string | null;
  exitCode: number | null;
  output: OutputLine[];
}): string {
  const lines = output
    .flatMap((entry) => entry.text.replace(/\r\n?/g, "\n").split("\n"))
    .map((line) => line.trimEnd())
    .filter((line) => line.trim() && !BENIGN_RUNNER_DIAGNOSTIC.test(line.trim()));
  const diagnostics: string[] = [];
  if (exitCode !== null) {
    diagnostics.push(`Process exited with code ${exitCode}.`);
  }
  if (errorMessage?.trim()) {
    diagnostics.push(errorMessage.trim());
  }
  diagnostics.push(...lines);
  return diagnostics.join("\n").trim() || "The process failed without diagnostic output.";
}

async function waitForRunnerRuntime(
  initial: RunnerRuntimeStatus,
  onUpdate: (runtime: RunnerRuntimeStatus) => void,
  isCurrent: () => boolean,
): Promise<RunnerRuntimeStatus | null> {
  let runtime = initial;
  onUpdate(runtime);
  while (runtime.state === "preparing" && isCurrent()) {
    await new Promise<void>((resolve) => {
      window.setTimeout(resolve, RUNNER_RUNTIME_POLL_INTERVAL_MS);
    });
    if (!isCurrent()) {
      return null;
    }
    runtime = await getPythonGuiRuntimeStatus();
    onUpdate(runtime);
  }
  if (!isCurrent()) {
    return null;
  }
  if (runtime.state !== "ready") {
    throw new Error(
      runtime.error ??
        "The secure offline Python GUI runtime is not ready. Check Docker connectivity and retry.",
    );
  }
  return runtime;
}

function runtimeLogLines(runtime: RunnerRuntimeStatus): OutputLine[] {
  const trustBoundary =
    "[atlas-runner] trusted preparation downloads only pinned Atlas dependencies; submitted code is not mounted or executed";
  const lines = runtime.log_tail.length > 0 ? runtime.log_tail : [runtime.message];
  return [trustBoundary, ...lines].map((text) => ({ stream: "stdout", text }));
}

function RunnerStatusBadge({ phase, exitCode }: { phase: Phase; exitCode: number | null }) {
  if (phase === "running") {
    return <span className="runner-status running">Running</span>;
  }
  if (phase === "loading") {
    return <span className="runner-status pending">Starting...</span>;
  }
  if (phase === "preparing") {
    return <span className="runner-status pending">Preparing runtime...</span>;
  }
  if (phase === "finished") {
    const ok = exitCode === 0;
    return <span className={`runner-status ${ok ? "ok" : "fail"}`}>{ok ? "Done" : `Exit ${exitCode}`}</span>;
  }
  if (phase === "error") {
    return <span className="runner-status fail">Error</span>;
  }
  if (phase === "docker-down") {
    return <span className="runner-status fail">Docker down</span>;
  }
  return <span className="runner-status pending">Ready</span>;
}

function runnerActivityLabel({
  clientLang,
  output,
  phase,
  runtimeStatus,
  vncReady,
  vncUrl,
  webUrl,
}: {
  clientLang: boolean;
  output: OutputLine[];
  phase: Phase;
  runtimeStatus: RunnerRuntimeStatus | null;
  vncReady: boolean;
  vncUrl: string | null;
  webUrl: string | null;
}) {
  if (clientLang) {
    return phase === "idle" ? "Client preview ready" : null;
  }
  if (phase === "loading") {
    return "Checking runner...";
  }
  if (phase === "preparing") {
    const progress = runtimeStatus ? Math.round(runtimeStatus.progress * 100) : 0;
    return `Building secure offline GUI runtime... ${progress}%`;
  }
  if (phase !== "running") {
    return null;
  }

  const latest = output.length ? output[output.length - 1].text.toLowerCase() : "";
  if (latest.includes("installing system dependencies")) {
    return vncUrl ? "Installing GUI dependencies..." : "Installing system dependencies...";
  }
  if (latest.includes("installing python packages")) {
    return "Installing Python packages...";
  }
  if (latest.includes("installing:") || latest.includes("resolving modules")) {
    return "Resolving dependencies...";
  }
  if (latest.includes("gui ready on port")) {
    return vncReady ? "Opening GUI preview..." : "Starting virtual display...";
  }
  if (latest.includes("web preview will use")) {
    return "Starting web preview...";
  }
  if (vncUrl) {
    return vncReady ? "Running GUI app" : "Starting virtual display...";
  }
  if (webUrl) {
    return "Running web app";
  }
  return "Running code...";
}

function ServerLogDrawer({
  errorMessage,
  isOpen,
  onToggle,
  output,
  outputLineCount,
  outputRef,
  phase,
}: {
  errorMessage: string | null;
  isOpen: boolean;
  onToggle: () => void;
  output: OutputLine[];
  outputLineCount: number;
  outputRef: React.MutableRefObject<HTMLDivElement | null>;
  phase: Phase;
}) {
  if (!isOpen) {
    return (
      <aside className="runner-log-drawer collapsed" aria-label="Runner logs">
        <button
          aria-expanded={false}
          className="runner-log-rail-button"
          onClick={onToggle}
          title="Show runner logs"
          type="button"
        >
          <PanelRightOpen size={15} />
          <span>Logs</span>
          {outputLineCount > 0 ? <span className="runner-log-count">{outputLineCount}</span> : null}
        </button>
      </aside>
    );
  }

  return (
    <aside className="runner-log-drawer open" aria-label="Runner logs">
      <div className="runner-log-header">
        <span className="runner-log-title">
          <Terminal size={15} /> Logs
        </span>
        <span className="runner-log-count">{outputLineCount}</span>
        <button
          aria-expanded={true}
          className="runner-log-close-button"
          onClick={onToggle}
          title="Hide runner logs"
          type="button"
        >
          <PanelRightClose size={15} />
        </button>
      </div>
      <ServerOutputPanel errorMessage={errorMessage} output={output} outputRef={outputRef} phase={phase} />
    </aside>
  );
}

function ServerOutputPanel({
  errorMessage,
  output,
  outputRef,
  phase,
}: {
  errorMessage: string | null;
  output: OutputLine[];
  outputRef: React.MutableRefObject<HTMLDivElement | null>;
  phase: Phase;
}) {
  return (
    <div className="runner-output" ref={outputRef}>
      {errorMessage ? <div className="runner-line stderr">{errorMessage}</div> : null}
      {output.length === 0 && phase !== "error" ? (
        <div className="runner-placeholder">{phase === "running" ? "Waiting for output..." : "No output."}</div>
      ) : null}
      {output.map((line, idx) => (
        <div className={`runner-line ${line.stream}`} key={idx}>
          {line.text.replace(/\r$/, "")}
        </div>
      ))}
    </div>
  );
}

function DockerDownPanel({ reason, onRetry }: { reason: string; onRetry: () => void }) {
  return (
    <div className="runner-center">
      <h2>Docker Desktop isn't running</h2>
      <p>{reason}</p>
      <button className="primary-button" onClick={() => onRetry()} type="button">
        Retry
      </button>
    </div>
  );
}

function ClientPreview({ code }: { code: string }) {
  const frameRef = useRef<HTMLIFrameElement | null>(null);
  const channel = useMemo(() => {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
    return `preview-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }, []);
  const [consoleLines, setConsoleLines] = useState<ClientPreviewLine[]>([]);
  const previewDocument = useMemo(() => buildClientPreviewDocument(code, channel), [channel, code]);

  useEffect(() => {
    setConsoleLines([]);
  }, [previewDocument]);

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.source !== frameRef.current?.contentWindow) {
        return;
      }
      const data = event.data as Partial<ClientPreviewEvent> | null;
      if (!data || data.source !== CLIENT_PREVIEW_MESSAGE_SOURCE || data.channel !== channel) {
        return;
      }
      if (data.type === "console" || data.type === "error") {
        const text = String(data.text ?? "").slice(0, MAX_CLIENT_PREVIEW_CONSOLE_CHARS).trim();
        if (!text) {
          return;
        }
        const requestedLevel = String(data.level ?? "log");
        const level =
          data.type === "error"
            ? "error"
            : ["log", "warn", "error"].includes(requestedLevel)
              ? (requestedLevel as ClientPreviewConsoleLevel)
              : "log";
        setConsoleLines((current) => [...current.slice(-79), { level, text }]);
      }
    };
    window.addEventListener("message", handleMessage);
    return () => {
      window.removeEventListener("message", handleMessage);
    };
  }, [channel]);

  return (
    <div className="runner-client-preview">
      <iframe
        className="runner-iframe"
        onLoad={() => {
          if (typeof navigator !== "undefined" && navigator.userAgent.toLowerCase().includes("jsdom")) {
            return;
          }
          try {
            frameRef.current?.contentWindow?.focus();
          } catch {
            // Some test/browser contexts expose focus but do not implement it.
          }
        }}
        ref={frameRef}
        sandbox={CLIENT_PREVIEW_SANDBOX}
        srcDoc={previewDocument}
        title="Atlas runner preview"
      />
      {consoleLines.length > 0 ? (
        <div className="runner-client-console" role="log">
          {consoleLines.map((line, index) => (
            <div className={`runner-client-console-line ${line.level}`} key={`${index}-${line.text}`}>
              <span className="runner-client-console-level">{line.level}</span>
              <span>{line.text}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function VncPane({ url }: { url: string }) {
  return (
    <RunnerUrlPreview
      background="dark"
      loadingLabel="Starting GUI..."
      title="Atlas GUI preview"
      url={url}
    />
  );
}

function WebPreviewPane({ url }: { url: string }) {
  return (
    <RunnerUrlPreview
      background="light"
      loadingLabel="Starting web preview..."
      title="Atlas web preview"
      url={url}
    />
  );
}

function RunnerUrlPreview({
  background,
  loadingLabel,
  title,
  url,
}: {
  background: "dark" | "light";
  loadingLabel: string;
  title: string;
  url: string;
}) {
  const [ready, setReady] = useState(false);
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    setReady(false);
    setSrc(null);
    let cancelled = false;
    let retryTimer: number | null = null;
    let probeController: AbortController | null = null;
    const start = Date.now();
    const deadlineMs = 45_000;
    const attempt = async () => {
      probeController?.abort();
      const controller = new AbortController();
      probeController = controller;
      const probeTimer = window.setTimeout(
        () => controller.abort(),
        Math.min(
          RUNNER_PREVIEW_PROBE_TIMEOUT_MS,
          Math.max(1, deadlineMs - (Date.now() - start)),
        ),
      );
      try {
        await fetch(url, {
          method: "GET",
          mode: "no-cors",
          signal: controller.signal,
        });
        if (!cancelled) {
          setSrc(url);
          setReady(true);
        }
      } catch {
        if (!cancelled && Date.now() - start < deadlineMs) {
          retryTimer = window.setTimeout(attempt, 500);
        } else if (!cancelled) {
          setSrc(url);
          setReady(true);
        }
      } finally {
        window.clearTimeout(probeTimer);
        if (probeController === controller) {
          probeController = null;
        }
      }
    };
    void attempt();
    return () => {
      cancelled = true;
      probeController?.abort();
      probeController = null;
      if (retryTimer !== null) {
        window.clearTimeout(retryTimer);
      }
    };
  }, [url]);

  return (
    <div className={`runner-vnc ${background === "light" ? "runner-web-preview" : ""}`}>
      {ready && src ? (
        <iframe
          className="runner-vnc-frame"
          referrerPolicy="no-referrer"
          sandbox={SERVER_PREVIEW_SANDBOX}
          src={src}
          title={title}
        />
      ) : (
        <div className="runner-vnc-placeholder">{loadingLabel}</div>
      )}
    </div>
  );
}
