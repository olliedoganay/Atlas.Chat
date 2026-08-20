import { useQueries, useQuery } from "@tanstack/react-query";
import { Activity, Clock, Database, Gauge, Server, ShieldCheck } from "lucide-react";
import { type ReactNode } from "react";
import { useNavigate } from "react-router-dom";

import { EmptyState } from "../components/ui/EmptyState";
import { getModels, getRun, getStatus, getThreads, getUsers } from "../lib/api";
import { displayThreadTitle } from "../lib/threadTitles";
import { useAtlasStore } from "../store/useAtlasStore";

export function AdvancedPage() {
  const navigate = useNavigate();
  const currentUserId = useAtlasStore((state) => state.currentUserId);
  const currentThreadId = useAtlasStore((state) => state.currentThreadId);

  const {
    data: users = [],
    isFetched: usersFetched,
    isError: usersError,
    error: usersQueryError,
  } = useQuery({
    queryKey: ["users"],
    queryFn: getUsers,
    staleTime: 5000,
  });
  const currentProfile = users.find((user) => user.user_id === currentUserId);
  const currentUserLocked = Boolean(currentProfile?.locked);
  const profileAccessConfirmed =
    Boolean(currentUserId) && usersFetched && Boolean(currentProfile) && !currentUserLocked;

  const {
    data: status,
    isPending: statusPending,
    isError: statusError,
    error: statusQueryError,
  } = useQuery({
    queryKey: ["status"],
    queryFn: getStatus,
    staleTime: 5000,
  });
  const {
    data: models,
    isError: modelsError,
    error: modelsQueryError,
  } = useQuery({
    queryKey: ["models"],
    queryFn: getModels,
    staleTime: 10000,
  });
  const {
    data: threads = [],
    isError: threadsError,
    error: threadsQueryError,
  } = useQuery({
    queryKey: ["threads", currentUserId],
    queryFn: () => getThreads(currentUserId),
    enabled: profileAccessConfirmed,
    staleTime: 2000,
  });

  const visibleThreads = profileAccessConfirmed ? threads : [];

  const recentRunIds = visibleThreads
    .map((thread) => thread.last_run_id)
    .filter((value, index, items): value is string => Boolean(value) && items.indexOf(value) === index)
    .slice(0, 6);
  const recentRunQueries = useQueries({
    queries: recentRunIds.map((runId) => ({
      queryKey: ["run", runId],
      queryFn: () => getRun(runId),
      staleTime: 2000,
    })),
  });
  const recentRuns = recentRunQueries
    .map((query) => query.data)
    .filter((item): item is NonNullable<(typeof recentRunQueries)[number]["data"]> => Boolean(item));
  const currentThread = visibleThreads.find((thread) => thread.thread_id === currentThreadId) ?? null;
  const currentRun = recentRuns.find((run) => run.thread_id === currentThreadId) ?? recentRuns[0] ?? null;
  const storageHardened =
    Boolean(status?.security.sqlite_encrypted_at_rest) &&
    Boolean(status?.security.vector_store_encrypted_at_rest);
  const backendOnline = Boolean(status) && !statusError;
  const providerLabel = models?.provider_label || status?.chat_provider_label || "Local provider";
  const providerOnline = !modelsError && Boolean(models?.provider_online ?? models?.ollama_online);
  const hasChatModels = !modelsError && Boolean(models?.has_chat_models ?? models?.has_local_models);
  const runQueryError = recentRunQueries.find((query) => query.isError)?.error;
  const diagnosticsError =
    statusError || modelsError || usersError || (profileAccessConfirmed && (threadsError || runQueryError))
      ? statusQueryError ?? modelsQueryError ?? usersQueryError ?? threadsQueryError ?? runQueryError
      : null;

  return (
    <section className="advanced-page">
      <div className="workspace-header advanced-page-header">
        <div className="workspace-header-copy">
          <h1>Diagnostics</h1>
          <p className="workspace-header-summary">Runtime health, storage posture, and recent model timing.</p>
        </div>
      </div>

      {diagnosticsError ? (
        <div className="error-banner" role="alert">
          {diagnosticsError instanceof Error
            ? diagnosticsError.message
            : "Some local diagnostics could not be loaded."}
        </div>
      ) : null}

      <section className="advanced-health-grid" aria-label="Runtime health">
        <AdvancedHealthCard
          detail={
            (backendOnline ? status?.backend : undefined) ||
            (statusPending ? "Waiting for the local runtime" : "Backend has not reported")
          }
          icon={<Server size={16} />}
          label="Backend"
          tone={backendOnline ? "online" : statusPending ? "warning" : "offline"}
          value={backendOnline ? "Online" : statusPending ? "Checking" : "Offline"}
        />
        <AdvancedHealthCard
          detail={providerOnline ? `${providerLabel} reachable` : `Start ${providerLabel} to chat locally`}
          icon={<Gauge size={16} />}
          label="Provider"
          tone={providerOnline ? "online" : "offline"}
          value={providerOnline ? "Connected" : "Unavailable"}
        />
        <AdvancedHealthCard
          detail={
            modelsError
              ? "Model inventory unavailable"
              : models?.catalog_source
                ? `Live ${providerLabel} inventory`
                : "Fallback catalog"
          }
          icon={<Database size={16} />}
          label="Models"
          tone={hasChatModels ? "online" : "warning"}
          value={modelsError ? "Unavailable" : `${models?.models.length ?? 0} installed`}
        />
        <AdvancedHealthCard
          detail={status?.security.profile_key_protection || "Profile key status unknown"}
          icon={<ShieldCheck size={16} />}
          label="Storage"
          tone={storageHardened ? "online" : "warning"}
          value={storageHardened ? "Encrypted" : "Partial"}
        />
      </section>

      {profileAccessConfirmed ? (
        <>
      <section className="advanced-thread-panel" aria-label="Current thread diagnostics">
        <div className="advanced-thread-copy">
          <p className="workspace-section-label">Current thread</p>
          <h2>{displayThreadTitle(currentThread, currentThreadId, "No active thread")}</h2>
          <p>{currentThread?.last_prompt ? truncateText(currentThread.last_prompt, 160) : "Open a thread to bind live run metrics here."}</p>
        </div>
        <dl className="advanced-thread-metrics">
          <div>
            <dt>Model</dt>
            <dd>{currentThread?.chat_model || "Not set"}</dd>
          </div>
          <div>
            <dt>Temperature</dt>
            <dd>{formatTemperature(currentThread?.temperature)}</dd>
          </div>
          <div>
            <dt>Last run</dt>
            <dd>{currentThread?.last_run_id ? shortRunId(currentThread.last_run_id) : "None"}</dd>
          </div>
          <div>
            <dt>Updated</dt>
            <dd>{formatDate(currentThread?.updated_at)}</dd>
          </div>
        </dl>
      </section>

      <div className="advanced-runs">
        <div className="advanced-runs-head">
          <div>
            <p className="workspace-section-label">Runs</p>
            <h2>Recent run metrics</h2>
          </div>
          {currentRun ? (
            <span className="muted-text">
              Latest: {displayThreadTitle(currentRun.thread_title, currentRun.thread_id)}
            </span>
          ) : null}
        </div>

        {recentRuns.length > 0 ? (
          <div className="advanced-run-list">
            {recentRuns.map((run) => (
              <article className="advanced-run-card" key={run.run_id}>
                <div className="advanced-run-head">
                  <div>
                    <strong>{displayThreadTitle(run.thread_title, run.thread_id)}</strong>
                    <p>{run.chat_model || "Local model"} · {run.mode}</p>
                  </div>
                  <span
                    className={`status-pill subtle ${run.status === "completed" ? "online" : run.status === "failed" ? "offline" : "muted"}`}
                  >
                    {run.status}
                  </span>
                </div>
                <dl className="advanced-metric-grid">
                  <div>
                    <dt>First token</dt>
                    <dd>{formatDuration(run.diagnostics?.first_token_latency_ms)}</dd>
                  </div>
                  <div>
                    <dt>Total</dt>
                    <dd>{formatDuration(run.diagnostics?.total_duration_ms)}</dd>
                  </div>
                  <div>
                    <dt>Tokens/sec</dt>
                    <dd>{formatTokensPerSecond(run.diagnostics?.output_tokens_per_second_estimate)}</dd>
                  </div>
                  <div>
                    <dt>Output tokens</dt>
                    <dd>{formatInteger(run.diagnostics?.output_tokens_estimate)}</dd>
                  </div>
                  <div>
                    <dt>Compaction gain</dt>
                    <dd>{formatInteger(run.diagnostics?.compaction_gain_tokens_estimate)}</dd>
                  </div>
                  <div>
                    <dt>Compactions</dt>
                    <dd>{formatInteger(run.diagnostics?.compaction_events_count)}</dd>
                  </div>
                </dl>
                <div className="advanced-run-foot">
                  <span>
                    <Clock size={13} />
                    {formatDate(run.started_at)}
                  </span>
                  <span>{truncateText(run.prompt || "No prompt", 112)}</span>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            icon={<Activity size={18} />}
            title="No diagnostics yet"
            description="Run a chat first. Atlas Chat will show timing, compaction gain, and output estimates here."
            actions={
              <button
                className="primary-button compact-button"
                onClick={() => navigate("/workspace")}
                type="button"
              >
                Open Workspace
              </button>
            }
          />
        )}
      </div>
        </>
      ) : (
        <section className="advanced-thread-panel" aria-label="Profile diagnostics unavailable">
          <EmptyState
            icon={<ShieldCheck size={18} />}
            title={currentUserLocked ? "Profile locked" : "Profile access required"}
            description={
              currentUserLocked
                ? "Unlock this profile in Settings before Atlas displays its thread or run details."
                : "Select an unlocked profile before Atlas displays private thread or run details."
            }
            actions={
              <button
                className="primary-button compact-button"
                onClick={() => navigate("/settings/profiles")}
                type="button"
              >
                Open profiles
              </button>
            }
          />
        </section>
      )}
    </section>
  );
}

function AdvancedHealthCard({
  detail,
  icon,
  label,
  tone,
  value,
}: {
  detail: string;
  icon: ReactNode;
  label: string;
  tone: "online" | "warning" | "offline";
  value: string;
}) {
  return (
    <article className={`advanced-health-card advanced-health-card-${tone}`}>
      <span className="advanced-health-icon">{icon}</span>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <p>{detail}</p>
      </div>
    </article>
  );
}

function formatTemperature(value: number | null | undefined) {
  if (value === undefined) {
    return "Not set";
  }
  if (value === null || Number.isNaN(value)) {
    return "Model setting";
  }
  return value.toFixed(1);
}

function formatDate(value?: string) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return "-";
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
  }).format(date);
}

function formatDuration(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  if (value < 1000) {
    return `${value} ms`;
  }
  return `${(value / 1000).toFixed(2)} s`;
}

function formatTokensPerSecond(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(2)} tok/s`;
}

function formatInteger(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value) || value <= 0) {
    return "-";
  }
  return value.toLocaleString();
}

function truncateText(value: string, maxLength: number) {
  const normalized = value.trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength).trimEnd()}...`;
}

function shortRunId(value: string) {
  return value.length > 8 ? value.slice(0, 8) : value;
}
