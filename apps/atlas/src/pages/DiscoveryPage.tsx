import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  Copy,
  Cpu,
  Download,
  HardDrive,
  Info,
  LoaderCircle,
  RefreshCcw,
  Server,
  X,
  Zap,
} from "lucide-react";
import {
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  cancelModelPull,
  getDiscovery,
  listModelPulls,
  startModelPull,
  type DiscoveryReport,
  type ModelPull,
} from "../lib/api";
import {
  discoveryStatusLabel,
  discoveryStatusTone,
  formatDiscoveryFitLabel,
  formatDiscoveryMemory,
  formatGpuMemoryLabel,
  formatGpuSourceLabel,
  selectPrimaryGpu,
} from "../lib/discoveryUi";

type DiscoveryFilter = "needs-pull" | "installed" | "all";

const FILTER_OPTIONS: Array<{ key: DiscoveryFilter; label: string }> = [
  { key: "needs-pull", label: "Pull candidates" },
  { key: "installed", label: "Installed" },
  { key: "all", label: "All" },
];

export function DiscoveryPage() {
  const queryClient = useQueryClient();
  const [copiedCommand, setCopiedCommand] = useState("");
  const [activeFilter, setActiveFilter] = useState<DiscoveryFilter>("needs-pull");
  const refreshedCompletedPulls = useRef(new Set<string>());
  const { data, isPending, isFetching, isError, error } = useQuery({
    queryKey: ["discovery"],
    queryFn: getDiscovery,
    staleTime: 10000,
    retry: 1,
    refetchOnWindowFocus: false,
  });
  const { data: modelPulls = [] } = useQuery({
    queryKey: ["model-pulls"],
    queryFn: listModelPulls,
    refetchInterval: (query) => {
      const pulls = (query.state.data as ModelPull[] | undefined) ?? [];
      return pulls.some((pull) => pull.status === "queued" || pull.status === "pulling")
        ? 750
        : false;
    },
    retry: 1,
    refetchOnWindowFocus: false,
  });
  const startPullMutation = useMutation({
    mutationFn: startModelPull,
    onSuccess: (pull) => {
      queryClient.setQueryData<ModelPull[]>(["model-pulls"], (current = []) => [
        pull,
        ...current.filter((item) => item.pull_id !== pull.pull_id),
      ]);
    },
  });
  const cancelPullMutation = useMutation({
    mutationFn: cancelModelPull,
    onSuccess: (pull) => {
      queryClient.setQueryData<ModelPull[]>(["model-pulls"], (current = []) =>
        current.map((item) => (item.pull_id === pull.pull_id ? pull : item)),
      );
    },
  });

  const recommendedModels = data?.recommended_models ?? [];
  const primaryGpu = data ? selectPrimaryGpu(data.system.gpus) : null;
  const nextStep = data ? selectNextStep(recommendedModels) : null;
  const providerLabel = data?.atlas.provider_label || "Ollama";
  const supportsManagedPulls = (data?.atlas.provider || "ollama") === "ollama";
  const providerOnline = Boolean(data?.atlas.provider_online ?? data?.atlas.ollama_online);
  const installedCount = recommendedModels.filter((item) => item.installed).length;
  const needsPullCount = recommendedModels.length - installedCount;
  const discoveryNotes = useMemo(() => {
    if (!data) {
      return [];
    }
    return [...data.atlas.notes, ...data.system.detection.notes]
      .filter((note, index, notes) => Boolean(note) && notes.indexOf(note) === index)
      .slice(0, 4);
  }, [data]);
  const sortedRecommendations = useMemo(
    () => sortRecommendations(recommendedModels, nextStep?.name),
    [recommendedModels, nextStep?.name],
  );
  const filteredRecommendations = useMemo(
    () => sortedRecommendations.filter((item) => matchesDiscoveryFilter(item, activeFilter)),
    [activeFilter, sortedRecommendations],
  );
  const pullsByModel = useMemo(
    () =>
      new Map(
        modelPulls.map((pull) => [pull.model.toLocaleLowerCase(), pull] as const),
      ),
    [modelPulls],
  );

  useEffect(() => {
    let shouldRefresh = false;
    for (const pull of modelPulls) {
      if (
        pull.status === "completed" &&
        !refreshedCompletedPulls.current.has(pull.pull_id)
      ) {
        refreshedCompletedPulls.current.add(pull.pull_id);
        shouldRefresh = true;
      }
    }
    if (shouldRefresh) {
      void Promise.all([
        queryClient.invalidateQueries({ queryKey: ["models"] }),
        queryClient.invalidateQueries({ queryKey: ["discovery"] }),
      ]);
    }
  }, [modelPulls, queryClient]);

  const refreshDiscovery = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["status"] }),
      queryClient.invalidateQueries({ queryKey: ["models"] }),
      queryClient.invalidateQueries({ queryKey: ["discovery"] }),
    ]);
  };

  const copyCommand = async (command: string) => {
    try {
      await navigator.clipboard.writeText(command);
      setCopiedCommand(command);
      window.setTimeout(() => {
        setCopiedCommand((current) => (current === command ? "" : current));
      }, 1400);
    } catch {
      setCopiedCommand("");
    }
  };

  const handleFilterKeyDown = (event: ReactKeyboardEvent<HTMLButtonElement>, index: number) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      return;
    }
    event.preventDefault();
    const nextIndex =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? FILTER_OPTIONS.length - 1
          : event.key === "ArrowRight"
            ? (index + 1) % FILTER_OPTIONS.length
            : (index - 1 + FILTER_OPTIONS.length) % FILTER_OPTIONS.length;
    const nextFilter = FILTER_OPTIONS[nextIndex];
    setActiveFilter(nextFilter.key);
    window.requestAnimationFrame(() => {
      document.getElementById(`discovery-filter-${nextFilter.key}`)?.focus();
    });
  };

  return (
    <section className="page-shell discovery-page">
      <div className="workspace-header discovery-page-header">
        <div className="workspace-header-copy">
          <h1>Discovery</h1>
          <p className="workspace-header-summary">Match local models to this machine and Atlas workload.</p>
        </div>
        <div className="workspace-header-controls discovery-header-controls">
          <button
            aria-label="Refresh discovery"
            className="ghost-button icon-button discovery-icon-action"
            onClick={() => void refreshDiscovery()}
            title={isFetching ? "Refreshing" : "Refresh"}
            type="button"
          >
            <RefreshCcw size={15} />
          </button>
        </div>
      </div>

      {isError ? (
        <div className="error-banner" role="alert">
          Discovery is unavailable right now.{" "}
          {error instanceof Error ? error.message : "Atlas could not load the discovery report."}
        </div>
      ) : null}

      {!data && isPending ? (
        <div aria-live="polite" className="discovery-loading-line" role="status">
          <span className="status-pill starting">
            <span className="status-dot" />
            Loading
          </span>
          <p>Checking the local provider, local models, and hardware.</p>
        </div>
      ) : null}

      {data ? (
        <div className="discovery-stack">
          <section
            aria-label="Machine summary"
            className={`discovery-overview discovery-overview-${discoveryStatusTone(data.atlas.status)}`}
          >
            <DiscoveryMetric
              detail={providerOnline ? `${providerLabel} reachable` : `${providerLabel} unavailable`}
              icon={<Server size={16} />}
              label="Status"
              tone={discoveryStatusTone(data.atlas.status)}
              value={discoveryStatusLabel(data.atlas.status)}
            />
            <DiscoveryMetric
              detail={primaryGpu ? formatGpuSourceLabel(primaryGpu) : "No dedicated GPU reported"}
              icon={<Zap size={16} />}
              label="GPU"
              value={primaryGpu ? `${shortGpuName(primaryGpu.name)} · ${formatGpuMemoryLabel(primaryGpu)}` : "CPU only"}
            />
            <DiscoveryMetric
              detail={data.system.os}
              icon={<HardDrive size={16} />}
              label="RAM"
              value={formatDiscoveryMemory(data.system.memory.total_gb)}
            />
            <DiscoveryMetric
              detail={formatCpuDetail(data.system.cpu)}
              icon={<Cpu size={16} />}
              label="CPU"
              value={shortCpuName(data.system.cpu.model)}
            />
            <DiscoveryMetric
              detail={`${needsPullCount} pull candidates`}
              icon={<Download size={16} />}
              label="Models"
              value={`${installedCount} installed`}
            />
          </section>

          <section className="discovery-recommendation" aria-label="Primary recommendation">
            {nextStep ? (
              <>
                <div className="discovery-recommendation-copy">
                  <p className="workspace-section-label">{nextStep.installed ? "Ready model" : "Recommended next"}</p>
                  <h2>{nextStep.name}</h2>
                  <p>{nextStep.reason}</p>
                  <div className="discovery-chip-line" aria-label="Recommendation details">
                    <span className={`discovery-fit-label ${fitTone(nextStep.fit)}`}>
                      {formatDiscoveryFitLabel(nextStep.fit)}
                    </span>
                    <span>{nextStep.runtime}</span>
                    <span>{useCaseLabel(nextStep.use_case)}</span>
                    {nextStep.supports_images ? <span>Vision</span> : null}
                  </div>
                </div>

                <div className="discovery-recommendation-action">
                  {nextStep.installed ? (
                    <span className="discovery-ready-text">
                      <Check size={14} />
                      Installed
                    </span>
                  ) : (
                    <div className="discovery-command-block">
                      <span>
                        {formatPullStatus(
                          pullsByModel.get(nextStep.name.toLocaleLowerCase()),
                        )}
                      </span>
                      <PullProgress
                        command={nextStep.pull_command}
                        pull={pullsByModel.get(nextStep.name.toLocaleLowerCase())}
                      />
                      <PullButtons
                        canPull={supportsManagedPulls}
                        copied={copiedCommand === nextStep.pull_command}
                        isCancelling={
                          cancelPullMutation.isPending &&
                          cancelPullMutation.variables ===
                            pullsByModel.get(nextStep.name.toLocaleLowerCase())?.pull_id
                        }
                        isStarting={
                          startPullMutation.isPending &&
                          startPullMutation.variables === nextStep.name
                        }
                        item={nextStep}
                        onCancel={(pullId) => cancelPullMutation.mutate(pullId)}
                        onCopy={copyCommand}
                        onPull={(model) => startPullMutation.mutate(model)}
                        pull={pullsByModel.get(nextStep.name.toLocaleLowerCase())}
                      />
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="discovery-recommendation-copy">
                <p className="workspace-section-label">Recommendation</p>
                <h2>No model recommendation</h2>
                <p>{data.atlas.summary}</p>
              </div>
            )}
          </section>

          {discoveryNotes.length ? (
            <section className="discovery-notes-strip" aria-label="Discovery notes">
              <Info size={15} />
              <div>
                {discoveryNotes.map((note) => (
                  <p key={note}>{note}</p>
                ))}
              </div>
            </section>
          ) : null}

          <section className="discovery-model-picker" aria-label="Model recommendations">
            <div className="discovery-picker-head">
              <div>
                <p className="workspace-section-label">Models</p>
                <h2>{filterHeading(activeFilter)}</h2>
              </div>
              <span>{filteredRecommendations.length}</span>
            </div>

            <div className="discovery-filter-tabs" role="tablist" aria-label="Recommendation filters">
              {FILTER_OPTIONS.map((filter, index) => {
                const count = recommendedModels.filter((item) =>
                  matchesDiscoveryFilter(item, filter.key),
                ).length;
                return (
                  <button
                    aria-controls="discovery-model-results"
                    aria-selected={activeFilter === filter.key}
                    className={`discovery-filter-tab${activeFilter === filter.key ? " active" : ""}`}
                    id={`discovery-filter-${filter.key}`}
                    key={filter.key}
                    onClick={() => setActiveFilter(filter.key)}
                    onKeyDown={(event) => handleFilterKeyDown(event, index)}
                    role="tab"
                    tabIndex={activeFilter === filter.key ? 0 : -1}
                    type="button"
                  >
                    <span>{filter.label}</span>
                    <small>{count}</small>
                  </button>
                );
              })}
            </div>

            {filteredRecommendations.length ? (
              <div
                aria-labelledby={`discovery-filter-${activeFilter}`}
                className="discovery-model-list"
                id="discovery-model-results"
                role="tabpanel"
              >
                <div className="discovery-model-table-head" aria-hidden="true">
                  <span>Model</span>
                  <span>Fit</span>
                  <span>Runtime</span>
                  <span>Use</span>
                  <span>Action</span>
                </div>
                {filteredRecommendations.map((item) => (
                  <RecommendationRow
                    canPull={supportsManagedPulls}
                    copiedCommand={copiedCommand}
                    item={item}
                    key={item.name}
                    onCancel={(pullId) => cancelPullMutation.mutate(pullId)}
                    onCopy={copyCommand}
                    onPull={(model) => startPullMutation.mutate(model)}
                    pendingCancelId={
                      cancelPullMutation.isPending ? cancelPullMutation.variables : undefined
                    }
                    pendingModel={
                      startPullMutation.isPending ? startPullMutation.variables : undefined
                    }
                    pull={pullsByModel.get(item.name.toLocaleLowerCase())}
                  />
                ))}
              </div>
            ) : (
              <div
                aria-labelledby={`discovery-filter-${activeFilter}`}
                className="discovery-empty-state"
                id="discovery-model-results"
                role="tabpanel"
              >
                <strong>No models in this view</strong>
              </div>
            )}
          </section>
        </div>
      ) : null}
    </section>
  );
}

function DiscoveryMetric({
  detail,
  icon,
  label,
  tone,
  value,
}: {
  detail: string;
  icon: ReactNode;
  label: string;
  tone?: "online" | "warning" | "offline" | "muted";
  value: string;
}) {
  return (
    <div className={`discovery-metric${tone ? ` discovery-metric-${tone}` : ""}`}>
      <span className="discovery-metric-icon">{icon}</span>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <p>{detail}</p>
      </div>
    </div>
  );
}

function RecommendationRow({
  item,
  canPull,
  copiedCommand,
  onCancel,
  onCopy,
  onPull,
  pendingCancelId,
  pendingModel,
  pull,
}: {
  item: DiscoveryReport["recommended_models"][number];
  canPull: boolean;
  copiedCommand: string;
  onCancel: (pullId: string) => void;
  onCopy: (command: string) => Promise<void>;
  onPull: (model: string) => void;
  pendingCancelId?: string;
  pendingModel?: string;
  pull?: ModelPull;
}) {
  return (
    <article className="discovery-model-row">
      <div className="discovery-model-name">
        <h3>
          {item.name}
          {item.installed ? (
            <span title="Installed">
              <Check size={13} />
            </span>
          ) : null}
        </h3>
        <p>{item.title}</p>
        <small>{item.reason}</small>
      </div>

      <span className={`discovery-fit-label ${fitTone(item.fit)}`}>
        {formatDiscoveryFitLabel(item.fit)}
      </span>

      <span className="discovery-model-runtime">{item.runtime}</span>

      <div className="discovery-model-caps">
        <span>{useCaseLabel(item.use_case)}</span>
        {item.supports_images ? <span>Vision</span> : null}
      </div>

      <div className="discovery-model-action">
        {item.installed ? (
          <span className="discovery-ready-text">
            <Check size={14} />
            Installed
          </span>
        ) : (
          <div className="discovery-row-pull">
            <PullProgress command={item.pull_command} compact pull={pull} />
            <PullButtons
              canPull={canPull}
              compact
              copied={copiedCommand === item.pull_command}
              isCancelling={pendingCancelId === pull?.pull_id}
              isStarting={pendingModel === item.name}
              item={item}
              onCancel={onCancel}
              onCopy={onCopy}
              onPull={onPull}
              pull={pull}
            />
          </div>
        )}
      </div>
    </article>
  );
}

function PullProgress({
  command,
  compact = false,
  pull,
}: {
  command: string;
  compact?: boolean;
  pull?: ModelPull;
}) {
  if (pull?.status === "queued" || pull?.status === "pulling") {
    const percentage =
      pull.progress === null ? null : Math.round(Math.max(0, Math.min(1, pull.progress)) * 100);
    return (
      <div
        aria-label={`${pull.model} download progress`}
        className={`discovery-pull-progress${compact ? " compact" : ""}`}
        role="status"
      >
        <progress max={1} value={pull.progress ?? undefined} />
        <span>{percentage === null ? pull.detail : `${percentage}% · ${pull.detail}`}</span>
      </div>
    );
  }
  if (pull?.status === "failed") {
    return (
      <span className="discovery-pull-error" role="alert" title={pull.error || pull.detail}>
        {compact ? "Download failed" : pull.error || pull.detail}
      </span>
    );
  }
  if (pull?.status === "completed") {
    return (
      <span className="discovery-ready-text" role="status">
        <Check size={14} />
        Model ready
      </span>
    );
  }
  return compact ? null : <code>{command}</code>;
}

function PullButtons({
  canPull,
  compact = false,
  copied,
  isCancelling,
  isStarting,
  item,
  onCancel,
  onCopy,
  onPull,
  pull,
}: {
  canPull: boolean;
  compact?: boolean;
  copied: boolean;
  isCancelling: boolean;
  isStarting: boolean;
  item: DiscoveryReport["recommended_models"][number];
  onCancel: (pullId: string) => void;
  onCopy: (command: string) => Promise<void>;
  onPull: (model: string) => void;
  pull?: ModelPull;
}) {
  const active = pull?.status === "queued" || pull?.status === "pulling";
  if (pull?.status === "completed") {
    return null;
  }
  if (active && pull) {
    return (
      <button
        aria-label={`Cancel ${item.name} download`}
        className="ghost-button compact-button"
        disabled={isCancelling}
        onClick={() => onCancel(pull.pull_id)}
        type="button"
      >
        <X size={14} />
        {compact ? "" : isCancelling ? "Cancelling..." : "Cancel"}
      </button>
    );
  }
  return (
    <div className="inline-actions discovery-pull-actions">
      <button
        aria-label={`${pull?.status === "failed" ? "Retry" : "Download"} ${item.name}`}
        className={compact ? "ghost-button compact-button" : "primary-button compact-button"}
        disabled={!canPull || isStarting}
        onClick={() => onPull(item.name)}
        title={canPull ? undefined : "Managed downloads require Ollama."}
        type="button"
      >
        {isStarting ? <LoaderCircle className="spin" size={14} /> : <Download size={14} />}
        {isStarting ? "Starting..." : pull?.status === "failed" ? "Retry" : "Download"}
      </button>
      <button
        aria-label={`Copy ${item.name} pull command`}
        className="ghost-button icon-button discovery-copy-action"
        onClick={() => void onCopy(item.pull_command)}
        title={copied ? "Copied" : item.pull_command}
        type="button"
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
    </div>
  );
}

function formatPullStatus(pull?: ModelPull) {
  if (!pull || pull.status === "cancelled") {
    return "Ollama model";
  }
  if (pull.status === "queued") {
    return "Queued";
  }
  if (pull.status === "pulling") {
    return "Downloading";
  }
  if (pull.status === "completed") {
    return "Ready";
  }
  return "Download failed";
}

function matchesDiscoveryFilter(
  item: DiscoveryReport["recommended_models"][number],
  filter: DiscoveryFilter,
) {
  if (filter === "all") {
    return true;
  }
  if (filter === "installed") {
    return item.installed;
  }
  return !item.installed;
}

function sortRecommendations(
  models: DiscoveryReport["recommended_models"],
  primaryName: string | undefined,
) {
  return [...models].sort((left, right) => {
    const leftPrimary = left.name === primaryName ? 0 : 1;
    const rightPrimary = right.name === primaryName ? 0 : 1;
    if (leftPrimary !== rightPrimary) {
      return leftPrimary - rightPrimary;
    }

    const leftInstall = left.installed ? 1 : 0;
    const rightInstall = right.installed ? 1 : 0;
    if (leftInstall !== rightInstall) {
      return leftInstall - rightInstall;
    }

    const leftFit = fitRank(left.fit);
    const rightFit = fitRank(right.fit);
    if (leftFit !== rightFit) {
      return leftFit - rightFit;
    }

    return left.name.localeCompare(right.name);
  });
}

function selectNextStep(models: DiscoveryReport["recommended_models"]) {
  return (
    models.find(
      (item) =>
        !item.installed &&
        item.fit !== "too-large" &&
        item.fit !== "unavailable" &&
        item.atlas_role === "chat",
    ) ??
    models.find((item) => !item.installed && item.fit !== "too-large" && item.fit !== "unavailable") ??
    models.find((item) => !item.installed) ??
    models[0] ??
    null
  );
}

function filterHeading(filter: DiscoveryFilter) {
  if (filter === "installed") {
    return "Installed models";
  }
  if (filter === "all") {
    return "Catalog";
  }
  return "Pull candidates";
}

function useCaseLabel(value: DiscoveryReport["recommended_models"][number]["use_case"]) {
  if (value === "chat") {
    return "Chat";
  }
  if (value === "coding") {
    return "Coding";
  }
  if (value === "embedding") {
    return "Memory";
  }
  if (value === "vision") {
    return "Vision";
  }
  return "Reasoning";
}

function fitTone(fit: DiscoveryReport["recommended_models"][number]["fit"]) {
  if (fit === "good") {
    return "online";
  }
  if (fit === "tight") {
    return "starting";
  }
  if (fit === "cpu-only") {
    return "warning";
  }
  return "muted";
}

function fitRank(fit: DiscoveryReport["recommended_models"][number]["fit"]) {
  if (fit === "good") {
    return 0;
  }
  if (fit === "tight") {
    return 1;
  }
  if (fit === "cpu-only") {
    return 2;
  }
  if (fit === "too-large") {
    return 3;
  }
  return 4;
}

function shortGpuName(name: string) {
  return name.replace(/^NVIDIA\s+/i, "").replace(/\s+Laptop GPU$/i, "");
}

function shortCpuName(name: string | null) {
  if (!name) {
    return "Unknown CPU";
  }
  return name
    .replace(/\(R\)|\(TM\)|CPU|Processor/gi, "")
    .replace(/\s+@.+$/i, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function formatCpuDetail(cpu: DiscoveryReport["system"]["cpu"]) {
  if (typeof cpu.logical_cores === "number" && cpu.logical_cores > 0) {
    return `${cpu.logical_cores} logical cores`;
  }
  return "Core count unavailable";
}
