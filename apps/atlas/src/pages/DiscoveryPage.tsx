import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  Copy,
  Cpu,
  Download,
  HardDrive,
  Info,
  RefreshCcw,
  Server,
  Zap,
} from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";

import { getDiscovery, type DiscoveryReport } from "../lib/api";
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
  const { data, isPending, isFetching, isError, error } = useQuery({
    queryKey: ["discovery"],
    queryFn: getDiscovery,
    staleTime: 10000,
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const recommendedModels = data?.recommended_models ?? [];
  const primaryGpu = data ? selectPrimaryGpu(data.system.gpus) : null;
  const nextStep = data ? selectNextStep(recommendedModels) : null;
  const providerLabel = data?.atlas.provider_label || "Ollama";
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
        <div className="error-banner">
          Discovery is unavailable right now.{" "}
          {error instanceof Error ? error.message : "Atlas could not load the discovery report."}
        </div>
      ) : null}

      {!data && isPending ? (
        <div className="discovery-loading-line">
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
                      <span>Ollama pull</span>
                      <code>{nextStep.pull_command}</code>
                      <button
                        aria-label={`Copy ${nextStep.name} pull command`}
                        className="ghost-button icon-button discovery-copy-action"
                        onClick={() => void copyCommand(nextStep.pull_command)}
                        title={copiedCommand === nextStep.pull_command ? "Copied" : nextStep.pull_command}
                        type="button"
                      >
                        {copiedCommand === nextStep.pull_command ? <Check size={14} /> : <Copy size={14} />}
                      </button>
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
              {FILTER_OPTIONS.map((filter) => {
                const count = recommendedModels.filter((item) =>
                  matchesDiscoveryFilter(item, filter.key),
                ).length;
                return (
                  <button
                    aria-selected={activeFilter === filter.key}
                    className={`discovery-filter-tab${activeFilter === filter.key ? " active" : ""}`}
                    key={filter.key}
                    onClick={() => setActiveFilter(filter.key)}
                    role="tab"
                    type="button"
                  >
                    <span>{filter.label}</span>
                    <small>{count}</small>
                  </button>
                );
              })}
            </div>

            {filteredRecommendations.length ? (
              <div className="discovery-model-list">
                <div className="discovery-model-table-head" aria-hidden="true">
                  <span>Model</span>
                  <span>Fit</span>
                  <span>Runtime</span>
                  <span>Use</span>
                  <span>Action</span>
                </div>
                {filteredRecommendations.map((item) => (
                  <RecommendationRow
                    copiedCommand={copiedCommand}
                    item={item}
                    key={item.name}
                    onCopy={copyCommand}
                  />
                ))}
              </div>
            ) : (
              <div className="discovery-empty-state">
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
  copiedCommand,
  onCopy,
}: {
  item: DiscoveryReport["recommended_models"][number];
  copiedCommand: string;
  onCopy: (command: string) => Promise<void>;
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
          <button
            aria-label={`Copy ${item.name} pull command`}
            className="ghost-button compact-button discovery-row-copy-action"
            onClick={() => void onCopy(item.pull_command)}
            title={copiedCommand === item.pull_command ? "Copied" : item.pull_command}
            type="button"
          >
            {copiedCommand === item.pull_command ? <Check size={14} /> : <Copy size={14} />}
            Pull
          </button>
        )}
      </div>
    </article>
  );
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
