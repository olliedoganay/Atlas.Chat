import type { DiscoveryReport } from "./api";

export function discoveryStatusTone(status: DiscoveryReport["atlas"]["status"]) {
  if (status === "ready") {
    return "online";
  }
  if (status === "memory-degraded") {
    return "warning";
  }
  if (status === "runtime-unavailable") {
    return "offline";
  }
  return "warning";
}

export function discoveryStatusLabel(status: DiscoveryReport["atlas"]["status"]) {
  if (status === "ready") {
    return "Atlas ready";
  }
  if (status === "memory-degraded") {
    return "Memory degraded";
  }
  if (status === "runtime-unavailable") {
    return "Provider offline";
  }
  return "Chat blocked";
}

export function formatDiscoveryFitLabel(fit: DiscoveryReport["recommended_models"][number]["fit"]) {
  if (fit === "cpu-only") {
    return "CPU only";
  }
  if (fit === "too-large") {
    return "Too large";
  }
  if (fit === "unavailable") {
    return "Estimate unavailable";
  }
  return fit.charAt(0).toUpperCase() + fit.slice(1);
}

export function formatDiscoveryMemory(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value) || value <= 0) {
    return "Unknown";
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} GB`;
}

export function selectPrimaryGpu(gpus: DiscoveryReport["system"]["gpus"]) {
  if (!gpus.length) {
    return null;
  }
  const ranked = [...gpus].sort((left, right) => {
    const leftKind = left.kind === "dedicated" ? 0 : left.kind === "unknown" ? 1 : 2;
    const rightKind = right.kind === "dedicated" ? 0 : right.kind === "unknown" ? 1 : 2;
    if (leftKind !== rightKind) {
      return leftKind - rightKind;
    }
    return (right.memory_gb ?? -1) - (left.memory_gb ?? -1);
  });
  return ranked[0] ?? null;
}

export function formatGpuMemoryLabel(gpu: DiscoveryReport["system"]["gpus"][number] | null) {
  if (!gpu) {
    return "No discrete GPU detected";
  }
  if (gpu.kind === "integrated") {
    return "Integrated graphics";
  }
  if (typeof gpu.memory_gb === "number" && gpu.memory_gb > 0) {
    return `${formatDiscoveryMemory(gpu.memory_gb)} VRAM`;
  }
  return "VRAM unavailable";
}

export function formatGpuSourceLabel(gpu: DiscoveryReport["system"]["gpus"][number] | null) {
  if (!gpu) {
    return "Atlas will estimate fit from system RAM.";
  }
  if (gpu.kind === "integrated") {
    return "Shared system memory is not used as dedicated VRAM.";
  }
  if (gpu.memory_source === "nvidia-smi") {
    return "Measured from NVIDIA runtime.";
  }
  if (gpu.memory_source === "adapterram") {
    return "Estimated from Windows display adapter data.";
  }
  return "GPU memory estimate.";
}
