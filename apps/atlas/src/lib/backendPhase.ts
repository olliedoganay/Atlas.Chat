import { useEffect, useState } from "react";

export type BackendPhase = "online" | "starting" | "offline";

const BACKEND_STARTUP_GRACE_MS = 15000;

type BackendPhaseOptions = {
  hasStatus: boolean;
  isError: boolean;
  isPending: boolean;
  isFetching: boolean;
  bootStartedAt: number;
};

export function resolveBackendPhase({
  hasStatus,
  isError,
  isPending,
  isFetching,
  bootStartedAt,
  now,
}: BackendPhaseOptions & { now: number }): BackendPhase {
  const graceEndsAt = bootStartedAt + BACKEND_STARTUP_GRACE_MS;
  if (hasStatus && !isError) {
    return "online";
  }
  if (isPending || isFetching || now < graceEndsAt) {
    return "starting";
  }
  return "offline";
}

export function useBackendPhase({
  hasStatus,
  isError,
  isPending,
  isFetching,
  bootStartedAt,
}: BackendPhaseOptions): BackendPhase {
  const graceEndsAt = bootStartedAt + BACKEND_STARTUP_GRACE_MS;
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (hasStatus) {
      return;
    }
    const remaining = graceEndsAt - Date.now();
    if (remaining <= 0) {
      setNow(Date.now());
      return;
    }
    const timer = window.setTimeout(() => setNow(Date.now()), remaining + 50);
    return () => window.clearTimeout(timer);
  }, [graceEndsAt, hasStatus, isFetching, isPending]);

  return resolveBackendPhase({
    hasStatus,
    isError,
    isPending,
    isFetching,
    bootStartedAt,
    now,
  });
}
