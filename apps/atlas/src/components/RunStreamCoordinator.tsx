import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { getRun, getThreadRuns, streamRun, type RunSummary } from "../lib/api";
import { selectRecoverableRun } from "../lib/runRecovery";
import { isUserCancelledRunError } from "../lib/runTermination";
import { useAtlasStore } from "../store/useAtlasStore";

export function RunStreamCoordinator({ canRecoverRun }: { canRecoverRun: boolean }) {
  const queryClient = useQueryClient();
  const currentUserId = useAtlasStore((state) => state.currentUserId);
  const currentThreadId = useAtlasStore((state) => state.currentThreadId);
  const currentRunId = useAtlasStore((state) => state.currentRunId);
  const currentRunMode = useAtlasStore((state) => state.currentRunMode);
  const activeRunUserId = useAtlasStore((state) => state.activeRunUserId);
  const activeRunThreadId = useAtlasStore((state) => state.activeRunThreadId);
  const isStreaming = useAtlasStore((state) => state.isStreaming);
  const recoverRun = useAtlasStore((state) => state.recoverRun);
  const appendThinking = useAtlasStore((state) => state.appendThinking);
  const appendToken = useAtlasStore((state) => state.appendToken);
  const setStage = useAtlasStore((state) => state.setStage);
  const completeRun = useAtlasStore((state) => state.completeRun);
  const cancelRun = useAtlasStore((state) => state.cancelRun);
  const failRun = useAtlasStore((state) => state.failRun);
  const showCompactionNotice = useAtlasStore((state) => state.showCompactionNotice);

  const teardownRef = useRef<(() => void) | null>(null);
  const attachedRunIdRef = useRef<string | null>(null);
  const tokenBufferRef = useRef("");
  const thinkingBufferRef = useRef("");
  const tokenFlushTimerRef = useRef<number | null>(null);
  const thinkingFlushTimerRef = useRef<number | null>(null);
  const attemptedRecoveryKeysRef = useRef(new Set<string>());
  const recoveryKey = JSON.stringify([currentUserId, currentThreadId]);
  const recoveryEnabled =
    canRecoverRun &&
    Boolean(currentUserId) &&
    Boolean(currentThreadId) &&
    !attemptedRecoveryKeysRef.current.has(recoveryKey) &&
    !currentRunId &&
    !isStreaming;
  const {
    data: recoveryRuns = [],
    isFetchedAfterMount: recoveryRunsFetchedAfterMount,
    isSuccess: recoveryRunsLoaded,
  } = useQuery({
    queryKey: ["thread-runs", currentUserId, currentThreadId],
    queryFn: () => getThreadRuns(currentThreadId, currentUserId),
    enabled: recoveryEnabled,
    retry: 1,
    refetchOnWindowFocus: false,
    staleTime: 0,
  });

  useEffect(() => {
    if (!recoveryEnabled || !recoveryRunsFetchedAfterMount || !recoveryRunsLoaded) {
      return;
    }
    attemptedRecoveryKeysRef.current.add(recoveryKey);
    const candidate = selectRecoverableRun(recoveryRuns, currentUserId, currentThreadId);
    if (candidate) {
      recoverRun(candidate);
    }
  }, [
    currentThreadId,
    currentUserId,
    recoverRun,
    recoveryKey,
    recoveryEnabled,
    recoveryRunsFetchedAfterMount,
    recoveryRuns,
    recoveryRunsLoaded,
  ]);

  const clearTokenFlushTimer = () => {
    if (tokenFlushTimerRef.current !== null) {
      window.clearTimeout(tokenFlushTimerRef.current);
      tokenFlushTimerRef.current = null;
    }
  };

  const clearThinkingFlushTimer = () => {
    if (thinkingFlushTimerRef.current !== null) {
      window.clearTimeout(thinkingFlushTimerRef.current);
      thinkingFlushTimerRef.current = null;
    }
  };

  const flushBufferedTokens = () => {
    if (!tokenBufferRef.current) {
      return;
    }
    appendToken(tokenBufferRef.current);
    tokenBufferRef.current = "";
  };

  const flushBufferedThinking = () => {
    if (!thinkingBufferRef.current) {
      return;
    }
    appendThinking(thinkingBufferRef.current);
    thinkingBufferRef.current = "";
  };

  const scheduleTokenFlush = () => {
    if (tokenFlushTimerRef.current !== null) {
      return;
    }
    tokenFlushTimerRef.current = window.setTimeout(() => {
      tokenFlushTimerRef.current = null;
      flushBufferedTokens();
    }, 32);
  };

  const scheduleThinkingFlush = () => {
    if (thinkingFlushTimerRef.current !== null) {
      return;
    }
    thinkingFlushTimerRef.current = window.setTimeout(() => {
      thinkingFlushTimerRef.current = null;
      flushBufferedThinking();
    }, 48);
  };

  const invalidateRunQueries = async (runId: string, userId: string, threadId: string) => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["threads", userId] }),
      queryClient.invalidateQueries({ queryKey: ["thread-history", userId, threadId] }),
      queryClient.invalidateQueries({ queryKey: ["thread-runs", userId, threadId] }),
      queryClient.invalidateQueries({ queryKey: ["thread-context", userId, threadId] }),
      queryClient.invalidateQueries({ queryKey: ["run", runId] }),
    ]);
  };

  const updateRunCompactionMetadata = (runId: string, payload: {
    compactedMessageCount: number;
    threadSummary: string;
    detectedContextWindow: number;
  }) => {
    queryClient.setQueryData<RunSummary | undefined>(["run", runId], (current) => {
      if (!current) {
        return current;
      }
      return {
        ...current,
        compacted_message_count: Math.max(
          Number(current.compacted_message_count ?? 0),
          payload.compactedMessageCount,
        ),
        thread_summary: payload.threadSummary || current.thread_summary || "",
        detected_context_window:
          payload.detectedContextWindow || Number(current.detected_context_window ?? 0),
      };
    });
  };

  const finalizeRunFromArtifact = async (runId: string, userId: string, threadId: string) => {
    await invalidateRunQueries(runId, userId, threadId);
    completeRun();
  };

  const recoverRunAfterStreamError = async (
    runId: string,
    userId: string,
    threadId: string,
    fallbackMessage: string,
  ) => {
    try {
      const artifact = await getRun(runId);
      if (artifact.status === "completed") {
        await finalizeRunFromArtifact(runId, userId, threadId);
        return;
      }
      if (artifact.status === "failed") {
        await invalidateRunQueries(runId, userId, threadId);
        if (isUserCancelledRunError(artifact.error)) {
          cancelRun();
        } else {
          failRun(artifact.error || fallbackMessage);
        }
        return;
      }
    } catch {
      // Fall back to the client-visible error state below.
    }
    failRun(fallbackMessage);
  };

  useEffect(() => {
    if (!currentRunId || !currentRunMode || !activeRunUserId || !activeRunThreadId || !isStreaming) {
      clearTokenFlushTimer();
      clearThinkingFlushTimer();
      teardownRef.current?.();
      teardownRef.current = null;
      attachedRunIdRef.current = null;
      return;
    }
    if (attachedRunIdRef.current === currentRunId) {
      return;
    }

    clearTokenFlushTimer();
    clearThinkingFlushTimer();
    teardownRef.current?.();

    attachedRunIdRef.current = currentRunId;
    teardownRef.current = streamRun(
      currentRunMode,
      currentRunId,
      (event) => {
        switch (event.type) {
          case "stage_changed":
            setStage(String(event.payload.stage ?? "running"));
            break;
          case "thinking_token":
            thinkingBufferRef.current += String(event.payload.text ?? "");
            scheduleThinkingFlush();
            break;
          case "token":
            tokenBufferRef.current += String(event.payload.text ?? "");
            scheduleTokenFlush();
            break;
          case "context_compacted": {
            const compactedMessageCount = Number(event.payload.compacted_message_count ?? 0);
            if (!Number.isFinite(compactedMessageCount) || compactedMessageCount <= 0) {
              break;
            }
            const newlyCompactedMessageCount = Number(event.payload.newly_compacted_message_count ?? 0);
            const threadSummary = String(event.payload.thread_summary ?? "");
            const detectedContextWindow = Number(event.payload.detected_context_window ?? 0);
            const compactionReason = String(event.payload.compaction_reason ?? "");
            const historyRepresentationTokensBeforeCompaction = Number(
              event.payload.history_representation_tokens_before_compaction ?? 0,
            );
            const historyRepresentationTokensAfterCompaction = Number(
              event.payload.history_representation_tokens_after_compaction ?? 0,
            );
            showCompactionNotice({
              runId: currentRunId,
              userId: activeRunUserId,
              threadId: activeRunThreadId,
              compactionReason,
              compactedMessageCount,
              newlyCompactedMessageCount: Number.isFinite(newlyCompactedMessageCount)
                ? Math.max(0, Math.trunc(newlyCompactedMessageCount))
                : 0,
              threadSummary,
              detectedContextWindow: Number.isFinite(detectedContextWindow)
                ? Math.max(0, Math.trunc(detectedContextWindow))
                : 0,
              historyRepresentationTokensBeforeCompaction: Number.isFinite(historyRepresentationTokensBeforeCompaction)
                ? Math.max(0, Math.trunc(historyRepresentationTokensBeforeCompaction))
                : 0,
              historyRepresentationTokensAfterCompaction: Number.isFinite(historyRepresentationTokensAfterCompaction)
                ? Math.max(0, Math.trunc(historyRepresentationTokensAfterCompaction))
                : 0,
            });
            updateRunCompactionMetadata(currentRunId, {
              compactedMessageCount,
              threadSummary,
              detectedContextWindow,
            });
            break;
          }
          case "run_completed":
            clearTokenFlushTimer();
            clearThinkingFlushTimer();
            flushBufferedTokens();
            flushBufferedThinking();
            attachedRunIdRef.current = null;
            teardownRef.current?.();
            teardownRef.current = null;
            void finalizeRunFromArtifact(currentRunId, activeRunUserId, activeRunThreadId);
            break;
          case "run_failed":
            clearTokenFlushTimer();
            clearThinkingFlushTimer();
            flushBufferedTokens();
            flushBufferedThinking();
            attachedRunIdRef.current = null;
            teardownRef.current?.();
            teardownRef.current = null;
            void invalidateRunQueries(currentRunId, activeRunUserId, activeRunThreadId);
            if (isUserCancelledRunError(event.payload.error)) {
              cancelRun();
            } else {
              failRun(String(event.payload.error ?? "Atlas run failed."));
            }
            break;
          default:
            break;
        }
      },
      (message) => {
        clearTokenFlushTimer();
        clearThinkingFlushTimer();
        flushBufferedTokens();
        flushBufferedThinking();
        attachedRunIdRef.current = null;
        teardownRef.current = null;
        void recoverRunAfterStreamError(currentRunId, activeRunUserId, activeRunThreadId, message);
      },
    );
  }, [
    activeRunThreadId,
    activeRunUserId,
    appendToken,
    appendThinking,
    cancelRun,
    completeRun,
    currentRunId,
    currentRunMode,
    failRun,
    isStreaming,
    queryClient,
    setStage,
    showCompactionNotice,
  ]);

  useEffect(() => {
    return () => {
      clearTokenFlushTimer();
      clearThinkingFlushTimer();
      teardownRef.current?.();
      teardownRef.current = null;
      attachedRunIdRef.current = null;
    };
  }, []);

  return null;
}
