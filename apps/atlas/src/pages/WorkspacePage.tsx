import * as ScrollArea from "@radix-ui/react-scroll-area";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Check, ChevronDown, ChevronLeft, ChevronRight, Copy, CornerUpLeft, Download, Edit3, FileText, GitBranch, ImagePlus, Lightbulb, Lock, Plus, RotateCcw, Search, Send, Square, X } from "lucide-react";
import {
  ChangeEvent,
  FormEvent,
  KeyboardEvent,
  UIEvent,
  forwardRef,
  memo,
  useCallback,
  useDeferredValue,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import { useNavigate } from "react-router-dom";

import { MessageContent } from "../components/MessageContent";
import { ResetDialog } from "../components/ResetDialog";
import {
  cancelRun,
  branchThread,
  getRun,
  getModels,
  getStatus,
  getThreadContextUsage,
  getThreadHistory,
  getThreadRuns,
  getThreads,
  getUsers,
  openExternalUrl,
  renameThread,
  restartManagedBackend,
  startCompact,
  startChat,
  unloadOllamaModel,
  type ImageAttachment,
  type ReasoningMode,
  type RunStatusEvent,
  type RunSummary,
  type ThreadContextUsage,
  type ThreadMessage,
} from "../lib/api";
import { useBackendPhase } from "../lib/backendPhase";
import { buildChatMarkdownExport, chatExportFilename, downloadMarkdownFile } from "../lib/chatExport";
import { buildContextMeter, type ContextMeter } from "../lib/contextMeter";
import { PROFILE_SETTINGS_PATH } from "../lib/settingsSections";
import { resolveStartupState } from "../lib/startupState";
import { displayThreadTitle, editableThreadTitle, requestThreadTitle } from "../lib/threadTitles";
import { useAtlasStore } from "../store/useAtlasStore";

type ConversationMessage = {
  role: "user" | "assistant" | "system";
  content: string;
  attachments?: ImageAttachment[];
  ephemeral?: boolean;
  dismissible?: boolean;
  kind?: string;
  runId?: string;
  timestamp?: string;
  compactionReason?: string;
  threadSummary?: string;
  compactedMessageCount?: number;
  newlyCompactedMessageCount?: number;
  detectedContextWindow?: number;
  historyRepresentationTokensBeforeCompaction?: number;
  historyRepresentationTokensAfterCompaction?: number;
  historyIndex?: number;
};

type RetryContext = {
  afterMessageCount: number;
  prompt: string;
  attachments: ImageAttachment[];
};

type StartPromptPayload = {
  prompt: string;
  attachments: ImageAttachment[];
};

type WorkspaceComposerHandle = {
  quoteMessage: (content: string) => void;
  setPrompt: (content: string) => void;
};

type ConversationPresentationItem = {
  message: ConversationMessage;
  branchAfterMessageCount: number;
  retryContext: RetryContext | null;
};

type ThinkingEntry = {
  runId: string;
  text: string;
  status: string;
  chatModel?: string;
  startedAt?: string;
  durationMs?: number | null;
  isLive?: boolean;
};

type ReasoningOption = {
  value: ReasoningMode;
  label: string;
};

type PendingModelSwitch = {
  fromModels: string[];
  toModel: string;
  payload: StartPromptPayload;
  clearDraft: () => void;
};

const STARTER_PROMPTS = [
  {
    label: "Condense rough notes",
    prompt: "Turn these rough notes into a concise summary with decisions, open questions, and next actions:\n\n",
  },
  {
    label: "Plan a project",
    prompt: "Help me turn this idea into a practical project plan with milestones, risks, and the first three actions:\n\n",
  },
  {
    label: "Explain code",
    prompt: "Explain what this code does, identify likely failure points, and suggest the smallest useful improvements:\n\n",
  },
] as const;

export function WorkspacePage() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const conversationViewportRef = useRef<HTMLDivElement | null>(null);
  const autoScrollToLatestRef = useRef(true);
  const scrollToLatestFrameRef = useRef<number | null>(null);
  const scrollToLatestRetryFrameRef = useRef<number | null>(null);
  const scrollToLatestTimeoutRef = useRef<number | null>(null);
  const composerRef = useRef<WorkspaceComposerHandle | null>(null);

  const currentUserId = useAtlasStore((state) => state.currentUserId);
  const currentThreadId = useAtlasStore((state) => state.currentThreadId);
  const currentThreadTitle = useAtlasStore((state) => state.currentThreadTitle);
  const draftThreadModel = useAtlasStore((state) => state.draftThreadModel);
  const draftThreadTemperature = useAtlasStore((state) => state.draftThreadTemperature);
  const reasoningMode = useAtlasStore((state) => state.reasoningMode);
  const crossChatMemoryEnabled = useAtlasStore((state) => state.crossChatMemoryEnabled);
  const autoCompactLongChats = useAtlasStore((state) => state.autoCompactLongChats);
  const currentRunId = useAtlasStore((state) => state.currentRunId);
  const currentRunMode = useAtlasStore((state) => state.currentRunMode);
  const activeRunUserId = useAtlasStore((state) => state.activeRunUserId);
  const activeRunThreadId = useAtlasStore((state) => state.activeRunThreadId);
  const backendStartupStartedAt = useAtlasStore((state) => state.backendStartupStartedAt);
  const markBackendBooting = useAtlasStore((state) => state.markBackendBooting);
  const currentStage = useAtlasStore((state) => state.currentStage);
  const pendingPrompt = useAtlasStore((state) => state.pendingPrompt);
  const pendingAttachments = useAtlasStore((state) => state.pendingAttachments);
  const liveThinking = useAtlasStore((state) => state.liveThinking);
  const liveAnswer = useAtlasStore((state) => state.liveAnswer);
  const liveError = useAtlasStore((state) => state.liveError);
  const compactionNotice = useAtlasStore((state) => state.compactionNotice);
  const isStreaming = useAtlasStore((state) => state.isStreaming);
  const setCurrentThreadId = useAtlasStore((state) => state.setCurrentThreadId);
  const setCurrentThreadTitle = useAtlasStore((state) => state.setCurrentThreadTitle);
  const setDraftThreadModel = useAtlasStore((state) => state.setDraftThreadModel);
  const setDraftThreadTemperature = useAtlasStore((state) => state.setDraftThreadTemperature);
  const setReasoningMode = useAtlasStore((state) => state.setReasoningMode);
  const beginRun = useAtlasStore((state) => state.beginRun);
  const setStage = useAtlasStore((state) => state.setStage);
  const failRun = useAtlasStore((state) => state.failRun);
  const clearCompactionNotice = useAtlasStore((state) => state.clearCompactionNotice);
  const searchJumpTarget = useAtlasStore((state) => state.searchJumpTarget);
  const stepSearchJumpTarget = useAtlasStore((state) => state.stepSearchJumpTarget);
  const clearSearchJumpTarget = useAtlasStore((state) => state.clearSearchJumpTarget);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [draftTitle, setDraftTitle] = useState("");
  const [expandedCompactionKeys, setExpandedCompactionKeys] = useState<Record<string, boolean>>({});
  const [isThinkingPanelOpen, setIsThinkingPanelOpen] = useState(false);
  const [highlightedHistoryIndex, setHighlightedHistoryIndex] = useState<number | null>(null);
  const [copiedMessageKey, setCopiedMessageKey] = useState<string | null>(null);
  const [pendingModelSwitch, setPendingModelSwitch] = useState<PendingModelSwitch | null>(null);
  const [modelSwitchCheckPending, setModelSwitchCheckPending] = useState(false);

  const {
    data: status,
    isPending: statusPending,
    isFetching: statusFetching,
  } = useQuery({
    queryKey: ["status"],
    queryFn: getStatus,
    staleTime: 5000,
  });
  const backendPhase = useBackendPhase({
    hasStatus: Boolean(status),
    isPending: statusPending,
    isFetching: statusFetching,
    bootStartedAt: backendStartupStartedAt,
  });
  const backendOnline = backendPhase === "online";
  const { data: models } = useQuery({
    queryKey: ["models"],
    queryFn: getModels,
    enabled: backendOnline,
    staleTime: 10000,
  });
  const { data: users = [] } = useQuery({
    queryKey: ["users"],
    queryFn: getUsers,
    enabled: backendOnline,
    staleTime: 5000,
    retry: 1,
    refetchOnWindowFocus: false,
  });
  const visibleUsers = useMemo(() => {
    const seen = new Set<string>();
    return users.filter((user) => {
      if (!user.user_id || seen.has(user.user_id)) {
        return false;
      }
      seen.add(user.user_id);
      return true;
    });
  }, [users]);
  const currentUserProfile = visibleUsers.find((user) => user.user_id === currentUserId) ?? null;
  const currentUserLocked = Boolean(currentUserProfile?.locked);
  const { data: threads = [] } = useQuery({
    queryKey: ["threads", currentUserId],
    queryFn: () => getThreads(currentUserId),
    enabled: backendOnline && Boolean(currentUserId) && !currentUserLocked,
    staleTime: 2000,
  });
  const { data: history = [] } = useQuery({
    queryKey: ["thread-history", currentUserId, currentThreadId],
    queryFn: () => getThreadHistory(currentThreadId, currentUserId),
    enabled: backendOnline && Boolean(currentUserId && currentThreadId) && !currentUserLocked,
    staleTime: 2000,
  });
  const { data: contextUsage } = useQuery({
    queryKey: ["thread-context", currentUserId, currentThreadId],
    queryFn: () => getThreadContextUsage(currentThreadId, currentUserId),
    enabled: backendOnline && Boolean(currentUserId && currentThreadId) && !currentUserLocked,
    staleTime: 2000,
  });

  const threadItems = useMemo(() => {
    if (currentUserLocked) {
      return [];
    }
    const seen = new Set<string>();
    return threads.filter((thread) => {
      const key = `${thread.user_id}::${thread.thread_id}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }, [currentUserLocked, threads]);
  const visibleHistory = currentUserLocked ? [] : history;

  const defaultTemperature =
    models?.default_temperature ?? status?.default_chat_temperature ?? status?.chat_temperature ?? null;
  const modelCatalogLoaded = Boolean(models);
  const availableModels = (models?.models ?? []).filter(Boolean);
  const loadedModels = models?.loaded_models ?? [];
  const providerOnline = Boolean(models?.provider_online ?? models?.ollama_online);
  const hasLocalModels = Boolean(models?.has_chat_models ?? models?.has_local_models);
  const providerLabel = models?.provider_label || status?.chat_provider_label || "Ollama";
  const providerBaseUrl = models?.provider_base_url || status?.chat_base_url || status?.ollama_url || "http://127.0.0.1:11434";
  const isOllamaProvider = (models?.provider || status?.chat_provider || "ollama") === "ollama";
  const supportsModelUnload = Boolean(models?.supports_model_unload ?? true);

  const currentThread = useMemo(() => {
    const existing = threadItems.find((item) => item.thread_id === currentThreadId);
    if (existing) {
      return existing;
    }
    if (!currentThreadId) {
      return threadItems[0];
    }
    return {
      user_id: currentUserId,
      thread_id: currentThreadId,
      title: editableThreadTitle(currentThreadTitle, currentThreadId),
      chat_model: draftThreadModel,
      temperature: draftThreadTemperature,
      last_mode: "chat",
      updated_at: new Date().toISOString(),
      last_prompt: "",
      last_run_id: "",
    };
  }, [currentThreadId, currentThreadTitle, currentUserId, draftThreadModel, draftThreadTemperature, threadItems]);
  const currentThreadEditableTitle =
    editableThreadTitle(currentThread?.title, currentThreadId) ||
    editableThreadTitle(currentThreadTitle, currentThreadId);
  const currentThreadDisplayTitle = displayThreadTitle(currentThreadEditableTitle, currentThreadId);

  const currentThreadHasActiveRun =
    Boolean(currentRunId) &&
    isStreaming &&
    activeRunUserId === currentUserId &&
    activeRunThreadId === currentThreadId;
  const currentThreadHasLiveError =
    Boolean(liveError) &&
    activeRunUserId === currentUserId &&
    activeRunThreadId === currentThreadId;
  const activeRunIdForThread = currentThreadHasActiveRun ? currentRunId : null;
  const lastRunId = activeRunIdForThread ?? currentThread?.last_run_id ?? "";
  const { data: runDetails } = useQuery({
    queryKey: ["run", lastRunId],
    queryFn: () => getRun(lastRunId),
    enabled: Boolean(lastRunId),
    staleTime: 2000,
  });
  const { data: threadRuns = [] } = useQuery({
    queryKey: ["thread-runs", currentUserId, currentThreadId],
    queryFn: () => getThreadRuns(currentThreadId, currentUserId),
    enabled: Boolean(currentUserId && currentThreadId),
    staleTime: 2000,
  });

  const threadHasHistory = Boolean(history.length || currentThread?.last_run_id || runDetails?.run_id);
  const lockedThreadModel = threadHasHistory
      ? runDetails?.chat_model ||
      currentThread?.chat_model ||
      ""
    : "";
  const lockedThreadTemperature = useMemo<number | null | undefined>(() => {
    if (!threadHasHistory) {
      return undefined;
    }
    const runTemperature = readStoredTemperature(runDetails);
    if (runTemperature !== undefined) {
      return runTemperature;
    }
    const threadTemperature = readStoredTemperature(currentThread);
    if (threadTemperature !== undefined) {
      return threadTemperature;
    }
    return defaultTemperature;
  }, [currentThread, defaultTemperature, runDetails, threadHasHistory]);
  const preferredDraftModel = useMemo(() => {
    if (threadHasHistory && lockedThreadModel) {
      return lockedThreadModel;
    }
    if (draftThreadModel && availableModels.includes(draftThreadModel)) {
      return draftThreadModel;
    }
    return "";
  }, [availableModels, draftThreadModel, lockedThreadModel, threadHasHistory]);
  const selectedModel = lockedThreadModel || preferredDraftModel;
  const selectedTemperature = lockedThreadTemperature !== undefined ? lockedThreadTemperature : draftThreadTemperature;
  const selectedModelDetails = useMemo(
    () => models?.model_details?.find((item) => item.name === selectedModel),
    [models?.model_details, selectedModel],
  );
  const selectedModelSupportsImages = Boolean(selectedModelDetails?.supports_images);
  const selectedModelReasoningStrategy = selectedModelDetails?.reasoning_mode_strategy ?? "none";
  const selectedModelSupportsReasoning = Boolean(selectedModelDetails?.supports_reasoning && selectedModelReasoningStrategy !== "none");
  const effectiveReasoningMode = useMemo(
    () => normalizeReasoningModeForModel(reasoningMode, selectedModelReasoningStrategy),
    [reasoningMode, selectedModelReasoningStrategy],
  );
  const reasoningOptions = useMemo(
    () => buildReasoningOptions(selectedModelReasoningStrategy),
    [selectedModelReasoningStrategy],
  );
  const activeReasoningOption = useMemo(
    () => reasoningOptions.find((option) => option.value === effectiveReasoningMode) ?? reasoningOptions[0] ?? null,
    [effectiveReasoningMode, reasoningOptions],
  );
  const startupState = resolveStartupState({
    backendPhase,
    currentUserId,
    currentUserLocked,
    modelCatalogLoaded,
    ollamaOnline: providerOnline,
    hasLocalModels,
    selectedModel,
    providerLabel,
    selectedModelSupportsImages,
    threadHasHistory,
  });
  const headerSummary =
    startupState.key === "ready"
      ? startupState.headerSummary
      : "Local-first conversation workspace.";
  const idleTitle = startupState.idleTitle;
  const idleDescription = startupState.idleDescription;
  const composerPlaceholder = startupState.composerPlaceholder;
  const canStartChat = startupState.canStartChat;
  const modelControlsDisabled = !currentUserId || currentUserLocked || availableModels.length === 0;
  const currentThreadCompactionNotice = useMemo(() => {
    if (!compactionNotice) {
      return null;
    }
    if (compactionNotice.userId !== currentUserId || compactionNotice.threadId !== currentThreadId) {
      return null;
    }
    return compactionNotice;
  }, [compactionNotice, currentThreadId, currentUserId]);
  const activeSearchNavigator = useMemo(() => {
    if (!searchJumpTarget) {
      return null;
    }
    if (searchJumpTarget.userId !== currentUserId || searchJumpTarget.threadId !== currentThreadId) {
      return null;
    }
    const historyIndices = (searchJumpTarget.historyIndices ?? []).filter((value): value is number => typeof value === "number");
    if (!historyIndices.length) {
      return null;
    }
    const activePosition = Math.max(0, Math.min(historyIndices.length - 1, searchJumpTarget.activePosition ?? 0));
    return {
      query: searchJumpTarget.query,
      historyIndices,
      activePosition,
      historyIndex: historyIndices[activePosition] ?? null,
    };
  }, [currentThreadId, currentUserId, searchJumpTarget]);
  const currentThreadHasActiveChatRun = currentThreadHasActiveRun && currentRunMode === "chat";
  const persistedThinking = useMemo(
    () => extractThinkingText(runDetails?.events ?? []),
    [runDetails?.events],
  );
  const liveThinkingText = useMemo(() => {
    if (!currentThreadHasActiveChatRun) {
      return "";
    }
    return preferLongerText(persistedThinking, liveThinking).trim();
  }, [currentThreadHasActiveChatRun, liveThinking, persistedThinking]);
  const thinkingEntries = useMemo<ThinkingEntry[]>(() => {
    const persistedEntries = threadRuns
      .filter((run) => run.mode === "chat")
      .map((run) => {
        const text = extractThinkingText(run.events ?? []).trim();
        return {
          runId: run.run_id,
          text,
          status: run.status,
          chatModel: run.chat_model,
          startedAt: run.started_at,
          durationMs: run.diagnostics?.total_duration_ms ?? null,
          isLive: false,
        } as ThinkingEntry;
      })
      .filter((entry) => entry.text);

    if (!currentThreadHasActiveChatRun || !currentRunId) {
      return persistedEntries;
    }

    const liveEntry: ThinkingEntry = {
      runId: currentRunId,
      text: liveThinkingText,
      status: "running",
      chatModel: selectedModel || runDetails?.chat_model,
      startedAt: runDetails?.started_at || new Date().toISOString(),
      durationMs: runDetails?.diagnostics?.total_duration_ms ?? null,
      isLive: true,
    };

    const existingIndex = persistedEntries.findIndex((entry) => entry.runId === currentRunId);
    if (existingIndex >= 0) {
      const nextEntries = [...persistedEntries];
      nextEntries[existingIndex] = {
        ...nextEntries[existingIndex],
        ...liveEntry,
        text: liveThinkingText || nextEntries[existingIndex].text,
      };
      return nextEntries;
    }
    return liveThinkingText ? [...persistedEntries, liveEntry] : persistedEntries;
  }, [currentRunId, currentThreadHasActiveChatRun, liveThinkingText, runDetails?.chat_model, runDetails?.diagnostics?.total_duration_ms, runDetails?.started_at, selectedModel, threadRuns]);
  const latestThinkingEntry = thinkingEntries[thinkingEntries.length - 1];
  const canToggleThinkingPanel = Boolean(currentThreadHasActiveChatRun || thinkingEntries.length);
  const thinkingPanelStatusLabel = currentThreadHasActiveChatRun
    ? chatWaitingLabel(currentStage)
    : formatRunStatusLabel(latestThinkingEntry ? ({
        run_id: latestThinkingEntry.runId,
        mode: "chat",
        user_id: currentUserId,
        thread_id: currentThreadId,
        prompt: "",
        status: latestThinkingEntry.status,
        started_at: latestThinkingEntry.startedAt || "",
        answer: "",
        events: [],
      } as RunSummary) : runDetails);
  const thinkingPanelStatusClass = currentThreadHasActiveChatRun
    ? "online"
    : latestThinkingEntry?.status === "failed" || runDetails?.status === "failed"
      ? "offline"
      : latestThinkingEntry?.status === "completed" || runDetails?.status === "completed"
        ? "subtle"
        : "muted";
  const currentChatWaitingLabel = chatWaitingLabel(currentStage);

  useEffect(() => {
    setDraftTitle(currentThreadEditableTitle);
    setIsEditingTitle(false);
  }, [currentThreadEditableTitle]);

  useEffect(() => {
    setExpandedCompactionKeys({});
  }, [currentUserId, currentThreadId]);

  useEffect(() => {
    setIsThinkingPanelOpen(false);
  }, [currentThreadId, currentUserId]);

  const clearScheduledScrollToLatest = useCallback(() => {
    if (scrollToLatestFrameRef.current !== null) {
      window.cancelAnimationFrame(scrollToLatestFrameRef.current);
      scrollToLatestFrameRef.current = null;
    }
    if (scrollToLatestRetryFrameRef.current !== null) {
      window.cancelAnimationFrame(scrollToLatestRetryFrameRef.current);
      scrollToLatestRetryFrameRef.current = null;
    }
    if (scrollToLatestTimeoutRef.current !== null) {
      window.clearTimeout(scrollToLatestTimeoutRef.current);
      scrollToLatestTimeoutRef.current = null;
    }
  }, []);

  const scrollConversationToLatest = useCallback(() => {
    const viewport = conversationViewportRef.current;
    if (!viewport || !autoScrollToLatestRef.current) {
      return;
    }
    viewport.scrollTo({ top: viewport.scrollHeight });
  }, []);

  const scheduleScrollToLatest = useCallback(() => {
    autoScrollToLatestRef.current = true;
    clearScheduledScrollToLatest();
    scrollToLatestFrameRef.current = window.requestAnimationFrame(() => {
      scrollToLatestFrameRef.current = null;
      scrollConversationToLatest();
      scrollToLatestRetryFrameRef.current = window.requestAnimationFrame(() => {
        scrollToLatestRetryFrameRef.current = null;
        scrollConversationToLatest();
      });
    });
    scrollToLatestTimeoutRef.current = window.setTimeout(() => {
      scrollToLatestTimeoutRef.current = null;
      scrollConversationToLatest();
    }, 160);
  }, [clearScheduledScrollToLatest, scrollConversationToLatest]);

  useEffect(() => clearScheduledScrollToLatest, [clearScheduledScrollToLatest]);

  const startRun = useMutation({
    mutationFn: async (payload: StartPromptPayload) => {
      if (!currentUserId) {
        throw new Error("Choose a profile before starting the first chat.");
      }
      if (!providerOnline) {
        throw new Error(`Start ${providerLabel} on this machine before starting the first chat.`);
      }
      if (!hasLocalModels) {
        throw new Error(`Add a local chat model to ${providerLabel} before starting the first chat.`);
      }
      const modelForRun = lockedThreadModel || selectedModel;
      if (!modelForRun) {
        throw new Error("Select a local chat model before starting this chat.");
      }
      const temperatureForRun = lockedThreadTemperature !== undefined ? lockedThreadTemperature : (selectedTemperature ?? null);
      return startChat(
        payload.prompt,
        currentUserId,
        currentThreadId,
        modelForRun,
        temperatureForRun,
        effectiveReasoningMode,
        requestThreadTitle(currentThreadEditableTitle, currentThreadId),
        crossChatMemoryEnabled,
        autoCompactLongChats,
        payload.attachments,
      );
    },
    onSuccess: ({ run_id }, payload) => {
      autoScrollToLatestRef.current = true;
      beginRun(run_id, "chat", payload.prompt, currentUserId, currentThreadId, payload.attachments);
    },
    onError: (error) => {
      failRun(error instanceof Error ? error.message : "Atlas run failed.", currentUserId, currentThreadId);
    },
  });

  const startManualCompact = useMutation({
    mutationFn: async () => {
      if (!currentUserId) {
        throw new Error("Create or select a user in Settings before compacting a chat.");
      }
      return startCompact(currentThreadId, currentUserId);
    },
    onSuccess: ({ run_id }) => {
      autoScrollToLatestRef.current = true;
      beginRun(run_id, "compact", "", currentUserId, currentThreadId, []);
    },
    onError: (error) => {
      failRun(error instanceof Error ? error.message : "Atlas could not compact this thread.", currentUserId, currentThreadId);
    },
  });

  const stopRun = useMutation({
    mutationFn: async () => {
      if (!currentRunId) {
        throw new Error("No active run to stop.");
      }
      return cancelRun(currentRunId);
    },
    onSuccess: () => {
      setStage("stopping");
    },
    onError: (error) => {
      failRun(error instanceof Error ? error.message : "Atlas could not stop this run.", currentUserId, currentThreadId);
    },
  });

  const branchMessage = useMutation({
    mutationFn: async (payload: { afterMessageCount: number }) => {
      if (!currentUserId) {
        throw new Error("Choose a profile before branching this chat.");
      }
      return branchThread(currentThreadId, currentUserId, payload.afterMessageCount);
    },
    onSuccess: async (thread) => {
      setCurrentThreadId(thread.thread_id);
      setCurrentThreadTitle(editableThreadTitle(thread.title, thread.thread_id));
      setDraftThreadModel(thread.chat_model || "");
      setDraftThreadTemperature(readStoredTemperature(thread) ?? null);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["threads", currentUserId] }),
        queryClient.invalidateQueries({ queryKey: ["thread-history", currentUserId] }),
      ]);
    },
  });

  const retryAssistantTurn = useMutation({
    mutationFn: async (payload: { afterMessageCount: number; prompt: string; attachments: ImageAttachment[] }) => {
      if (!currentUserId) {
        throw new Error("Choose a profile before retrying a response.");
      }
      const thread = await branchThread(currentThreadId, currentUserId, payload.afterMessageCount);
      const run = await startChat(
        payload.prompt,
        currentUserId,
        thread.thread_id,
        thread.chat_model || selectedModel,
        readStoredTemperature(thread) ?? selectedTemperature ?? null,
        effectiveReasoningMode,
        requestThreadTitle(thread.title, thread.thread_id),
        crossChatMemoryEnabled,
        autoCompactLongChats,
        payload.attachments,
      );
      return { thread, run, prompt: payload.prompt, attachments: payload.attachments };
    },
    onSuccess: async ({ thread, run, prompt: retryPrompt, attachments: retryAttachments }) => {
      autoScrollToLatestRef.current = true;
      setCurrentThreadId(thread.thread_id);
      setCurrentThreadTitle(editableThreadTitle(thread.title, thread.thread_id));
      setDraftThreadModel(thread.chat_model || "");
      setDraftThreadTemperature(readStoredTemperature(thread) ?? null);
      beginRun(run.run_id, "chat", retryPrompt, currentUserId, thread.thread_id, retryAttachments);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["threads", currentUserId] }),
        queryClient.invalidateQueries({ queryKey: ["thread-history", currentUserId] }),
      ]);
    },
    onError: (error) => {
      failRun(error instanceof Error ? error.message : "Atlas could not retry this response.", currentUserId, currentThreadId);
    },
  });

  const refreshModels = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["status"] }),
      queryClient.invalidateQueries({ queryKey: ["models"] }),
      queryClient.invalidateQueries({ queryKey: ["threads"] }),
    ]);
  };

  const restartBackend = async () => {
    markBackendBooting();
    await restartManagedBackend();
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["status"] }),
      queryClient.invalidateQueries({ queryKey: ["models"] }),
      queryClient.invalidateQueries({ queryKey: ["users"] }),
    ]);
  };

  useEffect(() => {
    if (!currentThreadId && threadItems.length) {
      setCurrentThreadId(threadItems[0].thread_id);
    }
  }, [currentThreadId, setCurrentThreadId, threadItems]);

  useEffect(() => {
    if (threadHasHistory) {
      return;
    }
    if (!preferredDraftModel) {
      return;
    }
    if (draftThreadModel !== preferredDraftModel) {
      setDraftThreadModel(preferredDraftModel);
    }
  }, [draftThreadModel, preferredDraftModel, setDraftThreadModel, threadHasHistory]);

  useEffect(() => {
    const resolvedTitle = editableThreadTitle(currentThread?.title, currentThreadId);
    if (resolvedTitle && resolvedTitle !== currentThreadTitle) {
      setCurrentThreadTitle(resolvedTitle);
    }
  }, [currentThread?.title, currentThreadId, currentThreadTitle, setCurrentThreadTitle]);

  const transcript = useMemo(() => {
    const items: ConversationMessage[] = visibleHistory.map((item: ThreadMessage, historyIndex) => ({
      role: item.role,
      content: item.content,
      attachments: item.attachments,
      historyIndex: typeof item.history_index === "number" ? item.history_index : historyIndex,
      kind: item.kind,
      runId: item.run_id,
      timestamp: item.timestamp,
      compactionReason: item.compaction_reason,
      threadSummary: item.thread_summary,
      compactedMessageCount: item.compacted_message_count,
      newlyCompactedMessageCount: item.newly_compacted_message_count,
      detectedContextWindow: item.detected_context_window,
      historyRepresentationTokensBeforeCompaction: item.history_representation_tokens_before_compaction,
      historyRepresentationTokensAfterCompaction: item.history_representation_tokens_after_compaction,
    }));
    if (currentThreadHasActiveRun && (pendingPrompt || pendingAttachments.length)) {
      items.push({ role: "user", content: pendingPrompt, attachments: pendingAttachments, ephemeral: true, runId: currentRunId || undefined });
    }
    if (currentThreadHasActiveRun && currentThreadCompactionNotice) {
      items.push(buildLiveCompactionMessage(currentThreadCompactionNotice));
    }
    if (currentThreadHasActiveRun && liveAnswer) {
      items.push({ role: "assistant", content: liveAnswer, ephemeral: true, runId: currentRunId || undefined });
    }
    return items;
  }, [currentRunId, currentThreadCompactionNotice, currentThreadHasActiveRun, liveAnswer, pendingAttachments, pendingPrompt, visibleHistory]);
  const transcriptPresentation = useMemo(() => buildConversationPresentation(transcript), [transcript]);
  const isCompactingContext =
    startManualCompact.isPending || (isStreaming && (currentRunMode === "compact" || currentStage === "compaction"));
  const showOllamaWarning = startupState.key === "ollama-offline" && transcript.length > 0;
  const idleKicker =
    startupState.key === "ready" && selectedModel
      ? formatModelLabel(selectedModel)
      : startupState.idleKicker;

  useEffect(() => {
    setHighlightedHistoryIndex(null);
    scheduleScrollToLatest();
  }, [currentUserId, currentThreadId, scheduleScrollToLatest]);

  useEffect(() => {
    if (!searchJumpTarget) {
      return;
    }
    if (searchJumpTarget.userId !== currentUserId || searchJumpTarget.threadId !== currentThreadId) {
      return;
    }
    if (searchJumpTarget.historyIndex === null || searchJumpTarget.historyIndex === undefined) {
      if (!searchJumpTarget.historyIndices?.length) {
        clearSearchJumpTarget();
      }
      return;
    }
    const targetHistoryIndex = searchJumpTarget.historyIndex;
    const targetElement = conversationViewportRef.current?.querySelector<HTMLElement>(
      `[data-history-index="${targetHistoryIndex}"]`,
    );
    if (!targetElement) {
      return;
    }
    autoScrollToLatestRef.current = false;
    targetElement.scrollIntoView({ behavior: "smooth", block: "center" });
    setHighlightedHistoryIndex(targetHistoryIndex);
  }, [clearSearchJumpTarget, currentThreadId, currentUserId, searchJumpTarget, transcript]);

  useEffect(() => {
    if (highlightedHistoryIndex === null) {
      return undefined;
    }
    const timer = window.setTimeout(() => setHighlightedHistoryIndex(null), 2400);
    return () => window.clearTimeout(timer);
  }, [highlightedHistoryIndex]);

  useEffect(() => {
    if (!activeSearchNavigator) {
      return undefined;
    }
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "F3") {
        event.preventDefault();
        stepSearchJumpTarget(event.shiftKey ? -1 : 1);
        return;
      }
      if (event.key === "Escape") {
        clearSearchJumpTarget();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeSearchNavigator, clearSearchJumpTarget, stepSearchJumpTarget]);

  useEffect(() => {
    if (!copiedMessageKey) {
      return undefined;
    }
    const timer = window.setTimeout(() => setCopiedMessageKey(null), 1600);
    return () => window.clearTimeout(timer);
  }, [copiedMessageKey]);

  useEffect(() => {
    if (!autoScrollToLatestRef.current) {
      return;
    }
    scheduleScrollToLatest();
  }, [scheduleScrollToLatest, transcript]);

  const handleConversationScroll = (event: UIEvent<HTMLDivElement>) => {
    autoScrollToLatestRef.current = isNearBottom(event.currentTarget);
  };

  const handleCopyMessage = useCallback(async (message: ConversationMessage, index: number) => {
    const key = messageActionKey(message, index, "copy");
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(message.content);
      }
      setCopiedMessageKey(key);
    } catch {
      // Ignore clipboard failures silently. The action should stay non-blocking.
    }
  }, []);

  const handleExportMarkdown = () => {
    const markdown = buildChatMarkdownExport(
      {
        title: currentThreadDisplayTitle,
        threadId: currentThreadId,
        userId: currentUserId,
        model: selectedModel || lockedThreadModel || currentThread?.chat_model,
      },
      visibleHistory.map((message) => ({
        role: message.role,
        content: message.content,
        attachments: message.attachments,
        timestamp: message.timestamp,
      })),
    );
    downloadMarkdownFile(chatExportFilename(currentThreadDisplayTitle, currentThreadId), markdown);
  };

  const handleQuoteMessage = useCallback((message: ConversationMessage) => {
    composerRef.current?.quoteMessage(message.content);
  }, []);

  const commitTitle = useMutation({
    mutationFn: async (title: string) => renameThread(currentThreadId, currentUserId, title),
    onSuccess: async (thread) => {
      setCurrentThreadTitle(editableThreadTitle(thread.title, thread.thread_id));
      setIsEditingTitle(false);
      await queryClient.invalidateQueries({ queryKey: ["threads", currentUserId] });
    },
  });

  const saveTitle = async () => {
    const normalized = draftTitle.trim();
    if (!threadHasHistory) {
      setCurrentThreadTitle(normalized);
      setDraftTitle(normalized);
      setIsEditingTitle(false);
      return;
    }
    await commitTitle.mutateAsync(normalized || currentThreadDisplayTitle);
  };

  const toggleCompactionSummary = useCallback((key: string) => {
    setExpandedCompactionKeys((current) => ({ ...current, [key]: !current[key] }));
  }, []);

  const openThinkingPanel = useCallback(() => {
    setIsThinkingPanelOpen(true);
  }, []);

  const branchMessageMutate = branchMessage.mutate;
  const handleBranchMessage = useCallback((afterMessageCount: number) => {
    branchMessageMutate({ afterMessageCount });
  }, [branchMessageMutate]);

  const retryAssistantTurnMutate = retryAssistantTurn.mutate;
  const handleRetryAssistantTurn = useCallback((retryContext: RetryContext) => {
    retryAssistantTurnMutate({
      afterMessageCount: retryContext.afterMessageCount,
      prompt: retryContext.prompt,
      attachments: retryContext.attachments,
    });
  }, [retryAssistantTurnMutate]);

  const startRunMutate = startRun.mutate;
  const handleSubmitPrompt = useCallback((payload: StartPromptPayload, clearDraft: () => void) => {
    if (modelSwitchCheckPending || pendingModelSwitch) {
      return;
    }
    const modelForRun = (lockedThreadModel || selectedModel || "").trim();
    setModelSwitchCheckPending(true);
    void queryClient.fetchQuery({ queryKey: ["models"], queryFn: getModels, staleTime: 0 })
      .catch(() => models)
      .then((freshModels) => {
        const canUnloadModels = Boolean(freshModels?.supports_model_unload ?? supportsModelUnload);
        const fromModels = canUnloadModels
          ? modelsToStopBeforeSwitch(freshModels?.loaded_models ?? loadedModels, modelForRun)
          : [];
        if (modelForRun && fromModels.length) {
          setPendingModelSwitch({
            fromModels,
            toModel: modelForRun,
            payload,
            clearDraft,
          });
          return;
        }
        startRunMutate(payload, { onSuccess: clearDraft });
      })
      .finally(() => setModelSwitchCheckPending(false));
  }, [loadedModels, lockedThreadModel, modelSwitchCheckPending, models, pendingModelSwitch, queryClient, selectedModel, startRunMutate, supportsModelUnload]);

  const confirmModelSwitch = useCallback(async () => {
    if (!pendingModelSwitch) {
      return;
    }
    try {
      for (const model of pendingModelSwitch.fromModels) {
        await unloadOllamaModel(model);
      }
      await queryClient.invalidateQueries({ queryKey: ["models"] });
    } catch (error) {
      failRun(
        error instanceof Error ? error.message : "Atlas could not stop the previous local model.",
        currentUserId,
        currentThreadId,
      );
      setPendingModelSwitch(null);
      return;
    }
    const { payload, clearDraft } = pendingModelSwitch;
    setPendingModelSwitch(null);
    startRunMutate(payload, { onSuccess: clearDraft });
  }, [currentThreadId, currentUserId, failRun, pendingModelSwitch, queryClient, startRunMutate]);

  const startManualCompactMutate = startManualCompact.mutate;
  const handleManualCompact = useCallback(() => {
    startManualCompactMutate();
  }, [startManualCompactMutate]);

  const stopRunMutate = stopRun.mutate;
  const handleStopRun = useCallback(() => {
    stopRunMutate();
  }, [stopRunMutate]);

  const toggleThinkingPanel = () => {
    if (!canToggleThinkingPanel) {
      return;
    }
    setIsThinkingPanelOpen((current) => !current);
  };

  const closeThinkingPanel = () => {
    setIsThinkingPanelOpen(false);
  };

  return (
    <section className={`workspace-main workspace-main-single${isThinkingPanelOpen ? " has-thinking-panel" : ""}`}>
      <div className="workspace-primary-column">
      <div className="workspace-main-header">
        <div className="workspace-title-block">
          <div className="workspace-title-row">
            {isEditingTitle ? (
              <>
                <input
                  aria-label="Chat title"
                  className="text-input workspace-title-input"
                  onChange={(event) => setDraftTitle(event.currentTarget.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      void saveTitle();
                    }
                    if (event.key === "Escape") {
                      setDraftTitle(currentThreadEditableTitle);
                      setIsEditingTitle(false);
                    }
                  }}
                  placeholder="Chat title"
                  value={draftTitle}
                />
                <button
                  aria-label="Save chat title"
                  className="ghost-button icon-button"
                  onClick={() => void saveTitle()}
                  title="Save title"
                  type="button"
                >
                  <Check size={16} />
                </button>
                <button
                  aria-label="Cancel title editing"
                  className="ghost-button icon-button"
                  onClick={() => {
                    setDraftTitle(currentThreadEditableTitle);
                    setIsEditingTitle(false);
                  }}
                  title="Cancel"
                  type="button"
                >
                  <X size={16} />
                </button>
              </>
            ) : (
              <>
                <h1>{currentThreadDisplayTitle}</h1>
                <button
                  aria-label="Edit chat title"
                  className="ghost-button icon-button title-edit-button"
                  onClick={() => setIsEditingTitle(true)}
                  title="Edit chat title"
                  type="button"
                >
                  <Edit3 size={16} />
                </button>
              </>
            )}
          </div>
          <p className="workspace-title-summary">{headerSummary}</p>
        </div>

        <div className="workspace-header-controls">
          <div className="workspace-header-control-strip">
            <div className="workspace-model-group">
              <span className="workspace-model-label">Export</span>
              <button
                className="ghost-button compact-button workspace-export-button"
                disabled={!visibleHistory.length}
                onClick={handleExportMarkdown}
                title={visibleHistory.length ? "Export chat to Markdown" : "No saved messages to export"}
                type="button"
              >
                <Download size={15} />
                Markdown
              </button>
            </div>

            {canToggleThinkingPanel ? (
              <div className="workspace-model-group">
                <span className="workspace-model-label">Trace</span>
                <button
                  aria-expanded={isThinkingPanelOpen}
                  aria-label={isThinkingPanelOpen ? "Hide thinking panel" : "Show thinking panel"}
                  className={`ghost-button compact-button workspace-thinking-open-button${isThinkingPanelOpen ? " active" : ""}`}
                  onClick={toggleThinkingPanel}
                  title={isThinkingPanelOpen ? "Hide thinking panel" : "Show thinking panel"}
                  type="button"
                >
                  <Lightbulb size={15} />
                  Thinking
                </button>
              </div>
            ) : null}

            <div className="workspace-model-group">
              <span className="workspace-model-label">Model</span>
              {lockedThreadModel ? (
                <div className="workspace-model-lock" title="This chat already started with this model.">
                  <Lock size={14} />
                  <span>{formatModelLabel(lockedThreadModel)}</span>
                </div>
              ) : (
                <label className="workspace-model-picker">
                  <select
                    aria-label="Chat model"
                    className="select-input workspace-model-select"
                    disabled={modelControlsDisabled}
                    onChange={(event) => setDraftThreadModel(event.currentTarget.value)}
                    value={selectedModel}
                  >
                    {availableModels.length > 0 ? (
                      <>
                        <option disabled value="">
                          Choose model
                        </option>
                        {availableModels.map((model) => (
                          <option key={model} value={model}>
                            {formatModelLabel(model)}
                          </option>
                        ))}
                      </>
                    ) : (
                      <option value="">No model available</option>
                    )}
                  </select>
                </label>
              )}
            </div>

            <div className="workspace-model-group">
              <span className="workspace-model-label">Temp</span>
              {lockedThreadTemperature !== undefined ? (
                <div className="workspace-model-lock" title="This chat already started with this temperature.">
                  <Lock size={14} />
                  <span>{formatTemperatureLabel(lockedThreadTemperature)}</span>
                </div>
              ) : (
                <label className="workspace-model-picker workspace-temperature-picker">
                  <select
                    aria-label="Chat temperature"
                    className="select-input workspace-model-select"
                    disabled={modelControlsDisabled}
                    onChange={(event) => setDraftThreadTemperature(parseTemperatureValue(event.currentTarget.value))}
                    value={formatTemperatureSelectValue(selectedTemperature)}
                  >
                    <option value={MODEL_DEFAULT_TEMPERATURE_VALUE}>Model setting</option>
                    {TEMPERATURE_OPTIONS.map((value) => (
                      <option key={value} value={value.toFixed(1)}>
                        {value.toFixed(1)}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>

          </div>
        </div>
      </div>

      <div className="conversation-shell">
        {showOllamaWarning ? (
          <div className="workspace-warning-banner" role="status">
            <div className="workspace-warning-copy">
              <strong>{providerLabel} is not running.</strong>
              <span>
                Start the local provider at <strong>{providerBaseUrl}</strong>, then refresh Atlas.
              </span>
            </div>
            <button className="ghost-button compact-button" onClick={() => void refreshModels()} type="button">
              Refresh
            </button>
          </div>
        ) : null}
        {activeSearchNavigator ? (
          <div className="search-inline-navigator">
            <div className="search-inline-copy">
              <span className="status-pill subtle muted">
                <Search size={13} />
                Find in chat
              </span>
              <strong>{activeSearchNavigator.query}</strong>
              <span className="muted-text">
                {activeSearchNavigator.activePosition + 1} / {activeSearchNavigator.historyIndices.length}
              </span>
            </div>
            <div className="search-inline-actions">
              <button
                aria-label="Previous search match"
                className="ghost-button icon-button"
                disabled={activeSearchNavigator.activePosition <= 0}
                onClick={() => stepSearchJumpTarget(-1)}
                type="button"
              >
                <ChevronLeft size={15} />
              </button>
              <button
                aria-label="Next search match"
                className="ghost-button icon-button"
                disabled={activeSearchNavigator.activePosition >= activeSearchNavigator.historyIndices.length - 1}
                onClick={() => stepSearchJumpTarget(1)}
                type="button"
              >
                <ChevronRight size={15} />
              </button>
              <button
                className="ghost-button compact-button"
                onClick={() => clearSearchJumpTarget()}
                type="button"
              >
                <X size={14} />
                Close
              </button>
            </div>
          </div>
        ) : null}
        <ScrollArea.Root className="conversation-scroll">
          <ScrollArea.Viewport
            className="conversation-viewport"
            onScroll={handleConversationScroll}
            ref={conversationViewportRef}
          >
            <div className="conversation-stack">
              {transcript.length === 0 ? (
                <div className="workspace-idle">
                  <div className={`workspace-idle-card${startupState.key === "no-profile" ? " workspace-onboarding-card" : ""}`}>
                    <div className="workspace-idle-mark">
                      <img alt="Atlas Chat" className="workspace-idle-logo workspace-idle-logo-large" src="/AtlasLogo.png" />
                    </div>
                    <span className="workspace-idle-kicker">{idleKicker}</span>
                    <h2>{idleTitle}</h2>
                    <p>{idleDescription}</p>
                    {startupState.key === "no-profile" ? (
                      <div className="workspace-idle-actions">
                        <button
                          className="primary-button"
                          onClick={() => navigate(PROFILE_SETTINGS_PATH)}
                          type="button"
                        >
                          Create or pick a profile
                        </button>
                      </div>
                    ) : startupState.key === "backend-offline" ? (
                      <div className="workspace-idle-actions">
                        <button
                          className="primary-button"
                          onClick={() => void restartBackend()}
                          type="button"
                        >
                          Restart local runtime
                        </button>
                      </div>
                    ) : startupState.key === "ollama-offline" ? (
                      <div className="workspace-idle-actions">
                        {isOllamaProvider ? (
                          <a
                            className="primary-button"
                            href="https://ollama.com/download"
                            onClick={(event) => {
                              event.preventDefault();
                              void openExternalUrl("https://ollama.com/download");
                            }}
                            rel="noreferrer"
                            target="_blank"
                          >
                            Get Ollama
                          </a>
                        ) : (
                          <button
                            className="primary-button"
                            onClick={() => void refreshModels()}
                            type="button"
                          >
                            Refresh
                          </button>
                        )}
                        <button
                          className="ghost-button"
                          onClick={() => navigate("/settings?section=connections")}
                          type="button"
                        >
                          Connections
                        </button>
                      </div>
                    ) : startupState.key === "no-local-models" ? (
                      <div className="workspace-idle-actions">
                        <button
                          className="primary-button"
                          onClick={() => navigate("/discovery")}
                          type="button"
                        >
                          Find a model
                        </button>
                      </div>
                    ) : startupState.key === "profile-locked" ? (
                      <div className="workspace-idle-actions">
                        <button
                          className="primary-button"
                          onClick={() => navigate(PROFILE_SETTINGS_PATH)}
                          type="button"
                        >
                          Unlock in Settings
                        </button>
                      </div>
                    ) : startupState.key === "ready" ? (
                      <div aria-label="Starter prompts" className="workspace-starter-grid">
                        {STARTER_PROMPTS.map((starter) => (
                          <button
                            className="workspace-starter-action"
                            key={starter.label}
                            onClick={() => composerRef.current?.setPrompt(starter.prompt)}
                            type="button"
                          >
                            <span>{starter.label}</span>
                            <ArrowRight size={15} />
                          </button>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : null}
              <ConversationTranscript
                branchPending={branchMessage.isPending}
                canBranch={Boolean(currentUserId)}
                copiedMessageKey={copiedMessageKey}
                expandedCompactionKeys={expandedCompactionKeys}
                highlightedHistoryIndex={highlightedHistoryIndex}
                isStreaming={isStreaming}
                items={transcriptPresentation}
                onBranchMessage={handleBranchMessage}
                onClearCompactionNotice={clearCompactionNotice}
                onCopyMessage={handleCopyMessage}
                onQuoteMessage={handleQuoteMessage}
                onRetryAssistantTurn={handleRetryAssistantTurn}
                onToggleCompactionSummary={toggleCompactionSummary}
                retryPending={retryAssistantTurn.isPending}
              />
              {currentThreadHasActiveRun && isStreaming && !liveAnswer ? (
                <article className={`message-card ${currentRunMode === "compact" ? "system" : "assistant"} message-card-waiting`}>
                  <div className="message-meta">
                    <span>{currentRunMode === "compact" ? "SYSTEM" : formatMessageRoleLabel("assistant")}</span>
                  </div>
                  {currentRunMode === "compact" ? (
                    <div className="stream-waiting-line" aria-live="polite">
                      <span className="stream-waiting-text">{compactWaitingLabel(currentStage)}</span>
                    </div>
                  ) : (
                    <button
                      aria-expanded={isThinkingPanelOpen}
                      aria-label={isThinkingPanelOpen ? "Hide live thinking" : "Show live thinking"}
                      className={`thinking-toggle ${canToggleThinkingPanel ? "interactive" : ""}`}
                      disabled={!canToggleThinkingPanel}
                      onClick={() => {
                        if (!canToggleThinkingPanel) {
                          return;
                        }
                        openThinkingPanel();
                      }}
                      type="button"
                    >
                      <span className="stream-waiting-line" aria-live="polite">
                        <span className={`stream-waiting-text${currentChatWaitingLabel === "Deciding" ? " deciding-sweep" : ""}`}>
                          {currentChatWaitingLabel}
                        </span>
                      </span>
                    </button>
                  )}
                </article>
              ) : null}
              {currentThreadHasLiveError ? <div className="error-banner" role="alert">{liveError}</div> : null}
            </div>
          </ScrollArea.Viewport>
          <ScrollArea.Scrollbar className="scrollbar" orientation="vertical">
            <ScrollArea.Thumb className="scrollbar-thumb" />
          </ScrollArea.Scrollbar>
        </ScrollArea.Root>
      </div>

      <WorkspaceComposer
        activeReasoningOption={activeReasoningOption}
        canStartChat={canStartChat}
        contextUsage={contextUsage}
        currentRunId={currentRunId}
        currentRunMode={currentRunMode}
        currentThreadCompactionNotice={currentThreadCompactionNotice}
        currentThreadHasActiveRun={currentThreadHasActiveRun}
        currentUserId={currentUserId}
        effectiveReasoningMode={effectiveReasoningMode}
        isCompactingContext={isCompactingContext}
        isStreaming={isStreaming}
        liveAnswer={liveAnswer}
        onManualCompact={handleManualCompact}
        onStopRun={handleStopRun}
        onSubmitPrompt={handleSubmitPrompt}
        pendingAttachments={pendingAttachments}
        pendingPrompt={pendingPrompt}
        placeholder={composerPlaceholder}
        providerLabel={providerLabel}
        reasoningOptions={reasoningOptions}
        ref={composerRef}
        runDetectedContextWindow={runDetails?.detected_context_window}
        selectedModelSupportsImages={selectedModelSupportsImages}
        selectedModelSupportsReasoning={selectedModelSupportsReasoning}
        setReasoningMode={setReasoningMode}
        startManualCompactPending={startManualCompact.isPending}
        startRunPending={startRun.isPending || modelSwitchCheckPending || Boolean(pendingModelSwitch)}
        stopRunPending={stopRun.isPending}
        threadHasHistory={threadHasHistory}
        visibleHistory={visibleHistory}
      />
      </div>
      {isThinkingPanelOpen ? (
        <aside className="workspace-thinking-inspector" aria-label="Thinking panel">
          <div className="workspace-thinking-panel">
            <div className="workspace-thinking-panel-header">
              <div className="workspace-thinking-panel-copy">
                <p className="workspace-section-label">Thinking</p>
                <h2>{displayThreadTitle(currentThreadEditableTitle, currentThreadId, "Run details")}</h2>
                <p className="workspace-thinking-panel-summary">
                  {currentThreadHasActiveChatRun
                    ? "New deciding traces append here and stay in order for this thread."
                    : "Saved deciding traces stay here in order for this thread."}
                </p>
              </div>
              <button
                aria-label="Close thinking panel"
                className="ghost-button icon-button"
                onClick={closeThinkingPanel}
                type="button"
              >
                <X size={16} />
              </button>
            </div>

            <div className="workspace-thinking-panel-meta">
              <span className={`status-pill ${thinkingPanelStatusClass}`}>
                <span className="status-dot" />
                <span>{thinkingPanelStatusLabel}</span>
              </span>
              {latestThinkingEntry?.chatModel ? <span className="workspace-thinking-meta-chip">{latestThinkingEntry.chatModel}</span> : null}
              {latestThinkingEntry?.durationMs ? (
                <span className="workspace-thinking-meta-chip">
                  {formatDurationLabel(latestThinkingEntry.durationMs)}
                </span>
              ) : null}
              {latestThinkingEntry?.startedAt ? (
                <span className="workspace-thinking-meta-chip">{formatInspectorTimestamp(latestThinkingEntry.startedAt)}</span>
              ) : null}
              {thinkingEntries.length > 1 ? (
                <span className="workspace-thinking-meta-chip">{`${thinkingEntries.length} runs`}</span>
              ) : null}
            </div>

            <div className="workspace-thinking-panel-body">
              <div className="workspace-thinking-detail">
                {thinkingEntries.length ? (
                  <div className="workspace-thinking-sequence">
                    {thinkingEntries.map((entry) => (
                      <section className={`workspace-thinking-entry${entry.isLive ? " live" : ""}`} key={entry.runId}>
                        <div className="workspace-thinking-entry-meta">
                          {entry.chatModel ? <span className="workspace-thinking-meta-chip">{entry.chatModel}</span> : null}
                          {entry.durationMs ? (
                            <span className="workspace-thinking-meta-chip">{formatDurationLabel(entry.durationMs)}</span>
                          ) : null}
                          {entry.startedAt ? (
                            <span className="workspace-thinking-meta-chip">{formatInspectorTimestamp(entry.startedAt)}</span>
                          ) : null}
                          {entry.isLive ? <span className="workspace-thinking-meta-chip accent">Live</span> : null}
                        </div>
                        <pre className="workspace-thinking-text">{entry.text}</pre>
                      </section>
                    ))}
                  </div>
                ) : canToggleThinkingPanel ? (
                  <div className="workspace-thinking-placeholder">
                    Atlas has not emitted any deciding trace yet.
                  </div>
                ) : (
                  <div className="workspace-thinking-empty">No deciding trace yet for this thread.</div>
                )}
              </div>
            </div>
          </div>
        </aside>
      ) : null}
      <ResetDialog
        open={Boolean(pendingModelSwitch)}
        onOpenChange={(open) => {
          if (!open) {
            setPendingModelSwitch(null);
          }
        }}
        title={pendingModelSwitch ? `Switch to ${formatModelLabel(pendingModelSwitch.toModel)}?` : "Switch model?"}
        description={modelSwitchDialogDescription(pendingModelSwitch, providerLabel)}
        confirmLabel="Switch model"
        confirmIntent="primary"
        busyLabel="Switching..."
        onConfirm={confirmModelSwitch}
      />
    </section>
  );
}

type WorkspaceComposerProps = {
  placeholder: string;
  canStartChat: boolean;
  currentUserId: string;
  currentRunId: string | null;
  currentRunMode: string | null;
  currentThreadHasActiveRun: boolean;
  currentThreadCompactionNotice: {
    detectedContextWindow?: number | null;
    historyRepresentationTokensAfterCompaction?: number | null;
  } | null;
  contextUsage?: ThreadContextUsage;
  runDetectedContextWindow?: number | null;
  visibleHistory: ThreadMessage[];
  pendingPrompt: string;
  pendingAttachments: ImageAttachment[];
  liveAnswer: string;
  providerLabel: string;
  isStreaming: boolean;
  isCompactingContext: boolean;
  threadHasHistory: boolean;
  startRunPending: boolean;
  startManualCompactPending: boolean;
  stopRunPending: boolean;
  selectedModelSupportsImages: boolean;
  selectedModelSupportsReasoning: boolean;
  activeReasoningOption: ReasoningOption | null;
  reasoningOptions: ReasoningOption[];
  effectiveReasoningMode: ReasoningMode;
  setReasoningMode: (mode: ReasoningMode) => void;
  onSubmitPrompt: (payload: StartPromptPayload, clearDraft: () => void) => void;
  onManualCompact: () => void;
  onStopRun: () => void;
};

const WorkspaceComposer = memo(forwardRef<WorkspaceComposerHandle, WorkspaceComposerProps>(function WorkspaceComposer({
  placeholder,
  canStartChat,
  currentUserId,
  currentRunId,
  currentRunMode,
  currentThreadHasActiveRun,
  currentThreadCompactionNotice,
  contextUsage,
  runDetectedContextWindow,
  visibleHistory,
  pendingPrompt,
  pendingAttachments,
  liveAnswer,
  providerLabel,
  isStreaming,
  isCompactingContext,
  threadHasHistory,
  startRunPending,
  startManualCompactPending,
  stopRunPending,
  selectedModelSupportsImages,
  selectedModelSupportsReasoning,
  activeReasoningOption,
  reasoningOptions,
  effectiveReasoningMode,
  setReasoningMode,
  onSubmitPrompt,
  onManualCompact,
  onStopRun,
}, ref) {
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const attachmentInputRef = useRef<HTMLInputElement | null>(null);
  const attachmentMenuRef = useRef<HTMLDivElement | null>(null);
  const attachmentMenuTriggerRef = useRef<HTMLButtonElement | null>(null);
  const reasoningMenuRef = useRef<HTMLDivElement | null>(null);
  const reasoningMenuTriggerRef = useRef<HTMLButtonElement | null>(null);
  const promptInputRef = useRef<HTMLTextAreaElement | null>(null);
  const [prompt, setPrompt] = useState("");
  const [attachments, setAttachments] = useState<ImageAttachment[]>([]);
  const [isAttachmentMenuOpen, setIsAttachmentMenuOpen] = useState(false);
  const [isReasoningMenuOpen, setIsReasoningMenuOpen] = useState(false);
  const deferredPrompt = useDeferredValue(prompt);

  useImperativeHandle(ref, () => ({
    quoteMessage(content: string) {
      const quoted = buildQuotedPrompt(content);
      setPrompt((current) => (current.trim() ? `${current.trim()}\n\n${quoted}` : quoted));
      window.requestAnimationFrame(() => promptInputRef.current?.focus());
    },
    setPrompt(content: string) {
      setPrompt(content);
      window.requestAnimationFrame(() => promptInputRef.current?.focus());
    },
  }), []);

  useEffect(() => {
    if (!isAttachmentMenuOpen) {
      return undefined;
    }
    const handlePointerDown = (event: PointerEvent) => {
      if (!attachmentMenuRef.current?.contains(event.target as Node | null)) {
        setIsAttachmentMenuOpen(false);
      }
    };
    window.addEventListener("pointerdown", handlePointerDown);
    return () => window.removeEventListener("pointerdown", handlePointerDown);
  }, [isAttachmentMenuOpen]);

  useEffect(() => {
    if (!isReasoningMenuOpen) {
      return undefined;
    }
    const handlePointerDown = (event: PointerEvent) => {
      if (!reasoningMenuRef.current?.contains(event.target as Node | null)) {
        setIsReasoningMenuOpen(false);
      }
    };
    window.addEventListener("pointerdown", handlePointerDown);
    return () => window.removeEventListener("pointerdown", handlePointerDown);
  }, [isReasoningMenuOpen]);

  const contextMeter = useMemo(() => {
    return buildContextMeter({
      contextUsage,
      compactionNotice: currentThreadCompactionNotice,
      runDetectedContextWindow,
      visibleHistory,
      hasActiveRun: currentThreadHasActiveRun,
      pendingPrompt,
      pendingAttachments,
      liveAnswer,
      draftPrompt: deferredPrompt,
    });
  }, [
    contextUsage,
    currentThreadCompactionNotice?.detectedContextWindow,
    currentThreadCompactionNotice?.historyRepresentationTokensAfterCompaction,
    currentThreadHasActiveRun,
    deferredPrompt,
    liveAnswer,
    pendingAttachments,
    pendingPrompt,
    runDetectedContextWindow,
    visibleHistory,
  ]);
  const contextMeterTitle = formatContextMeterTitle(contextMeter, currentThreadHasActiveRun, providerLabel);

  const appendAttachmentsFromFiles = async (files: File[]) => {
    if (!files.length) {
      return;
    }
    const nextAttachments = await Promise.all(files.map((file) => fileToAttachment(file)));
    setAttachments((current) => [...current, ...nextAttachments]);
  };

  const handleImageSelection = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.currentTarget.files ?? []);
    await appendAttachmentsFromFiles(files);
    event.currentTarget.value = "";
    setIsAttachmentMenuOpen(false);
    window.requestAnimationFrame(() => attachmentMenuTriggerRef.current?.focus());
  };

  const handleAttachmentSelection = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.currentTarget.files ?? []);
    await appendAttachmentsFromFiles(files);
    event.currentTarget.value = "";
    setIsAttachmentMenuOpen(false);
    window.requestAnimationFrame(() => attachmentMenuTriggerRef.current?.focus());
  };

  const clearDraft = useCallback(() => {
    setPrompt("");
    setAttachments([]);
    if (imageInputRef.current) {
      imageInputRef.current.value = "";
    }
    if (attachmentInputRef.current) {
      attachmentInputRef.current.value = "";
    }
  }, []);

  const submitCurrentPrompt = useCallback(() => {
    const trimmedPrompt = prompt.trim();
    if ((!trimmedPrompt && attachments.length === 0) || isStreaming || startRunPending || !canStartChat) {
      return;
    }
    onSubmitPrompt({ prompt: trimmedPrompt, attachments }, clearDraft);
  }, [attachments, canStartChat, clearDraft, isStreaming, onSubmitPrompt, prompt, startRunPending]);

  const submitPrompt = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    submitCurrentPrompt();
  };

  const handlePromptKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.altKey && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      submitCurrentPrompt();
    }
  };

  return (
    <form className="composer composer-shell" onSubmit={submitPrompt}>
      {attachments.length ? (
        <div className="composer-attachments">
          {attachments.map((item, index) => {
            const isImage = attachmentIsImage(item);
            return (
              <div className={`composer-attachment-card${isImage ? " image" : " file"}`} key={`${item.name}-${item.data_url || index}`}>
                {isImage ? (
                  <img alt={item.name || `attachment-${index + 1}`} className="composer-attachment-image" src={item.data_url} />
                ) : (
                  <div className="composer-attachment-file-icon" aria-hidden="true">
                    <FileText size={16} />
                  </div>
                )}
                <div className="composer-attachment-copy">
                  <div className="composer-attachment-title" title={item.name || `attachment-${index + 1}`}>
                    {item.name || `attachment-${index + 1}`}
                  </div>
                  <div className="composer-attachment-meta">{formatAttachmentMeta(item)}</div>
                </div>
                <button
                  aria-label={`Remove ${item.name || "attachment"}`}
                  className="ghost-button icon-button composer-attachment-remove"
                  onClick={() => setAttachments((current) => current.filter((_, currentIndex) => currentIndex !== index))}
                  type="button"
                >
                  <X size={14} />
                </button>
              </div>
            );
          })}
        </div>
      ) : null}

      <textarea
        aria-label="Message"
        className="prompt-input"
        onChange={(event) => setPrompt(event.currentTarget.value)}
        onKeyDown={handlePromptKeyDown}
        placeholder={placeholder}
        ref={promptInputRef}
        rows={3}
        value={prompt}
      />

      <div className="composer-actions">
        <input
          accept="image/*"
          className="hidden-file-input"
          onChange={handleImageSelection}
          multiple
          ref={imageInputRef}
          type="file"
        />
        <input
          accept={DOCUMENT_FILE_ACCEPT}
          className="hidden-file-input"
          onChange={handleAttachmentSelection}
          multiple
          ref={attachmentInputRef}
          type="file"
        />

        <button
          aria-label={contextMeterTitle}
          className={`composer-context-meter composer-context-meter-${contextMeter.tone}${isCompactingContext ? " composer-context-meter-compacting" : ""}`}
          disabled={isStreaming || !currentUserId || !threadHasHistory || startManualCompactPending}
          onClick={onManualCompact}
          title={contextMeterTitle}
          type="button"
        >
          <span
            aria-hidden="true"
            className="composer-context-meter-fill"
            style={{ width: `${contextMeter.remainingPercent}%` }}
          />
          <span className={`composer-context-meter-label${isCompactingContext ? " deciding-sweep" : ""}`}>
            {isCompactingContext
              ? "Compacting..."
              : `${contextMeter.remainingPercent}% until auto-compact`}
          </span>
        </button>
        <div className="composer-send-cluster">
          <div className="composer-menu-shell" ref={attachmentMenuRef}>
            <button
              aria-controls="composer-attachment-menu"
              aria-expanded={isAttachmentMenuOpen}
              aria-haspopup="menu"
              aria-label="Add an attachment"
              className="ghost-button icon-button"
              disabled={!canStartChat || isStreaming}
              onClick={() => {
                setIsReasoningMenuOpen(false);
                setIsAttachmentMenuOpen((current) => !current);
              }}
              onKeyDown={(event) => {
                if (event.key === "ArrowDown") {
                  event.preventDefault();
                  setIsReasoningMenuOpen(false);
                  setIsAttachmentMenuOpen(true);
                  window.requestAnimationFrame(() => focusFirstMenuItem(attachmentMenuRef.current));
                }
              }}
              ref={attachmentMenuTriggerRef}
              title="Add an attachment"
              type="button"
            >
              <Plus size={16} />
            </button>
            {isAttachmentMenuOpen ? (
              <div
                aria-label="Attachment options"
                className="composer-menu"
                id="composer-attachment-menu"
                onKeyDown={(event) =>
                  handleMenuNavigation(event, attachmentMenuRef.current, () => {
                    setIsAttachmentMenuOpen(false);
                    attachmentMenuTriggerRef.current?.focus();
                  })
                }
                role="menu"
              >
                <button
                  className="composer-menu-item"
                  disabled={!selectedModelSupportsImages}
                  onClick={() => imageInputRef.current?.click()}
                  role="menuitem"
                  type="button"
                >
                  <ImagePlus size={15} />
                  <span>Add image</span>
                </button>
                <button
                  className="composer-menu-item"
                  onClick={() => attachmentInputRef.current?.click()}
                  role="menuitem"
                  type="button"
                >
                  <FileText size={15} />
                  <span>Add file</span>
                </button>
              </div>
            ) : null}
          </div>
          {selectedModelSupportsReasoning ? (
            <div className="composer-select-shell" ref={reasoningMenuRef}>
              <button
                aria-controls="composer-reasoning-menu"
                aria-expanded={isReasoningMenuOpen}
                aria-haspopup="menu"
                aria-label={`Reasoning mode: ${activeReasoningOption?.label ?? "Reasoning"}`}
                className="composer-control-pill composer-control-trigger"
                onClick={() => {
                  setIsAttachmentMenuOpen(false);
                  setIsReasoningMenuOpen((current) => !current);
                }}
                onKeyDown={(event) => {
                  if (event.key === "ArrowDown") {
                    event.preventDefault();
                    setIsAttachmentMenuOpen(false);
                    setIsReasoningMenuOpen(true);
                    window.requestAnimationFrame(() => focusFirstMenuItem(reasoningMenuRef.current));
                  }
                }}
                ref={reasoningMenuTriggerRef}
                title="Reasoning mode"
                type="button"
              >
                <span className="composer-control-trigger-main">
                  <Lightbulb size={14} />
                  <span className="composer-control-trigger-label">{activeReasoningOption?.label ?? "Reasoning"}</span>
                </span>
                <ChevronDown className={`composer-control-trigger-chevron${isReasoningMenuOpen ? " open" : ""}`} size={14} />
              </button>
              {isReasoningMenuOpen ? (
                <div
                  aria-label="Reasoning modes"
                  className="composer-select-menu"
                  id="composer-reasoning-menu"
                  onKeyDown={(event) =>
                    handleMenuNavigation(event, reasoningMenuRef.current, () => {
                      setIsReasoningMenuOpen(false);
                      reasoningMenuTriggerRef.current?.focus();
                    })
                  }
                  role="menu"
                >
                  {reasoningOptions.map((option) => (
                    <button
                      aria-checked={option.value === effectiveReasoningMode}
                      className={`composer-select-option${option.value === effectiveReasoningMode ? " active" : ""}`}
                      key={option.value}
                      onClick={() => {
                        setReasoningMode(option.value);
                        setIsReasoningMenuOpen(false);
                        window.requestAnimationFrame(() => reasoningMenuTriggerRef.current?.focus());
                      }}
                      role="menuitemradio"
                      type="button"
                    >
                      <span>{option.label}</span>
                      {option.value === effectiveReasoningMode ? <Check size={14} /> : null}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}
          <button className="primary-button" disabled={isStreaming || startRunPending || (!prompt.trim() && attachments.length === 0) || !canStartChat} type="submit">
            <Send size={16} />
            {isStreaming ? (currentRunMode === "compact" ? "Compacting..." : "Running...") : "Send"}
          </button>
        </div>
        {isStreaming ? (
          <button
            className="ghost-button stop-button"
            disabled={stopRunPending || !currentRunId}
            onClick={onStopRun}
            type="button"
          >
            <Square size={14} fill="currentColor" />
            {stopRunPending ? "Stopping..." : "Stop"}
          </button>
        ) : null}
      </div>
    </form>
  );
}));

function focusFirstMenuItem(container: HTMLElement | null) {
  container
    ?.querySelector<HTMLButtonElement>('[role^="menuitem"]:not(:disabled)')
    ?.focus();
}

function handleMenuNavigation(
  event: KeyboardEvent<HTMLElement>,
  container: HTMLElement | null,
  closeMenu: () => void,
) {
  if (event.key === "Escape") {
    event.preventDefault();
    closeMenu();
    return;
  }
  if (event.key === "Tab") {
    closeMenu();
    return;
  }
  if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
    return;
  }
  const items = Array.from(
    container?.querySelectorAll<HTMLButtonElement>('[role^="menuitem"]:not(:disabled)') ?? [],
  );
  if (!items.length) {
    return;
  }
  event.preventDefault();
  const currentIndex = items.findIndex((item) => item === document.activeElement);
  const nextIndex =
    event.key === "Home"
      ? 0
      : event.key === "End"
        ? items.length - 1
        : event.key === "ArrowDown"
          ? (currentIndex + 1 + items.length) % items.length
          : (currentIndex - 1 + items.length) % items.length;
  items[nextIndex]?.focus();
}

type ConversationTranscriptProps = {
  items: ConversationPresentationItem[];
  canBranch: boolean;
  branchPending: boolean;
  retryPending: boolean;
  isStreaming: boolean;
  copiedMessageKey: string | null;
  expandedCompactionKeys: Record<string, boolean>;
  highlightedHistoryIndex: number | null;
  onCopyMessage: (message: ConversationMessage, index: number) => Promise<void>;
  onQuoteMessage: (message: ConversationMessage) => void;
  onBranchMessage: (afterMessageCount: number) => void;
  onRetryAssistantTurn: (retryContext: RetryContext) => void;
  onToggleCompactionSummary: (key: string) => void;
  onClearCompactionNotice: () => void;
};

const ConversationTranscript = memo(function ConversationTranscript({
  items,
  canBranch,
  branchPending,
  retryPending,
  isStreaming,
  copiedMessageKey,
  expandedCompactionKeys,
  highlightedHistoryIndex,
  onCopyMessage,
  onQuoteMessage,
  onBranchMessage,
  onRetryAssistantTurn,
  onToggleCompactionSummary,
  onClearCompactionNotice,
}: ConversationTranscriptProps) {
  return (
    <>
      {items.map(({ message, branchAfterMessageCount, retryContext }, index) => {
        const canBranchMessage = isBranchableMessage(message) && branchAfterMessageCount > 0 && canBranch;

        return isContextCompactionMessage(message) ? (
          <article
            className={`message-card system compact-context-message${message.ephemeral ? " active" : ""}`}
            key={compactionMessageKey(message, index)}
            role={message.ephemeral ? "status" : undefined}
          >
            <div className="message-meta compact-context-meta">
              <span>{formatMessageRoleLabel("system")}</span>
              <span className={`status-pill subtle ${timelineSystemBadgeClass(message)}`}>
                <span className="status-dot" />
                {timelineSystemBadgeLabel(message)}
              </span>
              {message.ephemeral ? <span className="ephemeral-tag">{timelineEphemeralLabel(message)}</span> : null}
            </div>
            <div className="message-content compact-context-copy">
              <p>{formatTimelineSystemMessageText(message)}</p>
            </div>
            <div className="compact-context-actions">
              {message.threadSummary ? (
                <button
                  className="ghost-button compact-summary-toggle"
                  onClick={() => onToggleCompactionSummary(compactionMessageKey(message, index))}
                  type="button"
                >
                  {expandedCompactionKeys[compactionMessageKey(message, index)] ? "Hide summary" : "Preview summary"}
                </button>
              ) : null}
              {message.dismissible ? (
                <button
                  aria-label="Dismiss compaction notice"
                  className="ghost-button icon-button"
                  onClick={onClearCompactionNotice}
                  type="button"
                >
                  <X size={14} />
                </button>
              ) : null}
            </div>
            {message.threadSummary && expandedCompactionKeys[compactionMessageKey(message, index)] ? (
              <div className="stack-card compaction-summary-preview compact-context-summary">
                <span className="compaction-summary-preview-label">Summary snapshot at this point</span>
                <pre className="compaction-summary-preview-text">{message.threadSummary}</pre>
              </div>
            ) : null}
          </article>
        ) : isTimelineSystemMessage(message) ? (
          <article
            className={`message-card system timeline-system-message${message.ephemeral ? " active" : ""}`}
            key={compactionMessageKey(message, index)}
            role={message.ephemeral ? "status" : undefined}
          >
            <div className="timeline-system-meta">
              <span className={`status-pill subtle ${timelineSystemBadgeClass(message)}`}>
                <span className="status-dot" />
                {timelineSystemBadgeLabel(message)}
              </span>
              {message.ephemeral ? <span className="ephemeral-tag">{timelineEphemeralLabel(message)}</span> : null}
            </div>
            <p className="timeline-system-text">{formatTimelineSystemMessageText(message)}</p>
          </article>
        ) : (
          <article
            className={`message-card ${message.role}${message.historyIndex === highlightedHistoryIndex ? " search-hit-active" : ""}`}
            data-history-index={message.historyIndex}
            key={messageRenderKey(message, index)}
          >
            <div className="message-meta message-meta-row">
              <span>{formatMessageRoleLabel(message.role)}</span>
              <div className="message-actions" aria-label="Message actions">
                <button
                  className="ghost-button compact-button message-action-button"
                  onClick={() => void onCopyMessage(message, index)}
                  type="button"
                >
                  <Copy size={14} />
                  <span>{copiedMessageKey === messageActionKey(message, index, "copy") ? "Copied" : "Copy"}</span>
                </button>
                <button
                  className="ghost-button compact-button message-action-button"
                  onClick={() => onQuoteMessage(message)}
                  type="button"
                >
                  <CornerUpLeft size={14} />
                  <span>Quote</span>
                </button>
                {canBranchMessage ? (
                  <button
                    className="ghost-button compact-button message-action-button"
                    disabled={branchPending || retryPending}
                    onClick={() => onBranchMessage(branchAfterMessageCount)}
                    type="button"
                  >
                    <GitBranch size={14} />
                    <span>{branchPending ? "Branching..." : "Branch"}</span>
                  </button>
                ) : null}
                {retryContext ? (
                  <button
                    className="ghost-button compact-button message-action-button"
                    disabled={retryPending || branchPending || isStreaming}
                    onClick={() => onRetryAssistantTurn(retryContext)}
                    type="button"
                  >
                    <RotateCcw size={14} />
                    <span>{retryPending ? "Retrying..." : "Retry"}</span>
                  </button>
                ) : null}
              </div>
            </div>
            {message.attachments?.length ? (
              <div className="message-attachments">
                {message.attachments.map((item, attachmentIndex) => (
                  attachmentIsImage(item) ? (
                    <img
                      alt={item.name || `attachment-${attachmentIndex + 1}`}
                      className="message-attachment-image"
                      key={`${item.data_url}-${attachmentIndex}`}
                      src={item.data_url}
                    />
                  ) : (
                    <div className="message-attachment-file" key={`${item.name}-${attachmentIndex}`}>
                      <FileText size={16} />
                      <span>{item.name || `file-${attachmentIndex + 1}`}</span>
                    </div>
                  )
                ))}
              </div>
            ) : null}
            <MessageContent content={message.content} streaming={Boolean(message.ephemeral && message.role === "assistant")} />
          </article>
        );
      })}
    </>
  );
});

const MODEL_DEFAULT_TEMPERATURE_VALUE = "model-default";
const TEMPERATURE_OPTIONS = Array.from({ length: 21 }, (_, index) => Number((index / 10).toFixed(1)));
const DOCUMENT_FILE_ACCEPT = [
  ".txt", ".md", ".markdown", ".json", ".csv", ".pdf", ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
  ".html", ".css", ".scss", ".sass", ".sql", ".yaml", ".yml", ".xml", ".sh", ".ps1", ".java", ".c",
  ".cc", ".cpp", ".h", ".hpp", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".kts", ".toml", ".ini",
].join(",");
const BOOLEAN_REASONING_OPTIONS: Array<{ value: ReasoningMode; label: string }> = [
  { value: "off", label: "Off" },
  { value: "on", label: "On" },
];
const LEVEL_REASONING_OPTIONS: Array<{ value: ReasoningMode; label: string }> = [
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
];

function formatModelLabel(value: string) {
  return value || "Select model";
}

function modelsToStopBeforeSwitch(loadedModels: string[], targetModel: string) {
  const normalizedTarget = targetModel.trim().toLowerCase();
  if (!normalizedTarget) {
    return [];
  }
  const seen = new Set<string>();
  return loadedModels.filter((model) => {
    const normalizedModel = model.trim().toLowerCase();
    if (!normalizedModel || normalizedModel === normalizedTarget || seen.has(normalizedModel)) {
      return false;
    }
    seen.add(normalizedModel);
    return true;
  });
}

function modelSwitchDialogDescription(pending: PendingModelSwitch | null, providerLabel: string) {
  if (!pending) {
    return "";
  }
  const stoppedModels = pending.fromModels.map(formatModelLabel).join(", ");
  const warmStateCopy = pending.fromModels.length === 1 ? "that model's" : "those models'";
  return `Atlas will stop ${stoppedModels} in ${providerLabel} before starting ${formatModelLabel(pending.toModel)}. Your saved chat history stays intact, but ${warmStateCopy} warm in-memory state will be cleared.`;
}

function buildReasoningOptions(strategy: "none" | "boolean" | "levels") {
  if (strategy === "levels") {
    return LEVEL_REASONING_OPTIONS;
  }
  if (strategy === "boolean") {
    return BOOLEAN_REASONING_OPTIONS;
  }
  return [];
}

function normalizeReasoningModeForModel(
  value: ReasoningMode,
  strategy: "none" | "boolean" | "levels",
): ReasoningMode {
  if (strategy === "levels") {
    if (value === "low" || value === "medium" || value === "high") {
      return value;
    }
    return "medium";
  }
  if (strategy === "boolean") {
    return value === "off" ? "off" : "on";
  }
  return "off";
}

function readStoredTemperature(value: { temperature?: number | null } | undefined): number | null | undefined {
  if (!value || !Object.prototype.hasOwnProperty.call(value, "temperature")) {
    return undefined;
  }
  if (typeof value.temperature === "number" && Number.isFinite(value.temperature)) {
    return value.temperature;
  }
  return null;
}

function parseTemperatureValue(value: string): number | null {
  if (value === MODEL_DEFAULT_TEMPERATURE_VALUE) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Number(parsed.toFixed(1)) : null;
}

function formatTemperatureSelectValue(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return MODEL_DEFAULT_TEMPERATURE_VALUE;
  }
  return value.toFixed(1);
}

function formatTemperatureLabel(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "Model setting";
  }
  return value.toFixed(1);
}

function formatContextMeterTitle(meter: ContextMeter, hasActiveRun: boolean, providerLabel: string) {
  if (!meter.detected) {
    return `~${meter.tokensUsed.toLocaleString()} tokens estimated. Exact model window unknown until the first run. Click to compact now.`;
  }
  const sourceCopy = meter.fromServer ? `detected from ${providerLabel}` : "estimated";
  const parts = [
    `Prompt context: ~${meter.tokensUsed.toLocaleString()} of ~${meter.autoCompactBudget.toLocaleString()} auto-compact budget used.`,
    `Model window: ${meter.contextWindow.toLocaleString()} tokens (${sourceCopy}).`,
  ];
  if (meter.summaryTokens || meter.rawMessageTokens) {
    parts.push(
      `Current representation: ${meter.summaryTokens.toLocaleString()} summary tokens + ${meter.rawMessageTokens.toLocaleString()} recent raw-message tokens.`,
    );
  }
  if (meter.compactedMessageCount || meter.recentRawMessageCount || meter.messageCount) {
    parts.push(
      `Messages: ${meter.compactedMessageCount.toLocaleString()} compacted, ${meter.recentRawMessageCount.toLocaleString()} still raw, ${meter.messageCount.toLocaleString()} total.`,
    );
  }
  if (hasActiveRun && meter.liveAnswerTokens > 0) {
    parts.push(
      `Live answer is ~${meter.liveAnswerTokens.toLocaleString()} tokens and will count after it is saved; projected next-turn usage is ~${meter.projectedTokensUsed.toLocaleString()} tokens.`,
    );
  }
  parts.push("Click to compact now.");
  return parts.join(" ");
}

function isNearBottom(element: HTMLDivElement, threshold = 72) {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= threshold;
}

function formatMessageRoleLabel(role: ConversationMessage["role"]) {
  if (role === "assistant") {
    return "MODEL";
  }
  return role;
}

function isContextCompactionMessage(message: ConversationMessage) {
  return message.role === "system" && message.kind === "context_compacted";
}

function isTimelineSystemMessage(message: ConversationMessage) {
  return message.role === "system" && Boolean(message.kind);
}

function buildLiveCompactionMessage(notice: {
  runId: string;
  compactionReason?: string;
  compactedMessageCount: number;
  newlyCompactedMessageCount: number;
  threadSummary: string;
  detectedContextWindow: number;
  historyRepresentationTokensBeforeCompaction?: number;
  historyRepresentationTokensAfterCompaction?: number;
}): ConversationMessage {
  return {
    role: "system",
    kind: "context_compacted",
    content: "Context compacted",
    ephemeral: true,
    dismissible: true,
    runId: notice.runId,
    compactionReason: notice.compactionReason,
    threadSummary: notice.threadSummary,
    compactedMessageCount: notice.compactedMessageCount,
    newlyCompactedMessageCount: notice.newlyCompactedMessageCount,
    detectedContextWindow: notice.detectedContextWindow,
    historyRepresentationTokensBeforeCompaction: notice.historyRepresentationTokensBeforeCompaction,
    historyRepresentationTokensAfterCompaction: notice.historyRepresentationTokensAfterCompaction,
  };
}

function compactionMessageKey(message: ConversationMessage, index: number) {
  return message.runId || message.timestamp || `context-compacted-${index}`;
}

function formatTimelineSystemMessageText(message: ConversationMessage) {
  if (!isContextCompactionMessage(message)) {
    return message.content;
  }
  const freshCount = Math.max(
    0,
    Number(message.newlyCompactedMessageCount ?? message.compactedMessageCount ?? 0),
  );
  const compactedCount = Math.max(0, Number(message.compactedMessageCount ?? 0));
  const countForCopy = freshCount > 0 ? freshCount : compactedCount;
  const reason = String(message.compactionReason ?? "").toLowerCase();
  const lead = countForCopy > 0
    ? `${countForCopy} earlier ${countForCopy === 1 ? "message was" : "messages were"} ${
        reason === "manual" ? "manually folded" : "folded"
      } into a running summary.`
    : reason === "manual"
      ? "Earlier turns were manually folded into a running summary."
      : "Earlier turns were folded into a running summary.";
  const beforeTokens = Math.max(0, Number(message.historyRepresentationTokensBeforeCompaction ?? 0));
  const afterTokens = Math.max(0, Number(message.historyRepresentationTokensAfterCompaction ?? 0));
  const reductionCopy =
    beforeTokens > afterTokens && afterTokens > 0
      ? ` Atlas compressed the represented thread context from ${beforeTokens.toLocaleString()} to ${afterTokens.toLocaleString()} estimated tokens.`
      : "";
  const windowCopy =
    message.detectedContextWindow && message.detectedContextWindow > 0
      ? ` Model window: ${message.detectedContextWindow.toLocaleString()} tokens.`
      : "";
  return `${lead}${reductionCopy} Future turns use this summary plus the most recent raw turns.${windowCopy}`;
}

function timelineSystemBadgeLabel(message: ConversationMessage) {
  return message.kind === "context_compacted" ? "Context compacted" : "System";
}

function timelineSystemBadgeClass(message: ConversationMessage) {
  return message.kind === "context_compacted" ? "compacted" : "muted";
}

function timelineEphemeralLabel(message: ConversationMessage) {
  return message.kind === "context_compacted" ? "during this response" : "live";
}

function chatWaitingLabel(stage: string) {
  if (stage === "queued") {
    return "Queued";
  }
  if (stage === "compaction") {
    return "Compacting";
  }
  if (stage === "stopping") {
    return "Stopping";
  }
  return "Deciding";
}

function compactWaitingLabel(stage: string) {
  if (stage === "queued") {
    return "Compaction queued";
  }
  if (stage === "stopping") {
    return "Stopping compaction";
  }
  return "Compacting older context";
}

function isBranchableMessage(message: ConversationMessage) {
  return (message.role === "user" || message.role === "assistant") && !message.ephemeral && !message.kind;
}

function buildConversationPresentation(transcript: ConversationMessage[]): ConversationPresentationItem[] {
  const branchCounts: number[] = [];
  let branchCount = 0;
  let latestAssistantIndex = -1;
  for (let index = 0; index < transcript.length; index += 1) {
    const message = transcript[index];
    if (isBranchableMessage(message)) {
      branchCount += 1;
      if (message.role === "assistant") {
        latestAssistantIndex = index;
      }
    }
    branchCounts[index] = branchCount;
  }

  const retryContexts = new Map<number, RetryContext>();
  if (latestAssistantIndex >= 0) {
    for (let cursor = latestAssistantIndex - 1; cursor >= 0; cursor -= 1) {
      const candidate = transcript[cursor];
      if (candidate.role !== "user" || candidate.ephemeral || candidate.kind) {
        continue;
      }
      retryContexts.set(latestAssistantIndex, {
        afterMessageCount: branchCounts[cursor] ?? 0,
        prompt: candidate.content,
        attachments: candidate.attachments ?? [],
      });
      break;
    }
  }

  return transcript.map((message, index) => ({
    message,
    branchAfterMessageCount: branchCounts[index] ?? 0,
    retryContext: retryContexts.get(index) ?? null,
  }));
}

function buildQuotedPrompt(content: string) {
  return content
    .split(/\r?\n/)
    .map((line) => `> ${line}`)
    .join("\n");
}

function messageActionKey(message: ConversationMessage, index: number, action: string) {
  return `${action}:${message.runId || message.timestamp || message.historyIndex || index}:${message.role}`;
}

function messageRenderKey(message: ConversationMessage, index: number) {
  if (message.runId) {
    return `message:${message.runId}:${message.role}:${message.kind || "plain"}`;
  }
  if (message.timestamp) {
    return `message:${message.timestamp}:${message.role}:${message.kind || "plain"}`;
  }
  if (message.historyIndex !== undefined && message.historyIndex !== null) {
    return `message:${message.historyIndex}:${message.role}:${message.kind || "plain"}`;
  }
  return `message:${index}:${message.role}:${message.kind || "plain"}`;
}

function extractThinkingText(events: RunStatusEvent[]) {
  return events
    .filter((event) => event.type === "thinking_token")
    .map((event) => String(event.payload.text ?? ""))
    .join("");
}

function preferLongerText(primary: string, fallback: string) {
  return primary.length >= fallback.length ? primary : fallback;
}

function formatInspectorStageLabel(value?: string) {
  const normalized = String(value ?? "").trim();
  if (!normalized) {
    return "Trace";
  }
  return normalized
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatRunStatusLabel(run: RunSummary | undefined) {
  if (!run) {
    return "No run";
  }
  if (run.status === "completed") {
    return "Completed";
  }
  if (run.status === "failed") {
    return "Failed";
  }
  if (run.status === "cancelling") {
    return "Stopping";
  }
  if (run.status === "queued") {
    return "Queued";
  }
  if (run.status === "running") {
    return "Running";
  }
  return formatInspectorStageLabel(run.status);
}

function formatDurationLabel(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value) || value <= 0) {
    return "";
  }
  if (value < 1000) {
    return `${value} ms`;
  }
  return `${(value / 1000).toFixed(1)} s`;
}

function formatInspectorTimestamp(value?: string | null) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return "";
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

async function fileToAttachment(file: File): Promise<ImageAttachment> {
  const normalizedName = file.name || "attachment";
  const mediaType = file.type || fallbackMediaTypeForFile(normalizedName);
  if (attachmentIsImage({ name: normalizedName, media_type: mediaType })) {
    return {
      kind: "image",
      name: normalizedName,
      media_type: mediaType || "image/png",
      data_url: await readFileAsDataUrl(file),
      byte_size: file.size,
    };
  }

  if (isTextLikeFile(normalizedName, mediaType)) {
    return {
      kind: "file",
      name: normalizedName,
      media_type: mediaType || "text/plain",
      text_content: await file.text(),
      byte_size: file.size,
    };
  }

  return {
    kind: "file",
    name: normalizedName,
    media_type: mediaType || "application/octet-stream",
    data_url: await readFileAsDataUrl(file),
    byte_size: file.size,
  };
}

function attachmentIsImage(attachment: Pick<ImageAttachment, "kind" | "media_type" | "data_url" | "name">) {
  if (attachment.kind === "image") {
    return true;
  }
  const mediaType = String(attachment.media_type || "").toLowerCase();
  if (mediaType.startsWith("image/")) {
    return true;
  }
  return String(attachment.data_url || "").startsWith("data:image/");
}

function formatAttachmentMeta(attachment: ImageAttachment) {
  const extension = getAttachmentExtensionLabel(attachment.name);
  const typeLabel = extension || (attachmentIsImage(attachment) ? "IMAGE" : "FILE");
  const sizeLabel = formatAttachmentSize(attachment.byte_size);
  return [typeLabel, sizeLabel].filter(Boolean).join(" - ");
}

function getAttachmentExtensionLabel(name?: string) {
  const normalized = String(name || "").trim();
  if (!normalized.includes(".")) {
    return "";
  }
  const extension = normalized.split(".").pop()?.trim().toUpperCase() ?? "";
  if (!extension || extension.length > 6) {
    return "";
  }
  return extension;
}

function formatAttachmentSize(value?: number) {
  if (!value || value <= 0) {
    return "";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const precision = size >= 10 || unitIndex === 0 ? 0 : 1;
  return `${size.toFixed(precision)} ${units[unitIndex]}`;
}

function isTextLikeFile(name: string, mediaType: string) {
  const normalizedName = name.toLowerCase();
  const normalizedType = mediaType.toLowerCase();
  if (normalizedType.startsWith("text/")) {
    return true;
  }
  if (["application/json", "application/xml", "application/x-yaml", "text/x-python"].includes(normalizedType)) {
    return true;
  }
  return [
    ".txt", ".md", ".markdown", ".json", ".csv", ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".html", ".css", ".scss", ".sass", ".sql", ".yaml", ".yml", ".xml", ".sh", ".ps1", ".java", ".c",
    ".cc", ".cpp", ".h", ".hpp", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".kts", ".toml", ".ini",
  ].some((extension) => normalizedName.endsWith(extension));
}

function fallbackMediaTypeForFile(name: string) {
  const normalizedName = name.toLowerCase();
  if (normalizedName.endsWith(".pdf")) {
    return "application/pdf";
  }
  if (normalizedName.endsWith(".json")) {
    return "application/json";
  }
  if (normalizedName.endsWith(".csv")) {
    return "text/csv";
  }
  if (normalizedName.endsWith(".md") || normalizedName.endsWith(".markdown")) {
    return "text/markdown";
  }
  if (isTextLikeFile(normalizedName, "")) {
    return "text/plain";
  }
  return "application/octet-stream";
}

async function readFileAsDataUrl(file: File): Promise<string> {
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Could not read file."));
    reader.onload = () => resolve(String(reader.result || ""));
    reader.readAsDataURL(file);
  });
}
