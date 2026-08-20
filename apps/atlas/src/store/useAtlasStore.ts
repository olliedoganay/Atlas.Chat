import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import type { ImageAttachment, ReasoningMode, RunMode, ThemeMode } from "../lib/api";
import type { RecoverableRun } from "../lib/runRecovery";

type CompactionNotice = {
  runId: string;
  userId: string;
  threadId: string;
  compactionReason?: string;
  compactedMessageCount: number;
  newlyCompactedMessageCount: number;
  threadSummary: string;
  detectedContextWindow: number;
  historyRepresentationTokensBeforeCompaction?: number;
  historyRepresentationTokensAfterCompaction?: number;
};

type SearchJumpTarget = {
  userId: string;
  threadId: string;
  historyIndex: number | null;
  historyIndices?: number[];
  activePosition?: number;
  query: string;
};

type AtlasState = {
  theme: ThemeMode;
  currentUserId: string;
  currentThreadId: string;
  currentThreadTitle: string;
  draftThreadModel: string;
  draftThreadTemperature: number | null;
  reasoningMode: ReasoningMode;
  crossChatMemoryEnabled: boolean;
  autoCompactLongChats: boolean;
  crtScanlines: boolean;
  navCollapsed: boolean;
  navWidth: number;
  settingsSidebarCollapsed: boolean;
  currentRunId: string | null;
  currentRunMode: RunMode | null;
  activeRunUserId: string | null;
  activeRunThreadId: string | null;
  currentStage: string;
  pendingPrompt: string;
  pendingAttachments: ImageAttachment[];
  liveThinking: string;
  liveAnswer: string;
  liveError: string;
  compactionNotice: CompactionNotice | null;
  searchJumpTarget: SearchJumpTarget | null;
  recentSearchQueries: string[];
  pinnedThreadKeys: string[];
  backendStartupStartedAt: number;
  isStreaming: boolean;
  firstRunDismissed: boolean;
  setFirstRunDismissed: (value: boolean) => void;
  setTheme: (theme: ThemeMode) => void;
  setCurrentUserId: (value: string) => void;
  setCurrentThreadId: (value: string) => void;
  setCurrentThreadTitle: (value: string) => void;
  setDraftThreadModel: (value: string) => void;
  setDraftThreadTemperature: (value: number | null) => void;
  setReasoningMode: (value: ReasoningMode) => void;
  setCrossChatMemoryEnabled: (value: boolean) => void;
  setAutoCompactLongChats: (value: boolean) => void;
  setCrtScanlines: (value: boolean) => void;
  toggleNavCollapsed: () => void;
  setNavWidth: (value: number) => void;
  toggleSettingsSidebarCollapsed: () => void;
  beginRun: (
    runId: string,
    mode: RunMode,
    prompt: string,
    userId: string,
    threadId: string,
    attachments?: ImageAttachment[],
  ) => void;
  recoverRun: (run: RecoverableRun) => boolean;
  appendThinking: (text: string) => void;
  appendToken: (text: string) => void;
  setStage: (stage: string) => void;
  restoreRunStage: (runId: string, stage: string) => boolean;
  completeRun: () => void;
  cancelRun: () => void;
  failRun: (message: string, userId?: string, threadId?: string) => void;
  showCompactionNotice: (notice: CompactionNotice) => void;
  clearCompactionNotice: () => void;
  setSearchJumpTarget: (target: SearchJumpTarget | null) => void;
  stepSearchJumpTarget: (delta: number) => void;
  clearSearchJumpTarget: () => void;
  addRecentSearchQuery: (query: string) => void;
  clearRecentSearchQueries: () => void;
  togglePinnedThread: (userId: string, threadId: string) => void;
  clearProfileState: (userId: string) => void;
  markBackendBooting: () => void;
  clearLiveRun: () => void;
  resetAfterDataWipe: () => void;
};

const defaultTheme = "dark";

export const useAtlasStore = create<AtlasState>()(
  persist(
    (set) => ({
      theme: defaultTheme,
      currentUserId: "",
      currentThreadId: "main",
      currentThreadTitle: "Main",
      draftThreadModel: "",
      draftThreadTemperature: null,
      reasoningMode: "on",
      crossChatMemoryEnabled: true,
      autoCompactLongChats: true,
      crtScanlines: true,
      navCollapsed: false,
      navWidth: 200,
      settingsSidebarCollapsed: false,
      currentRunId: null,
      currentRunMode: null,
      activeRunUserId: null,
      activeRunThreadId: null,
      currentStage: "idle",
      pendingPrompt: "",
      pendingAttachments: [],
      liveThinking: "",
      liveAnswer: "",
      liveError: "",
      compactionNotice: null,
      searchJumpTarget: null,
      recentSearchQueries: [],
      pinnedThreadKeys: [],
      backendStartupStartedAt: Date.now(),
      isStreaming: false,
      firstRunDismissed: false,
      setFirstRunDismissed: (value) => set({ firstRunDismissed: value }),
      setTheme: (theme) => set({ theme }),
      setCurrentUserId: (value) => set({ currentUserId: value }),
      setCurrentThreadId: (value) => set({ currentThreadId: value }),
      setCurrentThreadTitle: (value) => set({ currentThreadTitle: value }),
      setDraftThreadModel: (value) => set({ draftThreadModel: value }),
      setDraftThreadTemperature: (value) => set({ draftThreadTemperature: value }),
      setReasoningMode: (value) => set({ reasoningMode: value }),
      setCrossChatMemoryEnabled: (value) => set({ crossChatMemoryEnabled: value }),
      setAutoCompactLongChats: (value) => set({ autoCompactLongChats: value }),
      setCrtScanlines: (value) => set({ crtScanlines: value }),
      toggleNavCollapsed: () => set((state) => ({ navCollapsed: !state.navCollapsed })),
      setNavWidth: (value) => set({ navWidth: value }),
      toggleSettingsSidebarCollapsed: () =>
        set((state) => ({ settingsSidebarCollapsed: !state.settingsSidebarCollapsed })),
      beginRun: (runId, mode, prompt, userId, threadId, attachments = []) =>
        set({
          currentRunId: runId,
          currentRunMode: mode,
          activeRunUserId: userId,
          activeRunThreadId: threadId,
          pendingPrompt: prompt,
          pendingAttachments: attachments,
          currentStage: "starting",
          liveThinking: "",
          liveAnswer: "",
          liveError: "",
          compactionNotice: null,
          isStreaming: true,
        }),
      recoverRun: (run) => {
        let recovered = false;
        set((state) => {
          if (
            state.isStreaming ||
            state.currentRunId ||
            state.currentUserId !== run.userId ||
            state.currentThreadId !== run.threadId
          ) {
            return {};
          }
          recovered = true;
          return {
            currentRunId: run.runId,
            currentRunMode: run.mode,
            activeRunUserId: run.userId,
            activeRunThreadId: run.threadId,
            pendingPrompt: run.prompt,
            pendingAttachments: [],
            currentStage: run.stage,
            liveThinking: "",
            liveAnswer: "",
            liveError: "",
            compactionNotice: null,
            isStreaming: true,
          };
        });
        return recovered;
      },
      appendThinking: (text) => set((state) => ({ liveThinking: `${state.liveThinking}${text}` })),
      appendToken: (text) => set((state) => ({ liveAnswer: `${state.liveAnswer}${text}` })),
      setStage: (stage) => set({ currentStage: stage }),
      restoreRunStage: (runId, stage) => {
        let restored = false;
        set((state) => {
          if (!state.isStreaming || state.currentRunId !== runId) {
            return {};
          }
          restored = true;
          return { currentStage: stage };
        });
        return restored;
      },
      completeRun: () =>
        set({
          currentRunId: null,
          currentRunMode: null,
          activeRunUserId: null,
          activeRunThreadId: null,
          isStreaming: false,
          pendingPrompt: "",
          pendingAttachments: [],
          liveThinking: "",
          liveAnswer: "",
          liveError: "",
          compactionNotice: null,
          currentStage: "completed",
        }),
      cancelRun: () =>
        set({
          currentRunId: null,
          currentRunMode: null,
          activeRunUserId: null,
          activeRunThreadId: null,
          isStreaming: false,
          pendingPrompt: "",
          pendingAttachments: [],
          liveThinking: "",
          liveAnswer: "",
          liveError: "",
          compactionNotice: null,
          currentStage: "cancelled",
        }),
      failRun: (message, userId, threadId) =>
        set((state) => ({
          currentRunId: null,
          currentRunMode: null,
          activeRunUserId: userId ?? state.activeRunUserId,
          activeRunThreadId: threadId ?? state.activeRunThreadId,
          isStreaming: false,
          pendingPrompt: "",
          pendingAttachments: [],
          liveThinking: "",
          liveAnswer: "",
          liveError: message,
          compactionNotice: null,
          currentStage: "failed",
        })),
      showCompactionNotice: (notice) => set({ compactionNotice: notice }),
      clearCompactionNotice: () => set({ compactionNotice: null }),
      setSearchJumpTarget: (target) => set({ searchJumpTarget: target }),
      stepSearchJumpTarget: (delta) =>
        set((state) => {
          const current = state.searchJumpTarget;
          if (!current?.historyIndices?.length) {
            return {};
          }
          const maxIndex = current.historyIndices.length - 1;
          const currentPosition = Math.max(0, Math.min(maxIndex, current.activePosition ?? 0));
          const nextPosition = Math.max(0, Math.min(maxIndex, currentPosition + delta));
          return {
            searchJumpTarget: {
              ...current,
              activePosition: nextPosition,
              historyIndex: current.historyIndices[nextPosition] ?? current.historyIndex ?? null,
            },
          };
        }),
      clearSearchJumpTarget: () => set({ searchJumpTarget: null }),
      addRecentSearchQuery: (query) =>
        set((state) => {
          const normalized = query.trim();
          if (!normalized) {
            return {};
          }
          const deduped = [
            normalized,
            ...state.recentSearchQueries.filter(
              (item) => item.toLocaleLowerCase() !== normalized.toLocaleLowerCase(),
            ),
          ].slice(0, 6);
          return { recentSearchQueries: deduped };
        }),
      clearRecentSearchQueries: () => set({ recentSearchQueries: [] }),
      togglePinnedThread: (userId, threadId) =>
        set((state) => {
          const key = threadStorageKey(userId, threadId);
          if (!key) {
            return {};
          }
          const existing = new Set(state.pinnedThreadKeys);
          if (existing.has(key)) {
            existing.delete(key);
          } else {
            existing.add(key);
          }
          return { pinnedThreadKeys: Array.from(existing).slice(0, 100) };
        }),
      clearProfileState: (userId) =>
        set((state) => {
          const resolvedUserId = userId.trim();
          if (!resolvedUserId) {
            return {};
          }
          const pinnedThreadPrefix = `${resolvedUserId}::`;
          const nextPinnedThreadKeys = state.pinnedThreadKeys.filter(
            (key) => !key.startsWith(pinnedThreadPrefix),
          );
          if (state.currentUserId !== resolvedUserId) {
            return {
              pinnedThreadKeys: nextPinnedThreadKeys,
              searchJumpTarget:
                state.searchJumpTarget?.userId === resolvedUserId ? null : state.searchJumpTarget,
              compactionNotice:
                state.compactionNotice?.userId === resolvedUserId ? null : state.compactionNotice,
            };
          }
          return {
            currentUserId: "",
            currentThreadId: "main",
            currentThreadTitle: "Main",
            draftThreadModel: "",
            draftThreadTemperature: null,
            currentRunId: null,
            currentRunMode: null,
            activeRunUserId: null,
            activeRunThreadId: null,
            currentStage: "idle",
            pendingPrompt: "",
            pendingAttachments: [],
            liveThinking: "",
            liveAnswer: "",
            liveError: "",
            compactionNotice: null,
            searchJumpTarget: null,
            pinnedThreadKeys: nextPinnedThreadKeys,
            isStreaming: false,
          };
        }),
      markBackendBooting: () => set({ backendStartupStartedAt: Date.now() }),
      clearLiveRun: () =>
        set({
          currentRunId: null,
          currentRunMode: null,
          activeRunUserId: null,
          activeRunThreadId: null,
          currentStage: "idle",
          pendingPrompt: "",
          pendingAttachments: [],
          liveThinking: "",
          liveAnswer: "",
          liveError: "",
          compactionNotice: null,
          searchJumpTarget: null,
          isStreaming: false,
        }),
      resetAfterDataWipe: () =>
        set({
          currentUserId: "",
          currentThreadId: "main",
          currentThreadTitle: "Main",
          draftThreadModel: "",
          draftThreadTemperature: null,
          currentRunId: null,
          currentRunMode: null,
          activeRunUserId: null,
          activeRunThreadId: null,
          currentStage: "idle",
          pendingPrompt: "",
          pendingAttachments: [],
          liveThinking: "",
          liveAnswer: "",
          liveError: "",
          compactionNotice: null,
          searchJumpTarget: null,
          recentSearchQueries: [],
          pinnedThreadKeys: [],
          isStreaming: false,
        }),
    }),
    {
      name: "atlas-ui-state",
      version: 12,
      migrate: (persistedState, version) => {
        if (!persistedState || typeof persistedState !== "object") {
          return persistedState as AtlasState;
        }
        const state = persistedState as Partial<AtlasState>;
        const migrated: Partial<AtlasState> = { ...state };
        if (version < 2) {
          migrated.draftThreadTemperature = null;
        }
        if (version < 3) {
          migrated.navCollapsed = false;
          migrated.settingsSidebarCollapsed = false;
        }
        if (version < 4) {
          migrated.currentUserId = typeof migrated.currentUserId === "string" ? migrated.currentUserId : "";
        }
        if (version < 6) {
          migrated.recentSearchQueries = [];
        }
        if (version < 7) {
          migrated.reasoningMode = "on";
        }
        if (version < 10) {
          migrated.navWidth = 200;
        }
        if (version < 11) {
          migrated.firstRunDismissed = false;
        }
        if (version < 12) {
          migrated.pinnedThreadKeys = [];
        }
        if (migrated.currentThreadTitle === "main") {
          migrated.currentThreadTitle = "Main";
        }
        return migrated as AtlasState;
      },
      partialize: (state) => ({
        theme: state.theme,
        currentUserId: state.currentUserId,
        currentThreadId: state.currentThreadId,
        currentThreadTitle: state.currentThreadTitle,
        draftThreadModel: state.draftThreadModel,
        draftThreadTemperature: state.draftThreadTemperature,
        reasoningMode: state.reasoningMode,
        crossChatMemoryEnabled: state.crossChatMemoryEnabled,
        autoCompactLongChats: state.autoCompactLongChats,
        crtScanlines: state.crtScanlines,
        navCollapsed: state.navCollapsed,
        navWidth: state.navWidth,
        settingsSidebarCollapsed: state.settingsSidebarCollapsed,
        recentSearchQueries: state.recentSearchQueries,
        pinnedThreadKeys: state.pinnedThreadKeys,
        firstRunDismissed: state.firstRunDismissed,
      }),
      storage: createJSONStorage(() => localStorage),
    },
  ),
);

function threadStorageKey(userId: string, threadId: string) {
  const resolvedUserId = userId.trim();
  const resolvedThreadId = threadId.trim();
  if (!resolvedUserId || !resolvedThreadId) {
    return "";
  }
  return `${resolvedUserId}::${resolvedThreadId}`;
}
