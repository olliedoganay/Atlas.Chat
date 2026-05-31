import * as ScrollArea from "@radix-ui/react-scroll-area";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  Activity,
  ChevronLeft,
  ChevronRight,
  Compass,
  Copy,
  Maximize2,
  Minus,
  Pin,
  Plus,
  RotateCcw,
  Search,
  Settings,
  Workflow,
  X,
} from "lucide-react";
import {
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useState,
} from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";

import {
  duplicateThread,
  getModels,
  getStatus,
  getThreads,
  getUsers,
  resetThread,
  restartManagedBackend,
  type ThreadSummary,
  type UserSummary,
} from "../lib/api";
import { ChatSearchDialog } from "./ChatSearchDialog";
import { FirstRunWizard } from "./FirstRunWizard";
import { ProfileMenu } from "./ProfileMenu";
import { ResetDialog } from "./ResetDialog";
import { RunStreamCoordinator } from "./RunStreamCoordinator";
import { useAtlasStore } from "../store/useAtlasStore";
import { useBackendPhase } from "../lib/backendPhase";
import { resolveStartupState } from "../lib/startupState";
import { PROFILE_SETTINGS_PATH } from "../lib/settingsSections";
import { displayThreadTitle, editableThreadTitle, threadInitial } from "../lib/threadTitles";

const navigation = [
  { to: "/workspace", label: "Workspace", icon: Workflow },
  { to: "/discovery", label: "Discovery", icon: Compass },
  { to: "/advanced", label: "Diagnostics", icon: Activity },
  { to: "/settings", label: "Settings", icon: Settings },
];
const NAV_WIDTH_MIN = 200;
const NAV_WIDTH_MAX = 420;
const NAV_WIDTH_STEP = 16;
type WindowAction = "close" | "minimize" | "toggle-maximize";
type WindowResizeDirection = Parameters<ReturnType<typeof getCurrentWindow>["startResizeDragging"]>[0];
const WINDOW_RESIZE_HANDLES: Array<{ className: string; direction: WindowResizeDirection }> = [
  { className: "north", direction: "North" },
  { className: "east", direction: "East" },
  { className: "south", direction: "South" },
  { className: "west", direction: "West" },
  { className: "north-east", direction: "NorthEast" },
  { className: "north-west", direction: "NorthWest" },
  { className: "south-east", direction: "SouthEast" },
  { className: "south-west", direction: "SouthWest" },
];

export function AtlasShell() {
  const location = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const currentUserId = useAtlasStore((state) => state.currentUserId);
  const currentThreadId = useAtlasStore((state) => state.currentThreadId);
  const currentThreadTitle = useAtlasStore((state) => state.currentThreadTitle);
  const draftThreadModel = useAtlasStore((state) => state.draftThreadModel);
  const draftThreadTemperature = useAtlasStore((state) => state.draftThreadTemperature);
  const navCollapsed = useAtlasStore((state) => state.navCollapsed);
  const navWidth = useAtlasStore((state) => state.navWidth);
  const setCurrentUserId = useAtlasStore((state) => state.setCurrentUserId);
  const setCurrentThreadId = useAtlasStore((state) => state.setCurrentThreadId);
  const setCurrentThreadTitle = useAtlasStore((state) => state.setCurrentThreadTitle);
  const setDraftThreadModel = useAtlasStore((state) => state.setDraftThreadModel);
  const setDraftThreadTemperature = useAtlasStore((state) => state.setDraftThreadTemperature);
  const setSearchJumpTarget = useAtlasStore((state) => state.setSearchJumpTarget);
  const pinnedThreadKeys = useAtlasStore((state) => state.pinnedThreadKeys);
  const togglePinnedThread = useAtlasStore((state) => state.togglePinnedThread);
  const backendStartupStartedAt = useAtlasStore((state) => state.backendStartupStartedAt);
  const markBackendBooting = useAtlasStore((state) => state.markBackendBooting);
  const toggleNavCollapsed = useAtlasStore((state) => state.toggleNavCollapsed);
  const setNavWidth = useAtlasStore((state) => state.setNavWidth);
  const firstRunDismissed = useAtlasStore((state) => state.firstRunDismissed);
  const setFirstRunDismissed = useAtlasStore((state) => state.setFirstRunDismissed);
  const isWorkspaceRoute = location.pathname === "/workspace";
  const [threadToDelete, setThreadToDelete] = useState<ThreadSummary | null>(null);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [isNavResizing, setIsNavResizing] = useState(false);
  const [isWindowMaximized, setIsWindowMaximized] = useState(true);

  const {
    data: status,
    isPending: statusPending,
    isFetching: statusFetching,
  } = useQuery({
    queryKey: ["status"],
    queryFn: getStatus,
    refetchInterval: 5000,
    retry: 1,
    refetchOnWindowFocus: false,
  });
  const { data: models } = useQuery({
    queryKey: ["models"],
    queryFn: getModels,
    enabled: Boolean(status),
    staleTime: 10000,
    retry: 1,
    refetchOnWindowFocus: false,
  });
  const { data: users = [], isFetched: usersFetched } = useQuery({
    queryKey: ["users"],
    queryFn: getUsers,
    enabled: Boolean(status),
    staleTime: 5000,
    retry: 1,
    refetchOnWindowFocus: false,
  });
  const backendPhase = useBackendPhase({
    hasStatus: Boolean(status),
    isPending: statusPending,
    isFetching: statusFetching,
    bootStartedAt: backendStartupStartedAt,
  });
  const currentUserProfile = users.find((user) => user.user_id === currentUserId);
  const currentUserLocked = Boolean(currentUserProfile?.locked);
  const { data: threads = [] } = useQuery({
    queryKey: ["threads", currentUserId],
    queryFn: () => getThreads(currentUserId),
    enabled: isWorkspaceRoute && Boolean(currentUserId) && !currentUserLocked,
    staleTime: 2000,
  });

  const deleteThreadMutation = useMutation({
    mutationFn: async (thread: ThreadSummary) => resetThread(thread.thread_id, thread.user_id || currentUserId),
    onMutate: async (thread) => {
      const resolvedUserId = thread.user_id || currentUserId;
      const queryKey = ["threads", resolvedUserId] as const;
      const deletedWasCurrent = resolvedUserId === currentUserId && thread.thread_id === currentThreadId;

      await queryClient.cancelQueries({ queryKey });
      const previousThreads = queryClient.getQueryData<ThreadSummary[]>(queryKey) ?? [];
      const nextThreads = previousThreads.filter(
        (item) => !isSameThread(item, thread, resolvedUserId),
      );

      queryClient.setQueryData(queryKey, nextThreads);
      queryClient.removeQueries({ queryKey: ["thread-history", resolvedUserId, thread.thread_id] });

      if (deletedWasCurrent) {
        const remaining = nextThreads.filter((item) => !isSameThread(item, thread, resolvedUserId));
        if (remaining.length > 0) {
          setCurrentThreadId(remaining[0].thread_id);
          setCurrentThreadTitle(editableThreadTitle(remaining[0].title, remaining[0].thread_id));
          setDraftThreadModel(remaining[0].chat_model || "");
          setDraftThreadTemperature(resolveThreadTemperature(remaining[0], defaultTemperature));
        } else {
          createThread();
        }
      }
      return {
        previousThreads,
        queryKey,
        deletedWasCurrent,
        previousCurrentThreadId: currentThreadId,
        previousCurrentThreadTitle: currentThreadTitle,
        previousDraftThreadModel: draftThreadModel,
        previousDraftThreadTemperature: draftThreadTemperature,
      };
    },
    onError: (_error, _thread, context) => {
      if (context?.previousThreads) {
        queryClient.setQueryData(context.queryKey, context.previousThreads);
      }
      if (context?.deletedWasCurrent) {
        setCurrentThreadId(context.previousCurrentThreadId);
        setCurrentThreadTitle(context.previousCurrentThreadTitle);
        setDraftThreadModel(context.previousDraftThreadModel);
        setDraftThreadTemperature(context.previousDraftThreadTemperature);
      }
    },
    onSuccess: async () => {
      setThreadToDelete(null);
    },
    onSettled: async (_result, _error, thread) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["threads", thread.user_id || currentUserId] }),
        queryClient.invalidateQueries({ queryKey: ["thread-history", thread.user_id || currentUserId] }),
      ]);
    },
  });
  const duplicateThreadMutation = useMutation({
    mutationFn: async (thread: ThreadSummary) => duplicateThread(thread.thread_id, thread.user_id || currentUserId),
    onSuccess: async (thread) => {
      setCurrentThreadId(thread.thread_id);
      setCurrentThreadTitle(editableThreadTitle(thread.title, thread.thread_id));
      setDraftThreadModel(thread.chat_model || "");
      setDraftThreadTemperature(resolveThreadTemperature(thread, defaultTemperature));
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["threads", currentUserId] }),
        queryClient.invalidateQueries({ queryKey: ["thread-history", currentUserId] }),
      ]);
    },
  });

  const defaultTemperature =
    models?.default_temperature ?? status?.default_chat_temperature ?? status?.chat_temperature ?? null;
  const providerOnline = Boolean(models?.provider_online ?? models?.ollama_online);
  const hasChatModels = Boolean(models?.has_chat_models ?? models?.has_local_models);
  const providerLabel = models?.provider_label || status?.chat_provider_label || "Ollama";
  const providerBaseUrl = models?.provider_base_url || status?.chat_base_url || status?.ollama_url || "http://127.0.0.1:11434";
  const isOllamaProvider = (models?.provider || status?.chat_provider || "ollama") === "ollama";
  const startupState = resolveStartupState({
    backendPhase,
    currentUserId,
    currentUserLocked,
    modelCatalogLoaded: Boolean(models),
    ollamaOnline: providerOnline,
    hasLocalModels: hasChatModels,
    selectedModel: draftThreadModel,
    providerLabel,
  });
  const threadItems = useMemo(() => {
    const seen = new Set<string>();
    return threads.filter((thread) => {
      const key = `${thread.user_id}::${thread.thread_id}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }, [threads]);

  const displayThreadItems = useMemo(() => {
    if (!currentUserId || currentUserLocked) {
      return [];
    }
    const items = !currentThreadId || threadItems.some((item) => item.thread_id === currentThreadId)
      ? threadItems
      : [
          {
            user_id: currentUserId,
            thread_id: currentThreadId,
            title: editableThreadTitle(currentThreadTitle, currentThreadId),
            chat_model: draftThreadModel,
            temperature: draftThreadTemperature,
            last_mode: "chat",
            updated_at: new Date().toISOString(),
            last_prompt: "",
            last_run_id: "",
          },
          ...threadItems,
        ];
    return sortPinnedThreads(items, pinnedThreadKeys, currentUserId);
  }, [
    currentThreadId,
    currentThreadTitle,
    currentUserId,
    currentUserLocked,
    draftThreadModel,
    draftThreadTemperature,
    pinnedThreadKeys,
    threadItems,
  ]);

  useEffect(() => {
    if (usersFetched && currentUserId && !currentUserProfile) {
      setCurrentUserId("");
      setCurrentThreadTitle("Main");
      setCurrentThreadId("main");
    }
  }, [currentUserId, currentUserProfile, setCurrentThreadId, setCurrentThreadTitle, setCurrentUserId, usersFetched]);

  useEffect(() => {
    if (currentUserLocked) {
      setIsSearchOpen(false);
    }
  }, [currentUserLocked]);

  useEffect(() => {
    let mounted = true;
    let unlistenResize: (() => void) | undefined;
    let unlistenMove: (() => void) | undefined;

    const syncMaximizedState = () => {
      try {
        void getCurrentWindow()
          .isMaximized()
          .then((maximized) => {
            if (mounted) {
              setIsWindowMaximized(maximized);
            }
          })
          .catch(() => undefined);
      } catch {
        setIsWindowMaximized(false);
      }
    };

    try {
      const appWindow = getCurrentWindow();
      syncMaximizedState();
      void appWindow
        .onResized(() => syncMaximizedState())
        .then((unlisten) => {
          if (mounted) {
            unlistenResize = unlisten;
          } else {
            unlisten();
          }
        })
        .catch(() => undefined);
      void appWindow
        .onMoved(() => syncMaximizedState())
        .then((unlisten) => {
          if (mounted) {
            unlistenMove = unlisten;
          } else {
            unlisten();
          }
        })
        .catch(() => undefined);
    } catch {
      setIsWindowMaximized(false);
    }

    return () => {
      mounted = false;
      unlistenResize?.();
      unlistenMove?.();
    };
  }, []);

  useEffect(() => {
    if (isWorkspaceRoute && !currentThreadId && threadItems.length) {
      setCurrentThreadId(threadItems[0].thread_id);
      setCurrentThreadTitle(editableThreadTitle(threadItems[0].title, threadItems[0].thread_id));
    }
  }, [currentThreadId, isWorkspaceRoute, setCurrentThreadId, setCurrentThreadTitle, threadItems]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isWorkspaceRoute || !currentUserId || currentUserLocked) {
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setIsSearchOpen(true);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [currentUserId, currentUserLocked, isWorkspaceRoute]);

  const restartBackend = async () => {
    markBackendBooting();
    await restartManagedBackend();
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["status"] }),
      queryClient.invalidateQueries({ queryKey: ["models"] }),
      queryClient.invalidateQueries({ queryKey: ["threads"] }),
    ]);
  };

  const selectThread = (thread: ThreadSummary) => {
    setCurrentThreadId(thread.thread_id);
    setCurrentThreadTitle(editableThreadTitle(thread.title, thread.thread_id));
    setDraftThreadModel(thread.chat_model || "");
    setDraftThreadTemperature(resolveThreadTemperature(thread, defaultTemperature));
  };

  const createThread = () => {
    setCurrentThreadId(buildDraftThreadId());
    setCurrentThreadTitle("");
    setDraftThreadModel("");
    setDraftThreadTemperature(null);
  };

  const resizeNav = (value: number) => {
    setNavWidth(clamp(value, NAV_WIDTH_MIN, NAV_WIDTH_MAX));
  };

  const startNavResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (navCollapsed) {
      return;
    }
    event.preventDefault();
    event.currentTarget.focus();
    const startX = event.clientX;
    const startWidth = navWidth;
    setIsNavResizing(true);

    const handlePointerMove = (moveEvent: PointerEvent) => {
      resizeNav(startWidth + moveEvent.clientX - startX);
    };
    const stopResize = () => {
      setIsNavResizing(false);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopResize);
      window.removeEventListener("pointercancel", stopResize);
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopResize);
    window.addEventListener("pointercancel", stopResize);
  };

  const handleNavResizeKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (navCollapsed) {
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      resizeNav(navWidth - NAV_WIDTH_STEP);
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      resizeNav(navWidth + NAV_WIDTH_STEP);
    }
    if (event.key === "Home") {
      event.preventDefault();
      resizeNav(NAV_WIDTH_MIN);
    }
    if (event.key === "End") {
      event.preventDefault();
      resizeNav(NAV_WIDTH_MAX);
    }
  };

  const shellStyle = navCollapsed
    ? undefined
    : ({ "--atlas-nav-width": `${clamp(navWidth, NAV_WIDTH_MIN, NAV_WIDTH_MAX)}px` } as CSSProperties);

  const handleProfilePick = async (userId: string) => {
    setCurrentUserId(userId);
    setCurrentThreadId("main");
    setCurrentThreadTitle("Main");
    setDraftThreadModel("");
    setDraftThreadTemperature(null);
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["users"] }),
      queryClient.invalidateQueries({ queryKey: ["threads"] }),
      queryClient.invalidateQueries({ queryKey: ["thread-history"] }),
      queryClient.invalidateQueries({ queryKey: ["memories"] }),
    ]);
  };

  const handleProfileUnlock = (_userId: string) => {
    setIsSearchOpen(false);
    navigate(PROFILE_SETTINGS_PATH);
  };

  const showFirstRun =
    !firstRunDismissed &&
    usersFetched &&
    users.length === 0 &&
    backendPhase !== "starting";
  const activeNavigationLabel = navigation.find((item) => item.to === location.pathname)?.label ?? "Atlas";
  const titlebarLocation = isWorkspaceRoute
    ? `Workspace / ${displayThreadTitle(currentThreadTitle, currentThreadId, "Main")}`
    : activeNavigationLabel;

  return (
    <div className="app-frame">
      {isWindowMaximized ? null : (
        <div aria-hidden="true" className="atlas-window-resize-shell">
          {WINDOW_RESIZE_HANDLES.map(({ className, direction }) => (
            <div
              className={`atlas-window-resize-handle ${className}`}
              key={direction}
              onPointerDown={(event) => startWindowResize(event, direction)}
            />
          ))}
        </div>
      )}
      <header
        className="atlas-titlebar"
        onDoubleClick={handleTitlebarDoubleClick}
        onPointerDown={startWindowDrag}
      >
        <div className="atlas-titlebar-left">
          <img alt="" aria-hidden="true" className="atlas-titlebar-logo" src="/AtlasLogo.png" />
        </div>
        <div className="atlas-titlebar-center">
          <span className="atlas-titlebar-location">{titlebarLocation}</span>
        </div>
        <div className="atlas-titlebar-controls">
          <button aria-label="Minimize window" className="atlas-window-button" onClick={() => runWindowAction("minimize")} title="Minimize" type="button">
            <Minus size={14} strokeWidth={1.8} />
          </button>
          <button aria-label="Maximize window" className="atlas-window-button" onClick={() => runWindowAction("toggle-maximize")} title="Maximize" type="button">
            <Maximize2 size={13} strokeWidth={1.8} />
          </button>
          <button aria-label="Close window" className="atlas-window-button atlas-window-button-close" onClick={() => runWindowAction("close")} title="Close" type="button">
            <X size={15} strokeWidth={1.8} />
          </button>
        </div>
      </header>
      <div className={`app-shell ${navCollapsed ? "nav-collapsed" : ""}${isNavResizing ? " nav-resizing" : ""}`} style={shellStyle}>
        <RunStreamCoordinator />
        <aside className={`global-nav ${navCollapsed ? "collapsed" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-lockup-main">
            {navCollapsed ? <img alt="Atlas Chat" className="brand-logo" src="/AtlasLogo.png" /> : null}
            <div className="brand-copy">
              <strong>Atlas Chat</strong>
              <span>Local AI workspace</span>
            </div>
          </div>
          <button className="ghost-button icon-button nav-toggle" onClick={toggleNavCollapsed} type="button">
            {navCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
        </div>

        <div className="global-nav-main">
          <nav className="nav-links">
            {navigation.map(({ to, label, icon: Icon }) => (
              <NavLink
                className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
                key={to}
                to={to}
              >
                <Icon size={18} />
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>

          {isWorkspaceRoute ? (
            <section className="shell-threads">
              {navCollapsed ? (
                <div className="collapsed-thread-list">
                  <button
                    className="ghost-button icon-button"
                    disabled={!currentUserId || currentUserLocked}
                    onClick={() => setIsSearchOpen(true)}
                    title="Search chats"
                    type="button"
                  >
                    <Search size={16} />
                  </button>
                  <button className="primary-button icon-button" disabled={!currentUserId || currentUserLocked} onClick={createThread} type="button">
                    <Plus size={16} />
                  </button>
                  {displayThreadItems.map((thread) => (
                    <button
                      className={`collapsed-thread-button ${thread.thread_id === currentThreadId ? "active" : ""}`}
                      key={thread.thread_id}
                      onClick={() => selectThread(thread)}
                      title={displayThreadTitle(thread)}
                      type="button"
                    >
                      {threadInitial(thread)}
                    </button>
                  ))}
                </div>
              ) : (
                <>
                  <div className="workspace-section-head">
                    <div>
                      <p className="workspace-section-label">Chats</p>
                      <p className="muted-text">{formatChatCount(displayThreadItems.length)}</p>
                    </div>
                    <button className="primary-button icon-button" disabled={!currentUserId || currentUserLocked} onClick={createThread} type="button">
                      <Plus size={16} />
                    </button>
                  </div>

                  <button
                    className="search-launcher"
                    disabled={!currentUserId || currentUserLocked}
                    onClick={() => setIsSearchOpen(true)}
                    type="button"
                  >
                    <Search size={16} />
                    <span>Search chats</span>
                    <span className="search-launcher-shortcut">Ctrl+K</span>
                  </button>

                  <ScrollArea.Root className="thread-scroll shell-thread-scroll">
                    <ScrollArea.Viewport className="thread-scroll-viewport">
                      <div className="thread-list">
                        {displayThreadItems.map((thread) => {
                          const persistedThread = threadItems.some((item) => isSameThread(item, thread, currentUserId));
                          const pinned = isThreadPinned(thread, pinnedThreadKeys, currentUserId);
                          return (
                            <div
                              className={`thread-card ${thread.thread_id === currentThreadId ? "active" : ""}${pinned ? " pinned" : ""}`}
                              key={thread.thread_id}
                            >
                            {persistedThread ? (
                              <button
                                aria-label={`${pinned ? "Unpin" : "Pin"} ${displayThreadTitle(thread)}`}
                                className="ghost-button icon-button thread-card-pin"
                                onClick={() => togglePinnedThread(thread.user_id || currentUserId, thread.thread_id)}
                                title={pinned ? "Unpin chat" : "Pin chat"}
                                type="button"
                              >
                                <Pin size={14} fill={pinned ? "currentColor" : "none"} />
                              </button>
                            ) : null}
                            {persistedThread ? (
                              <button
                                aria-label={`Duplicate ${displayThreadTitle(thread)}`}
                                className="ghost-button icon-button thread-card-duplicate"
                                onClick={() => duplicateThreadMutation.mutate(thread)}
                                type="button"
                              >
                                <Copy size={14} />
                              </button>
                            ) : null}
                            {persistedThread ? (
                              <button
                                aria-label={`Delete ${displayThreadTitle(thread)}`}
                                className="ghost-button icon-button thread-card-delete"
                                onClick={() => setThreadToDelete(thread)}
                                type="button"
                              >
                                <X size={14} />
                              </button>
                            ) : null}
                            <button
                              className="thread-card-main"
                              onClick={() => selectThread(thread)}
                              type="button"
                            >
                              <div className="thread-card-top">
                                <strong>{displayThreadTitle(thread)}</strong>
                              </div>
                              <p>{thread.last_prompt || "Empty draft chat"}</p>
                              <div className="thread-card-foot">
                                <span>{thread.chat_model ? formatModelLabel(thread.chat_model) : "Select model"}</span>
                                <span>{formatDate(thread.updated_at)}</span>
                              </div>
                            </button>
                          </div>
                          );
                        })}
                        {displayThreadItems.length === 0 ? (
                          <div className="thread-empty">
                            <p>{currentUserId ? "No chats yet for this profile." : "Choose a profile in the workspace first."}</p>
                            <button className="ghost-button compact-button" disabled={!currentUserId || currentUserLocked} onClick={createThread} type="button">
                              <Plus size={15} />
                              {currentUserId ? "Create first chat" : "Create chat"}
                            </button>
                          </div>
                        ) : null}
                      </div>
                    </ScrollArea.Viewport>
                    <ScrollArea.Scrollbar className="scrollbar" orientation="vertical">
                      <ScrollArea.Thumb className="scrollbar-thumb" />
                    </ScrollArea.Scrollbar>
                  </ScrollArea.Root>
                </>
              )}
            </section>
          ) : null}
        </div>

        <div className="nav-footer">
          {!navCollapsed ? (
            <ProfileMenu
              currentUserId={currentUserId}
              onPick={(userId) => {
                void handleProfilePick(userId);
              }}
              onUnlock={handleProfileUnlock}
              users={users}
            />
          ) : null}
          <div className={`status-pill ${startupState.tone}`}>
            <span className="status-dot" />
            <span>{startupState.shellLabel}</span>
          </div>
          {startupState.key === "backend-offline" ? (
            <button className="ghost-button full-width" onClick={restartBackend} type="button">
              <RotateCcw size={16} />
              Restart backend
            </button>
          ) : null}
        </div>
        {!navCollapsed ? (
          <div
            aria-label="Resize sidebar"
            aria-orientation="vertical"
            aria-valuemax={NAV_WIDTH_MAX}
            aria-valuemin={NAV_WIDTH_MIN}
            aria-valuenow={clamp(navWidth, NAV_WIDTH_MIN, NAV_WIDTH_MAX)}
            className="nav-resize-handle"
            onKeyDown={handleNavResizeKeyDown}
            onPointerDown={startNavResize}
            role="separator"
            tabIndex={0}
            title="Drag to resize sidebar"
          />
        ) : null}
      </aside>

      <main className="main-shell">
        <section className="route-shell">
          <Outlet />
        </section>
      </main>

      <ResetDialog
        confirmLabel="Delete chat"
        description={`Delete "${threadToDelete ? displayThreadTitle(threadToDelete) : ""}", its thread history, traces, and thread-linked learned state?`}
        onConfirm={async () => {
          if (!threadToDelete) {
            return;
          }
          await deleteThreadMutation.mutateAsync(threadToDelete);
        }}
        onOpenChange={(open) => setThreadToDelete(open ? threadToDelete : null)}
        open={Boolean(threadToDelete)}
        title="Delete chat"
      />
      {showFirstRun ? (
        <FirstRunWizard
          embedModel={status?.embed_model}
          hasLocalModels={hasChatModels}
          isOllamaProvider={isOllamaProvider}
          ollamaOnline={providerOnline}
          onDismiss={() => setFirstRunDismissed(true)}
          onProfileCreated={async (userId) => {
            queryClient.setQueryData<UserSummary[]>(["users"], (current = []) => {
              if (current.some((user) => user.user_id === userId)) {
                return current;
              }
              return [
                ...current,
                {
                  user_id: userId,
                  protection: "passwordless",
                  locked: false,
                  updated_at: new Date().toISOString(),
                },
              ];
            });
            await handleProfilePick(userId);
          }}
          providerBaseUrl={providerBaseUrl}
          providerLabel={providerLabel}
        />
      ) : null}
      <ChatSearchDialog
        currentThreadId={currentThreadId}
        currentThreadTitle={currentThreadTitle}
        currentUserId={currentUserId}
        onOpenChange={setIsSearchOpen}
        onPick={(result, query, meta) => {
          const existingThread = threadItems.find((item) => item.thread_id === result.thread_id);
          const targetThread: ThreadSummary = existingThread ?? {
            user_id: currentUserId,
            thread_id: result.thread_id,
            title: editableThreadTitle(result.thread_title, result.thread_id),
            chat_model: result.chat_model || "",
            temperature: null,
            last_mode: "chat",
            updated_at: result.updated_at,
            last_prompt: "",
            last_run_id: "",
          };
          selectThread(targetThread);
          setSearchJumpTarget({
            userId: currentUserId,
            threadId: result.thread_id,
            historyIndex: typeof result.history_index === "number" ? result.history_index : null,
            historyIndices: meta?.historyIndices,
            activePosition: meta?.activePosition,
            query,
          });
          setIsSearchOpen(false);
        }}
        open={isSearchOpen}
      />
      </div>
    </div>
  );
}

function runWindowAction(action: WindowAction) {
  try {
    const appWindow = getCurrentWindow();
    const command =
      action === "minimize"
        ? appWindow.minimize()
        : action === "toggle-maximize"
          ? appWindow.toggleMaximize()
          : appWindow.close();
    void command.catch(() => undefined);
  } catch {
    // The desktop frame is also rendered during browser-based Vite tests.
  }
}

function startWindowDrag(event: ReactPointerEvent<HTMLElement>) {
  if (event.button !== 0 || event.detail > 1 || isWindowControlTarget(event.target)) {
    return;
  }
  event.preventDefault();
  try {
    void getCurrentWindow().startDragging().catch(() => undefined);
  } catch {
    // The desktop frame is also rendered during browser-based Vite tests.
  }
}

function startWindowResize(event: ReactPointerEvent<HTMLDivElement>, direction: WindowResizeDirection) {
  if (event.button !== 0) {
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  try {
    const appWindow = getCurrentWindow();
    void appWindow
      .isMaximized()
      .then((maximized) => (maximized ? undefined : appWindow.startResizeDragging(direction)))
      .catch(() => undefined);
  } catch {
    // The desktop frame is also rendered during browser-based Vite tests.
  }
}

function handleTitlebarDoubleClick(event: ReactMouseEvent<HTMLElement>) {
  if (isWindowControlTarget(event.target)) {
    return;
  }
  event.preventDefault();
  runWindowAction("toggle-maximize");
}

function isWindowControlTarget(target: EventTarget | null) {
  return target instanceof HTMLElement && Boolean(target.closest("button, a, input, select, textarea, [role='button']"));
}

function formatDate(value?: string) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return "";
  }
  const today = new Date();
  const dateOnly = new Date(date.getFullYear(), date.getMonth(), date.getDate()).valueOf();
  const todayOnly = new Date(today.getFullYear(), today.getMonth(), today.getDate()).valueOf();
  const dayDelta = Math.round((todayOnly - dateOnly) / 86400000);
  if (dayDelta === 0) {
    return "Today";
  }
  if (dayDelta === 1) {
    return "Yesterday";
  }
  if (dayDelta > 1 && dayDelta < 7) {
    return new Intl.DateTimeFormat(undefined, { weekday: "short" }).format(date);
  }
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(date);
}

function formatModelLabel(value: string) {
  return value || "Select model";
}

function formatChatCount(count: number) {
  return `${count} ${count === 1 ? "chat" : "chats"}`;
}

function resolveThreadTemperature(thread: ThreadSummary | null | undefined, fallback: number | null): number | null {
  if (!thread || !Object.prototype.hasOwnProperty.call(thread, "temperature")) {
    return fallback;
  }
  return typeof thread.temperature === "number" && Number.isFinite(thread.temperature) ? thread.temperature : null;
}

function isSameThread(thread: ThreadSummary, target: ThreadSummary, fallbackUserId: string) {
  const threadUserId = thread.user_id || fallbackUserId;
  const targetUserId = target.user_id || fallbackUserId;
  return threadUserId === targetUserId && thread.thread_id === target.thread_id;
}

function threadPinKey(thread: ThreadSummary, fallbackUserId: string) {
  return `${thread.user_id || fallbackUserId}::${thread.thread_id}`;
}

function isThreadPinned(thread: ThreadSummary, pinnedThreadKeys: string[], fallbackUserId: string) {
  return pinnedThreadKeys.includes(threadPinKey(thread, fallbackUserId));
}

function sortPinnedThreads(threads: ThreadSummary[], pinnedThreadKeys: string[], fallbackUserId: string) {
  return [...threads].sort((left, right) => {
    const leftPinned = isThreadPinned(left, pinnedThreadKeys, fallbackUserId);
    const rightPinned = isThreadPinned(right, pinnedThreadKeys, fallbackUserId);
    if (leftPinned !== rightPinned) {
      return leftPinned ? -1 : 1;
    }
    return 0;
  });
}

function clamp(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, Math.round(value)));
}

function buildDraftThreadId() {
  const timestamp = new Date().toISOString().replace(/[:.T]/g, "-").replace("Z", "");
  const suffix = typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID().slice(0, 6)
    : Math.random().toString(36).slice(2, 8);
  return `atlas-${timestamp}-${suffix}`;
}
