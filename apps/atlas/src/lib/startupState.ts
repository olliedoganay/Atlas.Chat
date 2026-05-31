import type { BackendPhase } from "./backendPhase";

export type StartupStateKey =
  | "backend-starting"
  | "backend-offline"
  | "no-profile"
  | "profile-locked"
  | "models-loading"
  | "ollama-offline"
  | "no-local-models"
  | "ready-no-model"
  | "ready";

export type StartupStateTone =
  | "online"
  | "starting"
  | "offline"
  | "warning"
  | "muted";

type StartupStateOptions = {
  backendPhase: BackendPhase;
  currentUserId: string;
  currentUserLocked: boolean;
  modelCatalogLoaded: boolean;
  ollamaOnline: boolean;
  hasLocalModels: boolean;
  selectedModel: string;
  providerLabel?: string;
  selectedModelSupportsImages?: boolean;
  threadHasHistory?: boolean;
};

export type StartupState = {
  key: StartupStateKey;
  tone: StartupStateTone;
  shellLabel: string;
  idleKicker: string;
  headerSummary: string;
  idleTitle: string;
  idleDescription: string;
  composerPlaceholder: string;
  canStartChat: boolean;
};

export function resolveStartupState({
  backendPhase,
  currentUserId,
  currentUserLocked,
  modelCatalogLoaded,
  ollamaOnline,
  hasLocalModels,
  selectedModel,
  providerLabel = "Ollama",
  selectedModelSupportsImages = false,
  threadHasHistory = false,
}: StartupStateOptions): StartupState {
  if (backendPhase === "starting") {
    return {
      key: "backend-starting",
      tone: "starting",
      shellLabel: "Starting backend",
      idleKicker: "Local runtime",
      headerSummary: "Atlas is starting the local runtime.",
      idleTitle: "Starting backend",
      idleDescription: "Atlas is starting the managed local runtime. Chats, profiles, and models will appear as soon as it comes online.",
      composerPlaceholder: "Starting local runtime...",
      canStartChat: false,
    };
  }

  if (backendPhase !== "online") {
    return {
      key: "backend-offline",
      tone: "offline",
      shellLabel: "Backend offline",
      idleKicker: "Local runtime",
      headerSummary: "Local runtime offline. Restart Atlas from the sidebar to continue.",
      idleTitle: "Backend offline",
      idleDescription: "Atlas cannot load chats or models until the managed backend comes back online. Use the restart control in the sidebar when the runtime is ready.",
      composerPlaceholder: "Backend offline. Restart the local runtime to continue.",
      canStartChat: false,
    };
  }

  if (!currentUserId) {
    return {
      key: "no-profile",
      tone: "muted",
      shellLabel: "Choose a profile",
      idleKicker: "Profiles",
      headerSummary: "Choose a profile before starting the first chat.",
      idleTitle: "Choose a profile",
      idleDescription: "Choose a profile first. Atlas keeps chats, memory, and search scoped to the active profile.",
      composerPlaceholder: "Choose a profile before sending the first message.",
      canStartChat: false,
    };
  }

  if (currentUserLocked) {
    return {
      key: "profile-locked",
      tone: "warning",
      shellLabel: "Profile locked",
      idleKicker: "Security",
      headerSummary: "Unlock this profile before Atlas opens chats or starts a new one.",
      idleTitle: "Profile locked",
      idleDescription: "This profile is locked. Unlock it in Settings before Atlas loads chats, memory, and search for this workspace.",
      composerPlaceholder: "Unlock the active profile before sending the first message.",
      canStartChat: false,
    };
  }

  if (!modelCatalogLoaded) {
    return {
      key: "models-loading",
      tone: "starting",
      shellLabel: "Checking models",
      idleKicker: "Models",
      headerSummary: "Checking the local model list.",
      idleTitle: "Checking models",
      idleDescription: "Atlas is checking the local model provider before the first message.",
      composerPlaceholder: "Checking the local model list.",
      canStartChat: false,
    };
  }

  if (!ollamaOnline) {
    return {
      key: "ollama-offline",
      tone: "warning",
      shellLabel: `${providerLabel} offline`,
      idleKicker: "Provider",
      headerSummary: `Atlas is online, but ${providerLabel} is not responding yet.`,
      idleTitle: `${providerLabel} unavailable`,
      idleDescription: `Start ${providerLabel} on this machine, then refresh the local model list before sending the first message.`,
      composerPlaceholder: `Start ${providerLabel} on this machine first.`,
      canStartChat: false,
    };
  }

  if (!hasLocalModels) {
    return {
      key: "no-local-models",
      tone: "warning",
      shellLabel: "No local models",
      idleKicker: "Models",
      headerSummary: `${providerLabel} is running, but there are no local chat models available yet.`,
      idleTitle: "No local models",
      idleDescription: `Add at least one local chat model to ${providerLabel}, then refresh Atlas to use it in new chats.`,
      composerPlaceholder: "Add a local chat model before sending the first message.",
      canStartChat: false,
    };
  }

  if (!selectedModel) {
    return {
      key: "ready-no-model",
      tone: "muted",
      shellLabel: "Choose a model",
      idleKicker: "New thread",
      headerSummary: "Choose a local model before the first message.",
      idleTitle: "Choose a model",
      idleDescription: "Select a local chat model before starting this thread.",
      composerPlaceholder: "Choose a local model to start this chat.",
      canStartChat: false,
    };
  }

  return {
    key: "ready",
    tone: "online",
    shellLabel: "Ready",
    idleKicker: "New thread",
    headerSummary: threadHasHistory
      ? "This chat is locked to its original model and temperature setting."
      : "Model and temperature setting lock after the first message in this thread.",
    idleTitle: "New thread",
    idleDescription: selectedModelSupportsImages
      ? "Ask a question, upload a photo for context, or use this thread as a clean branch for a new line of thinking."
      : "Use this thread to compare ideas, condense notes, or sketch the next move.",
    composerPlaceholder: selectedModelSupportsImages
      ? "Drop a photo, a rough brief, or the first line."
      : "Start with a question, a draft, or the next move.",
    canStartChat: true,
  };
}
