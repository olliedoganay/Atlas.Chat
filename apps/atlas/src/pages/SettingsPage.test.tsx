import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const apiMocks = vi.hoisted(() => ({
  clearProviderApiKey: vi.fn(),
  getAppDiagnostics: vi.fn(),
  getModels: vi.fn(),
  getMemories: vi.fn(),
  getProviderSettings: vi.fn(),
  getRunnerStatus: vi.fn(),
  getStatus: vi.fn(),
  getUsers: vi.fn(),
  deleteUser: vi.fn(),
  lockUser: vi.fn(),
  restartManagedBackend: vi.fn(),
  saveProviderSettings: vi.fn(),
}));

vi.mock("../lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("../lib/api")>();
  return {
    ...original,
    ...apiMocks,
  };
});

import {
  PROVIDER_BUSY_MESSAGE,
  PROVIDER_RUNNER_BUSY_MESSAGE,
} from "../lib/providerSettingsSafety";
import { useAtlasStore } from "../store/useAtlasStore";
import { SettingsPage } from "./SettingsPage";

const initialState = useAtlasStore.getInitialState();

describe("SettingsPage provider safety", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useAtlasStore.setState(initialState, true);
    apiMocks.getAppDiagnostics.mockResolvedValue(null);
    apiMocks.getModels.mockResolvedValue({
      models: [],
      model_details: [],
      context_window_presets: [],
      ollama_context_window: {},
      provider_online: true,
    });
    apiMocks.getProviderSettings.mockResolvedValue(providerSettings());
    apiMocks.getRunnerStatus.mockResolvedValue(runnerStatus());
    apiMocks.getMemories.mockResolvedValue([]);
    apiMocks.getStatus.mockResolvedValue({ busy: false, security: {} });
    apiMocks.getUsers.mockResolvedValue([]);
    apiMocks.deleteUser.mockResolvedValue({ status: "ok", user_id: "other" });
    apiMocks.lockUser.mockResolvedValue({
      user_id: "ollie",
      protection: "password",
      locked: true,
    });
    apiMocks.restartManagedBackend.mockResolvedValue(undefined);
    apiMocks.clearProviderApiKey.mockResolvedValue({ status: "ok" });
    apiMocks.saveProviderSettings.mockResolvedValue(providerSettings({
      provider: "lm-studio",
      base_url: "http://127.0.0.1:1234",
      has_api_key: false,
    }));
  });

  afterEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    useAtlasStore.setState(initialState, true);
    document.body.innerHTML = "";
  });

  it("keeps pristine save disabled and does not reuse a saved key after provider selection changes", async () => {
    const rendered = renderSettings();
    await waitForSettings();
    const saveButton = findButton(rendered.container, "Save & restart");
    expect(saveButton?.disabled).toBe(true);

    const keyInput = rendered.container.querySelector<HTMLInputElement>('input[aria-label="Local provider API key"]');
    const providerSelect = rendered.container.querySelector<HTMLSelectElement>('select[aria-label="Model provider"]');
    expect(keyInput).not.toBeNull();
    expect(providerSelect).not.toBeNull();
    act(() => setInputValue(keyInput as HTMLInputElement, "staged-secret"));
    expect(keyInput?.value).toBe("staged-secret");

    act(() => setSelectValue(providerSelect as HTMLSelectElement, "lm-studio"));
    expect(keyInput?.value).toBe("");
    expect(providerSelect?.value).toBe("lm-studio");
    expect(saveButton?.disabled).toBe(false);

    await act(async () => {
      saveButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    await vi.waitFor(() => expect(apiMocks.saveProviderSettings).toHaveBeenCalledTimes(1));
    expect(apiMocks.saveProviderSettings).toHaveBeenCalledWith({
      provider: "lm-studio",
      base_url: "http://127.0.0.1:1234",
      api_key: undefined,
      preserve_existing_key: false,
    });
    unmountSettings(rendered);
  });

  it("blocks save and key clearing while a backend run is active", async () => {
    apiMocks.getStatus.mockResolvedValue({ busy: true, security: {} });
    const rendered = renderSettings();
    await waitForSettings();

    await act(async () => {
      await vi.waitFor(() => expect(rendered.container.textContent).toContain(PROVIDER_BUSY_MESSAGE));
    });
    expect(findButton(rendered.container, "Save & restart")?.disabled).toBe(true);
    expect(findButton(rendered.container, "Clear key")?.disabled).toBe(true);
    unmountSettings(rendered);
  });

  it("blocks provider actions while chat is idle but the separate runner is busy", async () => {
    apiMocks.getStatus.mockResolvedValue({ busy: false, security: {} });
    apiMocks.getRunnerStatus.mockResolvedValue(runnerStatus({
      busy: true,
      active_runs: 1,
    }));
    const rendered = renderSettings();
    await waitForSettings();

    const providerSelect = rendered.container.querySelector<HTMLSelectElement>('select[aria-label="Model provider"]');
    expect(providerSelect).not.toBeNull();
    act(() => setSelectValue(providerSelect as HTMLSelectElement, "lm-studio"));
    await act(async () => {
      await vi.waitFor(() => expect(rendered.container.textContent).toContain(PROVIDER_RUNNER_BUSY_MESSAGE));
    });

    expect(findButton(rendered.container, "Save & restart")?.disabled).toBe(true);
    expect(findButton(rendered.container, "Clear key")?.disabled).toBe(true);
    expect(apiMocks.saveProviderSettings).not.toHaveBeenCalled();
    expect(apiMocks.clearProviderApiKey).not.toHaveBeenCalled();
    unmountSettings(rendered);
  });

  it("rechecks runner activity before saving a provider even when the displayed status was idle", async () => {
    const rendered = renderSettings();
    await waitForSettings();
    await vi.waitFor(() => expect(apiMocks.getRunnerStatus).toHaveBeenCalled());

    const providerSelect = rendered.container.querySelector<HTMLSelectElement>('select[aria-label="Model provider"]');
    const saveButton = findButton(rendered.container, "Save & restart");
    expect(providerSelect).not.toBeNull();
    act(() => setSelectValue(providerSelect as HTMLSelectElement, "lm-studio"));
    expect(saveButton?.disabled).toBe(false);
    apiMocks.getRunnerStatus.mockResolvedValue(runnerStatus({
      busy: true,
      active_runs: 1,
    }));

    await act(async () => {
      saveButton?.click();
      await vi.waitFor(() => expect(rendered.container.textContent).toContain(PROVIDER_RUNNER_BUSY_MESSAGE));
    });

    expect(apiMocks.saveProviderSettings).not.toHaveBeenCalled();
    expect(apiMocks.restartManagedBackend).not.toHaveBeenCalled();
    unmountSettings(rendered);
  });

  it("rechecks runner activity before clearing a provider key", async () => {
    const rendered = renderSettings();
    await waitForSettings();
    await vi.waitFor(() => expect(apiMocks.getRunnerStatus).toHaveBeenCalled());

    const clearButton = findButton(rendered.container, "Clear key");
    expect(clearButton?.disabled).toBe(false);
    apiMocks.getRunnerStatus.mockResolvedValue(runnerStatus({
      busy: true,
      runtime_preparing: true,
    }));

    await act(async () => {
      clearButton?.click();
      await vi.waitFor(() => expect(rendered.container.textContent).toContain(PROVIDER_RUNNER_BUSY_MESSAGE));
    });

    expect(apiMocks.clearProviderApiKey).not.toHaveBeenCalled();
    expect(apiMocks.restartManagedBackend).not.toHaveBeenCalled();
    unmountSettings(rendered);
  });

  it("clears persisted recent searches after successfully locking a profile", async () => {
    apiMocks.getUsers.mockResolvedValue([
      { user_id: "ollie", protection: "password", locked: false },
      { user_id: "other", protection: "passwordless", locked: false },
    ]);
    useAtlasStore.setState({
      currentUserId: "ollie",
      currentThreadId: "secret-thread",
      currentThreadTitle: "Secret customer title",
      draftThreadModel: "private-model",
      draftThreadTemperature: 1.5,
      recentSearchQueries: ["private search"],
      pinnedThreadKeys: ["ollie::secret-thread", "other::public-thread"],
    });
    const rendered = renderSettings("/settings?section=profiles");
    await act(async () => {
      await vi.waitFor(() => expect(findProfileButton(rendered.container, "ollie", "Lock")).not.toBeNull());
    });
    const lockButton = findProfileButton(rendered.container, "ollie", "Lock");
    expect(lockButton?.disabled).toBe(false);
    await act(async () => {
      lockButton?.click();
      await vi.waitFor(() => expect(apiMocks.lockUser).toHaveBeenCalledWith("ollie"));
    });

    expect(useAtlasStore.getState()).toMatchObject({
      currentUserId: "",
      currentThreadId: "main",
      currentThreadTitle: "Main",
      draftThreadModel: "",
      draftThreadTemperature: null,
      recentSearchQueries: [],
      pinnedThreadKeys: ["other::public-thread"],
    });
    const persistedState = JSON.parse(window.localStorage.getItem("atlas-ui-state") ?? "{}").state;
    expect(persistedState).toMatchObject({
      currentUserId: "",
      currentThreadId: "main",
      currentThreadTitle: "Main",
      draftThreadModel: "",
      pinnedThreadKeys: ["other::public-thread"],
    });
    expect(window.localStorage.getItem("atlas-ui-state")).not.toContain("Secret customer title");
    expect(findProfileButton(rendered.container, "other", "Use profile")?.disabled).toBe(false);
    unmountSettings(rendered);
  });

  it("clears persisted recent searches after successfully deleting a profile", async () => {
    apiMocks.getUsers.mockResolvedValue([
      { user_id: "ollie", protection: "passwordless", locked: false },
      { user_id: "other", protection: "passwordless", locked: false },
    ]);
    useAtlasStore.setState({
      currentUserId: "ollie",
      currentThreadId: "secret-thread",
      currentThreadTitle: "Secret customer title",
      draftThreadModel: "private-model",
      recentSearchQueries: ["private search"],
      pinnedThreadKeys: ["ollie::secret-thread", "other::public-thread"],
    });
    const rendered = renderSettings("/settings?section=profiles");
    await act(async () => {
      await vi.waitFor(() =>
        expect(rendered.container.querySelector('button[aria-label="Delete profile other"]')).not.toBeNull(),
      );
      rendered.container.querySelector<HTMLButtonElement>('button[aria-label="Delete profile other"]')?.click();
    });
    await act(async () => {
      await vi.waitFor(() => expect(findButton(document.body, "Delete other")).not.toBeNull());
      findButton(document.body, "Delete other")?.click();
      await vi.waitFor(() => expect(apiMocks.deleteUser).toHaveBeenCalledWith("other"));
    });

    expect(useAtlasStore.getState()).toMatchObject({
      currentUserId: "ollie",
      currentThreadId: "secret-thread",
      currentThreadTitle: "Secret customer title",
      draftThreadModel: "private-model",
      recentSearchQueries: [],
      pinnedThreadKeys: ["ollie::secret-thread"],
    });
    unmountSettings(rendered);
  });
});

function providerSettings(overrides: Record<string, unknown> = {}) {
  return {
    provider: "ollama",
    provider_label: "Ollama",
    base_url: "http://127.0.0.1:11434",
    has_api_key: true,
    secure_key_storage_available: true,
    providers: [
      { id: "ollama", label: "Ollama", default_base_url: "http://127.0.0.1:11434" },
      { id: "lm-studio", label: "LM Studio", default_base_url: "http://127.0.0.1:1234" },
    ],
    ...overrides,
  };
}

function runnerStatus(overrides: Record<string, unknown> = {}) {
  return {
    available: true,
    busy: false,
    active_runs: 0,
    runtime_preparing: false,
    supported_languages: [],
    server_languages: [],
    client_languages: [],
    ...overrides,
  };
}

function renderSettings(initialEntry = "/settings?section=connections") {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const queryClient = new QueryClient({
    defaultOptions: {
      mutations: { retry: false },
      queries: { retry: false },
    },
  });
  act(() => {
    root.render(
      createElement(
        QueryClientProvider,
        { client: queryClient },
        createElement(
          MemoryRouter,
          { initialEntries: [initialEntry] },
          createElement(SettingsPage),
        ),
      ),
    );
  });
  return { container, root };
}

async function waitForSettings() {
  await act(async () => {
    await vi.waitFor(() => {
      expect(document.querySelector('select[aria-label="Model provider"] option[value="lm-studio"]')).not.toBeNull();
      expect(findButton(document.body, "Save & restart")).not.toBeNull();
    });
  });
}

function findButton(container: ParentNode, label: string) {
  return Array.from(container.querySelectorAll("button")).find((button) => button.textContent?.includes(label));
}

function findProfileButton(container: ParentNode, userId: string, label: string) {
  const card = Array.from(container.querySelectorAll(".settings-user-card")).find((candidate) =>
    candidate.querySelector("strong")?.textContent === userId,
  );
  return Array.from(card?.querySelectorAll("button") ?? []).find((button) =>
    button.textContent?.trim() === label,
  ) ?? null;
}

function setInputValue(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
  setter?.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

function setSelectValue(select: HTMLSelectElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set;
  setter?.call(select, value);
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

function unmountSettings({ container, root }: { container: HTMLDivElement; root: Root }) {
  act(() => root.unmount());
  container.remove();
}
