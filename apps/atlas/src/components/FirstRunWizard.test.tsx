import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const apiMocks = vi.hoisted(() => ({
  createUser: vi.fn(),
  openExternalUrl: vi.fn(),
}));

vi.mock("../lib/api", () => ({
  createUser: apiMocks.createUser,
  openExternalUrl: apiMocks.openExternalUrl,
}));

import { FirstRunWizard, resolveFirstRunStage } from "./FirstRunWizard";

describe("resolveFirstRunStage", () => {
  it("moves through the setup gates in order", () => {
    expect(resolveFirstRunStage(false, false, false)).toBe("profile");
    expect(resolveFirstRunStage(true, false, false)).toBe("provider");
    expect(resolveFirstRunStage(true, true, false)).toBe("model");
    expect(resolveFirstRunStage(true, true, true)).toBe("ready");
  });
});

describe("FirstRunWizard", () => {
  afterEach(() => {
    vi.clearAllMocks();
    document.body.innerHTML = "";
  });

  it("opens as a labelled modal, focuses the profile field, and dismisses on Escape", async () => {
    const onDismiss = vi.fn();
    const rendered = renderWizard({ onDismiss });

    await flushEffects();

    const dialog = document.querySelector<HTMLElement>('[role="dialog"]');
    const profileInput = document.querySelector<HTMLInputElement>("#first-run-profile");
    expect(dialog?.getAttribute("aria-labelledby")).toBeTruthy();
    expect(dialog?.getAttribute("aria-describedby")).toBeTruthy();
    expect(profileInput).not.toBeNull();
    expect(document.activeElement).toBe(profileInput);

    act(() => {
      profileInput?.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key: "Escape" }));
    });
    expect(onDismiss).toHaveBeenCalledTimes(1);

    unmountWizard(rendered);
  });

  it("shows only the current setup task after creating a profile", async () => {
    apiMocks.createUser.mockResolvedValue({ user_id: "ollie" });
    const onProfileCreated = vi.fn();
    const rendered = renderWizard({ onProfileCreated });

    await flushEffects();
    const profileInput = document.querySelector<HTMLInputElement>("#first-run-profile");
    expect(profileInput).not.toBeNull();

    act(() => {
      setInputValue(profileInput as HTMLInputElement, "ollie");
    });
    const continueButton = Array.from(document.querySelectorAll("button")).find((button) =>
      button.textContent?.includes("Continue"),
    );
    await act(async () => {
      continueButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(apiMocks.createUser).toHaveBeenCalledWith("ollie");
    expect(onProfileCreated).toHaveBeenCalledWith("ollie");
    expect(document.body.textContent).toContain("Connect Ollama");
    expect(document.body.textContent).not.toContain("Create your profile");
    expect(document.querySelectorAll(".wizard-manual")).toHaveLength(1);
    expect(document.activeElement?.textContent).toContain("Connect Ollama");

    unmountWizard(rendered);
  });
});

function renderWizard({
  onDismiss = vi.fn(),
  onProfileCreated = vi.fn(),
}: {
  onDismiss?: () => void;
  onProfileCreated?: (userId: string) => void;
} = {}) {
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
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <FirstRunWizard
            hasLocalModels={false}
            ollamaOnline={false}
            onDismiss={onDismiss}
            onProfileCreated={onProfileCreated}
          />
        </MemoryRouter>
      </QueryClientProvider>,
    );
  });

  return { container, root };
}

function unmountWizard({ container, root }: { container: HTMLDivElement; root: Root }) {
  act(() => {
    root.unmount();
  });
  container.remove();
}

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

function setInputValue(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
  setter?.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}
