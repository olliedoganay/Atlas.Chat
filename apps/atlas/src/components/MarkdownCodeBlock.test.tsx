import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const runnerMocks = vi.hoisted(() => ({
  openRunnerWindow: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/runner", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/runner")>();
  return {
    ...actual,
    openRunnerWindow: runnerMocks.openRunnerWindow,
  };
});

import { MarkdownCodeBlock } from "./MarkdownCodeBlock";
import { useAtlasStore } from "../store/useAtlasStore";

let root: Root | null = null;
let container: HTMLDivElement | null = null;
const originalClipboardDescriptor = Object.getOwnPropertyDescriptor(Navigator.prototype, "clipboard");
const initialAtlasState = useAtlasStore.getInitialState();

function render(element: React.ReactElement) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  act(() => {
    root?.render(element);
  });
  return container;
}

describe("MarkdownCodeBlock", () => {
  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    root = null;
    container?.remove();
    container = null;
    runnerMocks.openRunnerWindow.mockClear();
    useAtlasStore.setState(initialAtlasState, true);
    vi.restoreAllMocks();
    if (originalClipboardDescriptor) {
      Object.defineProperty(Navigator.prototype, "clipboard", originalClipboardDescriptor);
    } else {
      Reflect.deleteProperty(Navigator.prototype, "clipboard");
    }
  });

  it("shows Run next to Copy for runnable code blocks", async () => {
    const node = render(<MarkdownCodeBlock code="print('hello')" language="python" />);

    const buttons = Array.from(node.querySelectorAll("button"));
    expect(buttons).toHaveLength(2);
    expect(buttons[0].textContent).toContain("Run");
    expect(buttons[1].textContent).toContain("Copy");

    await act(async () => {
      buttons[0].dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(runnerMocks.openRunnerWindow).toHaveBeenCalledWith({
      language: "python",
      code: "print('hello')",
    });
  });

  it("only shows Copy for unsupported code blocks", () => {
    const node = render(<MarkdownCodeBlock code="not runnable" language="brainfuck" />);

    const buttons = Array.from(node.querySelectorAll("button"));
    expect(buttons).toHaveLength(1);
    expect(buttons[0].textContent).toContain("Copy");
  });

  it("links a run to the active local profile and thread for safe repair handoff", async () => {
    useAtlasStore.setState({ currentUserId: "ollie", currentThreadId: "snake-chat" });
    const node = render(<MarkdownCodeBlock code="print('hello')" language="python" />);
    const runButton = Array.from(node.querySelectorAll("button")).find((button) =>
      button.textContent?.includes("Run"),
    );

    await act(async () => {
      runButton?.click();
    });

    expect(runnerMocks.openRunnerWindow).toHaveBeenCalledWith({
      language: "python",
      code: "print('hello')",
      originUserId: "ollie",
      originThreadId: "snake-chat",
    });
  });

  it("copies code when clipboard access is available", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    setClipboard({ writeText } as unknown as Clipboard);
    const node = render(<MarkdownCodeBlock code="print('hello')" language="python" />);
    const copyButton = Array.from(node.querySelectorAll("button")).find((button) =>
      button.textContent?.includes("Copy"),
    );

    await act(async () => {
      copyButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(writeText).toHaveBeenCalledWith("print('hello')");
    expect(copyButton?.textContent).toContain("Copied");
  });

  it("does not throw when clipboard access is unavailable", async () => {
    setClipboard(undefined);
    const node = render(<MarkdownCodeBlock code="print('hello')" language="python" />);
    const copyButton = Array.from(node.querySelectorAll("button")).find((button) =>
      button.textContent?.includes("Copy"),
    );

    await act(async () => {
      copyButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(copyButton?.textContent).toContain("Copy");
  });
});

function setClipboard(value: Clipboard | undefined) {
  Object.defineProperty(Navigator.prototype, "clipboard", {
    configurable: true,
    get: () => value,
  });
}
