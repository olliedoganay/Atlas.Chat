import { act, createElement, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  appendComposerPrompt,
  buildConversationPresentation,
  canAcceptRunnerRepairRequest,
  validateAttachmentSelection,
  WorkspaceComposer,
  workspaceComposerDraftKey,
} from "./WorkspacePage";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(() => {
  document.body.innerHTML = "";
});

function file(name: string, size: number, type = "application/octet-stream") {
  return { name, size, type };
}

describe("workspace attachment validation", () => {
  it("enforces backend attachment limits before reading file contents", () => {
    expect(
      validateAttachmentSelection([], Array.from({ length: 9 }, (_, index) => file(`${index}.png`, 1, "image/png"))),
    ).toContain("up to 8");
    expect(validateAttachmentSelection([], [file("large.png", 10 * 1024 * 1024 + 1, "image/png")])).toContain(
      "10 MiB",
    );
    expect(validateAttachmentSelection([], [file("notes.txt", 500_001, "text/plain")])).toContain("500 KB");
    expect(
      validateAttachmentSelection(
        [{ byte_size: 20 * 1024 * 1024 }],
        [file("more.pdf", 6 * 1024 * 1024, "application/pdf")],
      ),
    ).toContain("25 MiB");
    expect(validateAttachmentSelection([], [file(`${"n".repeat(256)}.txt`, 10, "text/plain")])).toContain(
      "255 characters",
    );
  });

  it("accepts a valid mixed selection", () => {
    expect(
      validateAttachmentSelection(
        [{ byte_size: 1024 }],
        [file("photo.png", 2048, "image/png"), file("notes.txt", 4096, "text/plain")],
      ),
    ).toBeNull();
  });

  it("rejects active XML-based image formats before reading them", () => {
    expect(
      validateAttachmentSelection([], [file("diagram.svg", 1024, "image/svg+xml")]),
    ).toContain("SVG");
    expect(
      validateAttachmentSelection([], [file("vector.img", 1024, "image/example+xml")]),
    ).toContain("XML");
  });
});

describe("workspace composer draft isolation", () => {
  it("appends a runner repair draft without overwriting existing unsent text", () => {
    expect(appendComposerPrompt("keep this note", "repair request")).toBe(
      "keep this note\n\nrepair request",
    );
  });

  it("accepts repair drafts only for an unlocked, fetched, exact profile and thread", () => {
    const request = {
      version: 1 as const,
      requestId: "repair-1",
      language: "python",
      code: "print('private')",
      diagnostics: "exit 1",
      originUserId: "ollie",
      originThreadId: "snake-chat",
      createdAt: 1,
    };
    const scope = {
      currentUserId: "ollie",
      currentThreadId: "snake-chat",
      currentUserAvailable: true,
      currentUserLocked: false,
      usersFetched: true,
    };

    expect(canAcceptRunnerRepairRequest(request, scope)).toBe(true);
    expect(canAcceptRunnerRepairRequest(request, { ...scope, currentThreadId: "other-chat" })).toBe(false);
    expect(canAcceptRunnerRepairRequest(request, { ...scope, currentUserLocked: true })).toBe(false);
    expect(canAcceptRunnerRepairRequest(request, { ...scope, usersFetched: false })).toBe(false);
  });

  it("clears typed text when the active thread changes", () => {
    const rendered = renderComposer("ollie", "thread-a");
    const textarea = rendered.container.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]');
    expect(textarea).not.toBeNull();

    act(() => setTextareaValue(textarea as HTMLTextAreaElement, "private thread-a draft"));
    expect(textarea?.value).toBe("private thread-a draft");

    rerenderComposer(rendered.root, "ollie", "thread-b");

    expect(rendered.container.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]')?.value).toBe("");
    unmountComposer(rendered);
  });

  it("clears an image attachment when the active profile changes", async () => {
    const rendered = renderComposer("ollie", "main");
    const imageInput = rendered.container.querySelector<HTMLInputElement>('input[type="file"][accept="image/*"]');
    expect(imageInput).not.toBeNull();

    const image = new File(["image-bytes"], "private.png", { type: "image/png" });
    Object.defineProperty(imageInput, "files", { configurable: true, value: [image] });
    await act(async () => {
      imageInput?.dispatchEvent(new Event("change", { bubbles: true }));
      await new Promise((resolve) => window.setTimeout(resolve, 0));
    });
    await vi.waitFor(() => {
      expect(rendered.container.querySelector('img[alt="private.png"]')).not.toBeNull();
    });

    rerenderComposer(rendered.root, "other-profile", "main");

    expect(rendered.container.querySelector('img[alt="private.png"]')).toBeNull();
    unmountComposer(rendered);
  });
});

describe("assistant retry context", () => {
  it("branches before the target user message so retry does not duplicate it", () => {
    const firstTurn = buildConversationPresentation([
      { role: "user", content: "make a snake game" },
      { role: "assistant", content: "broken code" },
    ]);
    expect(firstTurn[1].retryContext).toMatchObject({
      afterMessageCount: 0,
      prompt: "make a snake game",
    });

    const laterTurn = buildConversationPresentation([
      { role: "user", content: "first" },
      { role: "assistant", content: "first answer" },
      { role: "system", content: "context compacted", kind: "context_compacted" },
      { role: "user", content: "second" },
      { role: "assistant", content: "second answer" },
    ]);
    expect(laterTurn[4].retryContext).toMatchObject({
      afterMessageCount: 2,
      prompt: "second",
    });
  });
});

type ComposerProps = ComponentProps<typeof WorkspaceComposer>;

function renderComposer(userId: string, threadId: string) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  rerenderComposer(root, userId, threadId);
  return { container, root };
}

function rerenderComposer(root: Root, userId: string, threadId: string) {
  act(() => {
    root.render(
      createElement(WorkspaceComposer, {
        ...composerProps(userId),
        key: workspaceComposerDraftKey(userId, threadId),
      }),
    );
  });
}

function composerProps(currentUserId: string): ComposerProps {
  return {
    activeReasoningOption: null,
    canStartChat: true,
    currentRunId: null,
    currentRunMode: null,
    currentThreadCompactionNotice: null,
    currentThreadHasActiveRun: false,
    currentUserId,
    effectiveReasoningMode: "on",
    isCompactingContext: false,
    isStreaming: false,
    liveAnswer: "",
    onManualCompact: vi.fn(),
    onStopRun: vi.fn(),
    onSubmitPrompt: vi.fn(),
    pendingAttachments: [],
    pendingPrompt: "",
    placeholder: "Message Atlas",
    providerLabel: "Ollama",
    reasoningOptions: [],
    runDetectedContextWindow: null,
    selectedModelSupportsImages: true,
    selectedModelSupportsReasoning: false,
    setReasoningMode: vi.fn(),
    startManualCompactPending: false,
    startRunPending: false,
    stopRunPending: false,
    threadHasHistory: false,
    visibleHistory: [],
  };
}

function setTextareaValue(textarea: HTMLTextAreaElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
  setter?.call(textarea, value);
  textarea.dispatchEvent(new Event("input", { bubbles: true }));
}

function unmountComposer({ container, root }: { container: HTMLDivElement; root: Root }) {
  act(() => root.unmount());
  container.remove();
}
