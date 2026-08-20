import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const apiMocks = vi.hoisted(() => ({
  openExternalUrl: vi.fn(),
}));

vi.mock("../lib/api", () => ({
  openExternalUrl: apiMocks.openExternalUrl,
}));

import {
  MarkdownMessageContent,
  markdownCodeLanguage,
  safeExternalHref,
} from "./MarkdownMessageContent";

describe("MarkdownMessageContent", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    apiMocks.openExternalUrl.mockResolvedValue(undefined);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  it("preserves punctuation in fenced-code language names", () => {
    expect(markdownCodeLanguage("language-c++")).toBe("c++");
    expect(markdownCodeLanguage("highlight language-c# extra")).toBe("c#");
  });

  it("accepts only absolute credential-free http links", () => {
    expect(safeExternalHref("https://example.com/docs")).toBe("https://example.com/docs");
    expect(safeExternalHref("http://example.com")).toBe("http://example.com/");
    expect(safeExternalHref("/relative")).toBeNull();
    expect(safeExternalHref("javascript:alert(1)")).toBeNull();
    expect(safeExternalHref("file:///tmp/private")).toBeNull();
    expect(safeExternalHref("https://user:secret@example.com")).toBeNull();
    expect(safeExternalHref(" https://example.com")).toBeNull();
    expect(safeExternalHref("https://example.com/\nnext")).toBeNull();
  });

  it("routes web links through the guarded external opener without a native href", async () => {
    act(() => {
      root.render(<MarkdownMessageContent content="[Atlas docs](https://example.com/docs)" />);
    });
    const linkButton = container.querySelector<HTMLButtonElement>('[role="link"]');

    expect(linkButton?.textContent).toContain("Atlas docs");
    expect(container.querySelector("a[href]")).toBeNull();

    await act(async () => {
      linkButton?.click();
      await Promise.resolve();
    });

    expect(apiMocks.openExternalUrl).toHaveBeenCalledWith("https://example.com/docs");
  });

  it("renders unsafe schemes inert and announces external-open failures", async () => {
    act(() => {
      root.render(
        <MarkdownMessageContent content={"[Unsafe](javascript:alert(1)) and [Web](https://example.com)"} />,
      );
    });

    expect(container.querySelector(".markdown-link-disabled")?.textContent).toBe("Unsafe");
    expect(container.querySelectorAll('[role="link"]')).toHaveLength(1);
    expect(container.querySelector("a[href]")).toBeNull();

    apiMocks.openExternalUrl.mockRejectedValueOnce(new Error("Link blocked by desktop policy."));
    await act(async () => {
      container.querySelector<HTMLButtonElement>('[role="link"]')?.click();
      await Promise.resolve();
    });

    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      "Link blocked by desktop policy.",
    );
  });
});
