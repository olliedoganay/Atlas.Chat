import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

import { ResetDialog } from "./ResetDialog";

describe("ResetDialog", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    document.querySelectorAll("[data-radix-portal]").forEach((portal) => portal.remove());
    vi.clearAllMocks();
  });

  it("keeps the dialog open and announces confirmation failures", async () => {
    const onOpenChange = vi.fn();
    const onConfirm = vi.fn().mockRejectedValue(new Error("Delete failed safely."));
    act(() => {
      root.render(
        <ResetDialog
          confirmLabel="Delete"
          description="Delete local data"
          onConfirm={onConfirm}
          onOpenChange={onOpenChange}
          open
          title="Delete?"
        />,
      );
    });

    const confirmButton = Array.from(document.querySelectorAll("button")).find(
      (button) => button.textContent === "Delete",
    );
    await act(async () => {
      confirmButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onOpenChange).not.toHaveBeenCalledWith(false);
    expect(document.querySelector('[role="alert"]')?.textContent).toContain(
      "Delete failed safely.",
    );
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();
  });
});
