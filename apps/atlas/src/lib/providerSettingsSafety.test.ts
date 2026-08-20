import { describe, expect, it } from "vitest";

import {
  providerControlState,
  providerSelectionDraft,
  shouldPreserveExistingProviderKey,
} from "./providerSettingsSafety";

describe("provider settings safety", () => {
  it("never carries a staged key or previous provider key into a provider switch", () => {
    expect(
      providerSelectionDraft("lm-studio", [
        { id: "ollama", default_base_url: "http://127.0.0.1:11434" },
        { id: "lm-studio", default_base_url: "http://127.0.0.1:1234" },
      ]),
    ).toEqual({
      provider: "lm-studio",
      baseUrl: "http://127.0.0.1:1234",
      apiKey: "",
    });
    expect(shouldPreserveExistingProviderKey("ollama", "lm-studio", "")).toBe(false);
    expect(shouldPreserveExistingProviderKey("ollama", "ollama", "new-key")).toBe(false);
    expect(shouldPreserveExistingProviderKey("ollama", "ollama", "")).toBe(true);
  });

  it("blocks pristine and busy provider actions", () => {
    const base = {
      hasSettings: true,
      provider: "ollama",
      baseUrl: "http://127.0.0.1:11434",
      savePending: false,
      clearPending: false,
    };

    expect(providerControlState({ ...base, dirty: false, busy: false }).saveDisabled).toBe(true);
    expect(providerControlState({ ...base, dirty: true, busy: false }).saveDisabled).toBe(false);
    expect(providerControlState({ ...base, dirty: true, busy: true })).toMatchObject({
      busy: true,
      saveDisabled: true,
      clearDisabled: true,
    });
  });
});
