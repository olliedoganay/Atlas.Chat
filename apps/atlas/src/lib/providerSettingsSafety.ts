export const PROVIDER_BUSY_MESSAGE =
  "Wait for the active chat or compaction run to finish before changing model connections.";
export const PROVIDER_RUNNER_BUSY_MESSAGE =
  "Wait for active code runs or runtime preparation to finish before changing model connections.";

type ProviderDefinition = {
  id: string;
  default_base_url: string;
};

export function providerSelectionDraft(
  provider: string,
  providers: ProviderDefinition[] = [],
) {
  const definition = providers.find((item) => item.id === provider);
  return {
    provider,
    baseUrl: definition?.default_base_url ?? "",
    apiKey: "",
  };
}

export function shouldPreserveExistingProviderKey(
  savedProvider: string | null | undefined,
  nextProvider: string,
  apiKeyDraft: string,
) {
  return Boolean(
    savedProvider &&
      savedProvider === nextProvider &&
      !apiKeyDraft.trim(),
  );
}

type ProviderControlStateInput = {
  hasSettings: boolean;
  provider: string;
  baseUrl: string;
  dirty: boolean;
  busy: boolean;
  savePending: boolean;
  clearPending: boolean;
};

export function providerControlState({
  hasSettings,
  provider,
  baseUrl,
  dirty,
  busy,
  savePending,
  clearPending,
}: ProviderControlStateInput) {
  return {
    busy,
    saveDisabled:
      !hasSettings ||
      !provider.trim() ||
      !baseUrl.trim() ||
      !dirty ||
      busy ||
      savePending ||
      clearPending,
    clearDisabled: busy || savePending || clearPending,
  };
}
