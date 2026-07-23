(() => {
  const defaultTheme = "dark";
  const storageKey = "atlas-ui-state";
  let resolvedTheme = defaultTheme;
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (raw) {
      const parsed = JSON.parse(raw);
      const state = parsed && typeof parsed === "object" ? (parsed.state ?? parsed) : null;
      if (state && state.theme === "light") {
        resolvedTheme = "light";
      }
    }
  } catch {
    resolvedTheme = defaultTheme;
  }
  document.documentElement.dataset.theme = resolvedTheme;
})();
