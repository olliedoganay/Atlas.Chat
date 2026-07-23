import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { lazy, Suspense, useEffect } from "react";
import { HashRouter, Navigate, Route, Routes } from "react-router-dom";

import { AtlasShell } from "./components/AtlasShell";
import { WorkspacePage } from "./pages/WorkspacePage";
import { useAtlasStore } from "./store/useAtlasStore";

const AdvancedPage = lazy(async () => ({ default: (await import("./pages/AdvancedPage")).AdvancedPage }));
const CodeRunnerPage = lazy(async () => ({ default: (await import("./pages/CodeRunnerPage")).CodeRunnerPage }));
const DiscoveryPage = lazy(async () => ({ default: (await import("./pages/DiscoveryPage")).DiscoveryPage }));
const SettingsPage = lazy(async () => ({ default: (await import("./pages/SettingsPage")).SettingsPage }));

const queryClient = new QueryClient();

function ThemeBridge() {
  const theme = useAtlasStore((state) => state.theme);
  const crtScanlines = useAtlasStore((state) => state.crtScanlines);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    const isCrt = theme === "crt-green" || theme === "crt-amber";
    if (isCrt && crtScanlines) {
      document.documentElement.dataset.scanlines = "on";
    } else {
      delete document.documentElement.dataset.scanlines;
    }
    document.title = "Atlas Chat";
  }, [theme, crtScanlines]);

  return null;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeBridge />
      <HashRouter>
        <Routes>
          <Route
            element={<Suspense fallback={<RouteFallback />}><CodeRunnerPage /></Suspense>}
            path="/runner/:token"
          />
          <Route element={<AtlasShell />} path="/">
            <Route element={<Navigate replace to="/workspace" />} index />
            <Route element={<WorkspacePage />} path="workspace" />
            <Route
              element={<Suspense fallback={<RouteFallback />}><DiscoveryPage /></Suspense>}
              path="discovery"
            />
            <Route
              element={<Suspense fallback={<RouteFallback />}><AdvancedPage /></Suspense>}
              path="advanced"
            />
            <Route
              element={<Suspense fallback={<RouteFallback />}><SettingsPage /></Suspense>}
              path="settings"
            />
            <Route element={<Navigate replace to="/workspace" />} path="*" />
          </Route>
        </Routes>
      </HashRouter>
    </QueryClientProvider>
  );
}

function RouteFallback() {
  return (
    <div aria-live="polite" className="route-loading" role="status">
      <span className="status-dot" />
      Loading view…
    </div>
  );
}

export default App;
