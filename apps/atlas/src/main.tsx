import React from "react";
import ReactDOM from "react-dom/client";
import { isTauri } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";
import "@fontsource/ibm-plex-sans/latin-600.css";
import "@fontsource/space-grotesk/latin-500.css";
import "@fontsource/space-grotesk/latin-700.css";
import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

function revealTauriWindow() {
  const appWindow = getCurrentWindow();
  void appWindow
    .show()
    .then(() => appWindow.setFocus())
    .catch(() => undefined);
}

if (isTauri()) {
  if (document.readyState === "complete") {
    window.setTimeout(revealTauriWindow, 0);
  } else {
    window.addEventListener("load", () => window.setTimeout(revealTauriWindow, 0), { once: true });
  }
}
