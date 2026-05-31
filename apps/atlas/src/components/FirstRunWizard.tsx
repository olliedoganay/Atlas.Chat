import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Check, ExternalLink, Terminal } from "lucide-react";

import { createUser, openExternalUrl } from "../lib/api";
import { detectDesktopPlatform, ollamaInstallCopy, platformShellName } from "../lib/platformCopy";

type StepState = "active" | "pending" | "done";

export function FirstRunWizard({
  ollamaOnline,
  hasLocalModels,
  embedModel,
  providerLabel = "Ollama",
  providerBaseUrl = "http://127.0.0.1:11434",
  isOllamaProvider = true,
  onProfileCreated,
  onDismiss,
}: {
  ollamaOnline: boolean;
  hasLocalModels: boolean;
  embedModel?: string;
  providerLabel?: string;
  providerBaseUrl?: string;
  isOllamaProvider?: boolean;
  onProfileCreated: (userId: string) => Promise<void> | void;
  onDismiss: () => void;
}) {
  const [profileName, setProfileName] = useState("");
  const [profileError, setProfileError] = useState<string | null>(null);
  const [profileCreated, setProfileCreated] = useState(false);
  const navigate = useNavigate();

  const createMutation = useMutation({
    mutationFn: async () => createUser(profileName.trim()),
    onSuccess: async (user) => {
      setProfileCreated(true);
      setProfileError(null);
      await onProfileCreated(user.user_id);
    },
    onError: (error) => {
      setProfileError(error instanceof Error && error.message ? error.message : "Could not create profile.");
    },
  });

  const step1: StepState = profileCreated ? "done" : "active";
  const step2: StepState = !profileCreated ? "pending" : ollamaOnline ? "done" : "active";
  const step3: StepState = !profileCreated || !ollamaOnline ? "pending" : hasLocalModels ? "done" : "active";

  const allDone = profileCreated && ollamaOnline && hasLocalModels;
  const resolvedEmbedModel = embedModel?.trim() || "nomic-embed-text:latest";
  const starterChatModel = "gpt-oss:20b";
  const platform = detectDesktopPlatform();
  const shellName = platformShellName(platform);
  const installCopy = ollamaInstallCopy(platform);

  return (
    <div className="wizard-overlay" role="dialog" aria-modal="true" aria-labelledby="wizard-title">
      <div className="wizard-card">
        <div className="wizard-header">
          <h2 id="wizard-title">Welcome to Atlas Chat</h2>
          <p>Three quick steps and you're chatting with a local model.</p>
        </div>

        <div className={`wizard-step ${step1}`}>
          <div className="wizard-step-head">
            <span className="wizard-step-num">{step1 === "done" ? <Check size={14} /> : "1"}</span>
            <h3>Create a profile</h3>
          </div>
          <div className="wizard-step-body">
            <span>Profiles separate your chats and memories. You can have several.</span>
            {!profileCreated ? (
              <div className="wizard-form">
                <input
                  aria-label="Profile name"
                  className="text-input"
                  onChange={(event) => setProfileName(event.currentTarget.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && profileName.trim() && !createMutation.isPending) {
                      createMutation.mutate();
                    }
                  }}
                  placeholder="my_profile"
                  value={profileName}
                />
                <button
                  className="primary-button compact-button"
                  disabled={!profileName.trim() || createMutation.isPending}
                  onClick={() => createMutation.mutate()}
                  type="button"
                >
                  {createMutation.isPending ? "Creating..." : "Create"}
                </button>
              </div>
            ) : (
              <span style={{ color: "var(--success)" }}>Profile "{profileName.trim()}" created.</span>
            )}
            {profileError ? <span style={{ color: "var(--danger)" }}>{profileError}</span> : null}
          </div>
        </div>

        <div className={`wizard-step ${step2}`}>
          <div className="wizard-step-head">
            <span className="wizard-step-num">{step2 === "done" ? <Check size={14} /> : "2"}</span>
            <h3>Connect {providerLabel}</h3>
          </div>
          <div className="wizard-step-body">
            <span>
              Atlas Chat uses {providerLabel} as the local runtime for chat models on this machine.{" "}
              <strong style={{ color: ollamaOnline ? "var(--success)" : "var(--danger)" }}>
                {ollamaOnline ? "Connected" : "Not running"}
              </strong>
            </span>
            {!ollamaOnline ? (
              <div className="wizard-form">
                {isOllamaProvider ? (
                  <a
                    className="ghost-button compact-button"
                    href="https://ollama.com/download"
                    onClick={(event) => {
                      event.preventDefault();
                      void openExternalUrl("https://ollama.com/download");
                    }}
                    rel="noreferrer"
                    target="_blank"
                  >
                    <ExternalLink size={14} />
                    Download Ollama
                  </a>
                ) : (
                  <span style={{ color: "var(--muted)", fontSize: "var(--text-sm)" }}>
                    Start the local API at {providerBaseUrl}.
                  </span>
                )}
              </div>
            ) : null}
          </div>
        </div>

        <div className={`wizard-step ${step3}`}>
          <div className="wizard-step-head">
            <span className="wizard-step-num">{step3 === "done" ? <Check size={14} /> : "3"}</span>
            <h3>Install a model</h3>
          </div>
          <div className="wizard-step-body">
            <span>Download at least one chat model. Discovery can recommend one that fits this computer.</span>
            <div className="wizard-form">
              <button
                className="primary-button compact-button"
                disabled={!profileCreated || !ollamaOnline}
                onClick={() => {
                  navigate("/discovery");
                  onDismiss();
                }}
                type="button"
              >
                Open Discovery
              </button>
            </div>
          </div>
        </div>

        <section className="wizard-help-panel" aria-labelledby="wizard-help-title">
          <div className="wizard-help-heading">
            <Terminal size={16} />
            <div>
              <h3 id="wizard-help-title">New to local AI?</h3>
              <p>Connect a local provider first, then make at least one chat model available.</p>
            </div>
          </div>
          <div className="wizard-help-steps">
            <div>
              <strong>1. Connect {providerLabel}</strong>
              <p>{isOllamaProvider ? installCopy : `Start ${providerLabel} and expose its OpenAI-compatible local API at ${providerBaseUrl}.`}</p>
              {isOllamaProvider ? (
                <a
                  className="source-link wizard-help-link"
                  href="https://ollama.com/download"
                  onClick={(event) => {
                    event.preventDefault();
                    void openExternalUrl("https://ollama.com/download");
                  }}
                  rel="noreferrer"
                  target="_blank"
                >
                  Open Ollama download
                  <ExternalLink size={13} />
                </a>
              ) : null}
            </div>
            <div>
              <strong>2. Pull a chat model</strong>
              <p>
                {isOllamaProvider
                  ? `Open ${shellName} and run this example command. You can choose a different model later.`
                  : `Load or install a chat model inside ${providerLabel}.`}
              </p>
              <code>{isOllamaProvider ? `ollama pull ${starterChatModel}` : "Use your provider's local model manager"}</code>
            </div>
            <div>
              <strong>3. Pull the memory model</strong>
              <p>This embedding model lets Atlas Chat support local memory and retrieval features.</p>
              <code>{isOllamaProvider ? `ollama pull ${resolvedEmbedModel}` : resolvedEmbedModel}</code>
            </div>
            <div>
              <strong>4. Return to Atlas</strong>
              <p>When downloads finish, open Discovery or refresh the model list. Installed models appear automatically.</p>
            </div>
          </div>
        </section>

        <div className="wizard-footer">
          <button className="ghost-button compact-button" onClick={onDismiss} type="button">
            Skip for now
          </button>
          {allDone ? (
            <button
              className="primary-button compact-button"
              onClick={onDismiss}
              type="button"
            >
              Start chatting
            </button>
          ) : (
            <span style={{ color: "var(--muted)", fontSize: "var(--text-sm)" }}>
              Complete the steps above to begin.
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
