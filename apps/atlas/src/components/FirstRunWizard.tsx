import * as Dialog from "@radix-ui/react-dialog";
import { useMutation } from "@tanstack/react-query";
import { ArrowRight, Check, ExternalLink, Terminal, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { createUser, openExternalUrl } from "../lib/api";
import { detectDesktopPlatform, ollamaInstallCopy, platformShellName } from "../lib/platformCopy";

export type FirstRunStage = "profile" | "provider" | "model" | "ready";

export function resolveFirstRunStage(
  profileCreated: boolean,
  providerOnline: boolean,
  hasLocalModels: boolean,
): FirstRunStage {
  if (!profileCreated) {
    return "profile";
  }
  if (!providerOnline) {
    return "provider";
  }
  if (!hasLocalModels) {
    return "model";
  }
  return "ready";
}

const setupSteps: Array<{ id: FirstRunStage; label: string }> = [
  { id: "profile", label: "Profile" },
  { id: "provider", label: "Provider" },
  { id: "model", label: "Model" },
  { id: "ready", label: "First prompt" },
];

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
  const profileInputRef = useRef<HTMLInputElement | null>(null);
  const stageHeadingRef = useRef<HTMLHeadingElement | null>(null);
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

  const stage = resolveFirstRunStage(profileCreated, ollamaOnline, hasLocalModels);
  const activeStepIndex = setupSteps.findIndex((step) => step.id === stage);
  const resolvedEmbedModel = embedModel?.trim() || "nomic-embed-text:latest";
  const starterChatModel = "gpt-oss:20b";
  const platform = detectDesktopPlatform();
  const shellName = platformShellName(platform);
  const installCopy = ollamaInstallCopy(platform);

  useEffect(() => {
    if (stage === "profile") {
      return;
    }
    const frame = window.requestAnimationFrame(() => stageHeadingRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [stage]);

  const dismiss = () => {
    onDismiss();
  };

  return (
    <Dialog.Root
      onOpenChange={(open) => {
        if (!open) {
          dismiss();
        }
      }}
      open
    >
      <Dialog.Portal>
        <Dialog.Overlay className="wizard-overlay" />
        <Dialog.Content
          className="wizard-card"
          onOpenAutoFocus={(event) => {
            if (profileInputRef.current) {
              event.preventDefault();
              profileInputRef.current.focus();
            }
          }}
        >
          <div className="wizard-header">
            <div>
              <p className="workspace-section-label">Local setup</p>
              <Dialog.Title>Welcome to Atlas Chat</Dialog.Title>
              <Dialog.Description>
                Create a local profile, connect a provider, and choose a model.
              </Dialog.Description>
            </div>
            <Dialog.Close asChild>
              <button aria-label="Skip setup" className="ghost-button icon-button wizard-close" type="button">
                <X size={16} />
              </button>
            </Dialog.Close>
          </div>

          <ol aria-label="Setup progress" className="wizard-progress">
            {setupSteps.map((step, index) => {
              const done = index < activeStepIndex || stage === "ready";
              const active = step.id === stage;
              return (
                <li aria-current={active ? "step" : undefined} className={active ? "active" : done ? "done" : ""} key={step.id}>
                  <span className="wizard-step-num" aria-hidden="true">
                    {done ? <Check size={13} /> : index + 1}
                  </span>
                  <span>{step.label}</span>
                </li>
              );
            })}
          </ol>

          <div aria-live="polite" className="wizard-stage">
            {stage === "profile" ? (
              <>
                <div className="wizard-stage-copy">
                  <span className="wizard-stage-kicker">Step 1</span>
                  <h3 ref={stageHeadingRef} tabIndex={-1}>Create your profile</h3>
                  <p>Profiles keep chats, search, and memory separate on this device.</p>
                </div>
                <form
                  className="wizard-form wizard-profile-form"
                  onSubmit={(event) => {
                    event.preventDefault();
                    if (profileName.trim() && !createMutation.isPending) {
                      createMutation.mutate();
                    }
                  }}
                >
                  <label htmlFor="first-run-profile">Profile name</label>
                  <div className="wizard-field-row">
                    <input
                      autoComplete="username"
                      className="text-input"
                      id="first-run-profile"
                      onChange={(event) => setProfileName(event.currentTarget.value)}
                      placeholder="my_profile"
                      ref={profileInputRef}
                      value={profileName}
                    />
                    <button
                      className="primary-button compact-button"
                      disabled={!profileName.trim() || createMutation.isPending}
                      type="submit"
                    >
                      {createMutation.isPending ? "Creating…" : "Continue"}
                      {!createMutation.isPending ? <ArrowRight size={14} /> : null}
                    </button>
                  </div>
                </form>
                {profileError ? (
                  <p className="wizard-inline-message error" role="alert">
                    {profileError}
                  </p>
                ) : null}
              </>
            ) : null}

            {stage === "provider" ? (
              <>
                <div className="wizard-stage-copy">
                  <span className="wizard-stage-kicker">Step 2</span>
                  <h3 ref={stageHeadingRef} tabIndex={-1}>Connect {providerLabel}</h3>
                  <p>
                    Atlas is ready. Start the local provider at <strong>{providerBaseUrl}</strong>; this screen updates automatically.
                  </p>
                </div>
                <div className="wizard-primary-action">
                  {isOllamaProvider ? (
                    <a
                      className="primary-button"
                      href="https://ollama.com/download"
                      onClick={(event) => {
                        event.preventDefault();
                        void openExternalUrl("https://ollama.com/download");
                      }}
                      rel="noreferrer"
                      target="_blank"
                    >
                      Download Ollama
                      <ExternalLink size={14} />
                    </a>
                  ) : (
                    <span className="status-pill warning">
                      <span className="status-dot" />
                      Waiting for {providerLabel}
                    </span>
                  )}
                </div>
                <ManualSetup
                  embedModel={resolvedEmbedModel}
                  installCopy={installCopy}
                  isOllamaProvider={isOllamaProvider}
                  providerLabel={providerLabel}
                  shellName={shellName}
                  starterChatModel={starterChatModel}
                />
              </>
            ) : null}

            {stage === "model" ? (
              <>
                <div className="wizard-stage-copy">
                  <span className="wizard-stage-kicker">Step 3</span>
                  <h3 ref={stageHeadingRef} tabIndex={-1}>Add your first model</h3>
                  <p>Discovery recommends models for this machine and keeps the next action in one place.</p>
                </div>
                <div className="wizard-primary-action">
                  <button
                    className="primary-button"
                    onClick={() => {
                      navigate("/discovery");
                      dismiss();
                    }}
                    type="button"
                  >
                    Open Discovery
                    <ArrowRight size={14} />
                  </button>
                </div>
                <ManualSetup
                  embedModel={resolvedEmbedModel}
                  installCopy={installCopy}
                  isOllamaProvider={isOllamaProvider}
                  providerLabel={providerLabel}
                  shellName={shellName}
                  starterChatModel={starterChatModel}
                />
              </>
            ) : null}

            {stage === "ready" ? (
              <div className="wizard-ready">
                <span className="wizard-ready-mark" aria-hidden="true">
                  <Check size={20} />
                </span>
                <div className="wizard-stage-copy">
                  <span className="wizard-stage-kicker">Setup complete</span>
                  <h3 ref={stageHeadingRef} tabIndex={-1}>Your local workspace is ready</h3>
                  <p>Start with a question, a draft, or a file you want to understand.</p>
                </div>
                <Dialog.Close asChild>
                  <button className="primary-button" type="button">
                    Write your first prompt
                    <ArrowRight size={14} />
                  </button>
                </Dialog.Close>
              </div>
            ) : null}
          </div>

          <div className="wizard-footer">
            <Dialog.Close asChild>
              <button className="ghost-button compact-button" type="button">
                Skip for now
              </button>
            </Dialog.Close>
            <span aria-live="polite">
              {stage === "ready" ? "Ready to chat" : `Step ${activeStepIndex + 1} of ${setupSteps.length}`}
            </span>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function ManualSetup({
  embedModel,
  installCopy,
  isOllamaProvider,
  providerLabel,
  shellName,
  starterChatModel,
}: {
  embedModel: string;
  installCopy: string;
  isOllamaProvider: boolean;
  providerLabel: string;
  shellName: string;
  starterChatModel: string;
}) {
  return (
    <details className="wizard-manual">
      <summary>
        <Terminal size={15} />
        Manual setup
      </summary>
      <div className="wizard-manual-body">
        <p>
          {isOllamaProvider
            ? installCopy
            : `Start ${providerLabel} and load a model with its local model manager.`}
        </p>
        {isOllamaProvider ? (
          <>
            <span>Run in {shellName}:</span>
            <code>{`ollama pull ${starterChatModel}`}</code>
            <code>{`ollama pull ${embedModel}`}</code>
          </>
        ) : null}
      </div>
    </details>
  );
}
