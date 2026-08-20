import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";

type ResetDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  confirmLabel: string;
  confirmIntent?: "danger" | "primary";
  busyLabel?: string;
  onConfirm: () => Promise<void> | void;
};

export function ResetDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  confirmIntent = "danger",
  busyLabel = "Applying...",
  onConfirm,
}: ResetDialogProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) {
      setBusy(false);
      setError("");
    }
  }, [open]);

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen && busy) {
      return;
    }
    onOpenChange(nextOpen);
  };

  const handleConfirm = async () => {
    if (busy) {
      return;
    }
    setError("");
    setBusy(true);
    try {
      await onConfirm();
      onOpenChange(false);
    } catch (confirmError) {
      setError(confirmError instanceof Error ? confirmError.message : "The action could not be completed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog.Root open={open} onOpenChange={handleOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content aria-busy={busy} className="dialog-content">
          <Dialog.Title className="dialog-title">{title}</Dialog.Title>
          <Dialog.Description className="dialog-description">{description}</Dialog.Description>
          {error ? (
            <p className="error-inline" role="alert">
              {error}
            </p>
          ) : null}
          <div className="dialog-actions">
            <Dialog.Close asChild>
              <button className="ghost-button" disabled={busy} type="button">
                Cancel
              </button>
            </Dialog.Close>
            <button
              className={confirmIntent === "primary" ? "primary-button" : "danger-button"}
              disabled={busy}
              onClick={handleConfirm}
              type="button"
            >
              {busy ? busyLabel : confirmLabel}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
