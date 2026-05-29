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

  useEffect(() => {
    if (!open) {
      setBusy(false);
    }
  }, [open]);

  const handleConfirm = async () => {
    if (busy) {
      return;
    }
    setBusy(true);
    try {
      await onConfirm();
      onOpenChange(false);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="dialog-content">
          <Dialog.Title className="dialog-title">{title}</Dialog.Title>
          <Dialog.Description className="dialog-description">{description}</Dialog.Description>
          <div className="dialog-actions">
            <Dialog.Close asChild>
              <button className="ghost-button" type="button">
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
