import { type KeyboardEvent as ReactKeyboardEvent, useEffect, useRef, useState } from "react";
import { ChevronDown, Lock, Settings as SettingsIcon, Unlock, User } from "lucide-react";
import { useNavigate } from "react-router-dom";

import type { UserSummary } from "../lib/api";
import { PROFILE_SETTINGS_PATH } from "../lib/settingsSections";

export function ProfileMenu({
  users,
  currentUserId,
  onPick,
  onUnlock,
}: {
  users: UserSummary[];
  currentUserId: string;
  onPick: (userId: string) => void;
  onUnlock: (userId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const itemRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const initialMenuIndexRef = useRef(0);
  const navigate = useNavigate();

  const current = users.find((u) => u.user_id === currentUserId);
  const label = currentUserId || "No profile";
  const initial = (currentUserId || "?").slice(0, 1).toUpperCase();

  useEffect(() => {
    if (!open) {
      return;
    }
    const handleClick = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    window.addEventListener("mousedown", handleClick);
    return () => {
      window.removeEventListener("mousedown", handleClick);
    };
  }, [open]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const frame = window.requestAnimationFrame(() => {
      itemRefs.current[initialMenuIndexRef.current]?.focus();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [open]);

  const closeMenu = (restoreFocus = false) => {
    setOpen(false);
    if (restoreFocus) {
      window.requestAnimationFrame(() => triggerRef.current?.focus());
    }
  };

  const handleMenuKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const items = itemRefs.current.filter((item): item is HTMLButtonElement => Boolean(item && !item.disabled));
    const currentIndex = items.findIndex((item) => item === document.activeElement);
    if (event.key === "Escape") {
      event.preventDefault();
      closeMenu(true);
      return;
    }
    if (event.key === "Tab") {
      closeMenu();
      return;
    }
    if (event.key === "Home" || event.key === "End" || event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const nextIndex =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? items.length - 1
            : event.key === "ArrowDown"
              ? (currentIndex + 1 + items.length) % items.length
              : (currentIndex - 1 + items.length) % items.length;
      items[nextIndex]?.focus();
    }
  };

  return (
    <div className="profile-menu" ref={ref}>
      <button
        aria-controls="atlas-profile-menu"
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label={`Switch profile. Current profile: ${label}`}
        className="profile-menu-trigger"
        onClick={() => {
          initialMenuIndexRef.current = Math.max(0, users.findIndex((user) => user.user_id === currentUserId));
          setOpen((prev) => !prev);
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            initialMenuIndexRef.current = event.key === "ArrowUp" ? users.length : 0;
            setOpen(true);
          }
        }}
        ref={triggerRef}
        type="button"
        title="Switch profile"
      >
        <div className="profile-menu-avatar">{initial}</div>
        <div className="profile-menu-copy">
          <strong>{label}</strong>
          <span>{current ? (current.protection === "password" ? "Password protected" : "Active profile") : "Tap to choose"}</span>
        </div>
        <ChevronDown className="profile-menu-chevron" size={16} />
      </button>

      {open ? (
        <div
          aria-label="Profiles"
          className="profile-menu-pop"
          id="atlas-profile-menu"
          onKeyDown={handleMenuKeyDown}
          role="menu"
        >
          {users.length === 0 ? (
            <div className="profile-menu-empty">No profiles yet. Create one in Profiles.</div>
          ) : (
            users.map((user, index) => {
              const isActive = user.user_id === currentUserId;
              const isLocked = Boolean(user.locked);
              return (
                <button
                  aria-checked={isActive}
                  className={`profile-menu-item${isActive ? " active" : ""}`}
                  key={user.user_id}
                  onClick={() => {
                    closeMenu();
                    if (isLocked) {
                      onUnlock(user.user_id);
                    } else {
                      onPick(user.user_id);
                    }
                  }}
                  ref={(element) => {
                    itemRefs.current[index] = element;
                  }}
                  role="menuitemradio"
                  type="button"
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    {user.protection === "password" ? (isLocked ? <Lock size={14} /> : <Unlock size={14} />) : <User size={14} />}
                    {user.user_id}
                  </span>
                  <span className="profile-menu-item-meta">
                    {isLocked ? "Locked" : isActive ? "Current" : ""}
                  </span>
                </button>
              );
            })
          )}
          <div className="profile-menu-divider" />
          <button
            className="profile-menu-link"
            onClick={() => {
              closeMenu();
              navigate(PROFILE_SETTINGS_PATH);
            }}
            ref={(element) => {
              itemRefs.current[users.length] = element;
            }}
            role="menuitem"
            type="button"
          >
            <SettingsIcon size={14} />
            Manage profiles
          </button>
        </div>
      ) : null}
    </div>
  );
}
