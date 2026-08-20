import * as Dialog from "@radix-ui/react-dialog";
import { useQuery } from "@tanstack/react-query";
import { Search, X } from "lucide-react";
import {
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { searchChats, type ChatSearchResult } from "../lib/api";
import { displayThreadTitle } from "../lib/threadTitles";
import { useAtlasStore } from "../store/useAtlasStore";

type ChatSearchDialogProps = {
  currentThreadId: string;
  currentThreadTitle: string;
  currentUserId: string;
  onOpenChange: (open: boolean) => void;
  onPick: (
    result: ChatSearchResult,
    query: string,
    meta?: { historyIndices?: number[]; activePosition?: number },
  ) => void;
  open: boolean;
};

type SearchResultRef = {
  result: ChatSearchResult;
  scope: "current" | "other";
};

export function ChatSearchDialog({
  currentThreadId,
  currentThreadTitle,
  currentUserId,
  onOpenChange,
  onPick,
  open,
}: ChatSearchDialogProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const resultsRef = useRef<HTMLDivElement | null>(null);
  const [query, setQuery] = useState("");
  const [activeResultIndex, setActiveResultIndex] = useState(0);
  const deferredQuery = useDeferredValue(query.trim());
  const recentSearchQueries = useAtlasStore((state) => state.recentSearchQueries);
  const addRecentSearchQuery = useAtlasStore((state) => state.addRecentSearchQuery);
  const canSearch = open && Boolean(currentUserId) && deferredQuery.length >= 2;

  const { data, isFetching, isError, error } = useQuery({
    queryKey: ["chat-search", currentUserId, currentThreadId, deferredQuery],
    queryFn: () => searchChats(deferredQuery, currentUserId, currentThreadId),
    enabled: canSearch,
    staleTime: 1000,
  });

  useEffect(() => {
    if (!open) {
      setQuery("");
      setActiveResultIndex(0);
    }
  }, [open]);

  const currentResults = data?.current_thread_results ?? [];
  const otherResults = data?.other_thread_results ?? [];
  const totalResults = currentResults.length + otherResults.length;
  const currentThreadHistoryIndices = useMemo(
    () =>
      currentResults
        .map((result) => result.history_index)
        .filter((value): value is number => typeof value === "number"),
    [currentResults],
  );
  const currentThreadLabel = displayThreadTitle(currentThreadTitle, currentThreadId, "Current chat");
  const flatResults = useMemo<SearchResultRef[]>(
    () => [
      ...currentResults.map((result) => ({ result, scope: "current" as const })),
      ...otherResults.map((result) => ({ result, scope: "other" as const })),
    ],
    [currentResults, otherResults],
  );
  const activeResultKey = flatResults[activeResultIndex]
    ? searchResultKey(flatResults[activeResultIndex].result)
    : null;
  const otherResultGroups = useMemo(() => {
    const groups = new Map<
      string,
      {
        threadId: string;
        threadTitle: string;
        chatModel?: string;
        updatedAt?: string;
        results: ChatSearchResult[];
      }
    >();
    otherResults.forEach((result) => {
      const existing = groups.get(result.thread_id);
      if (existing) {
        existing.results.push(result);
        return;
      }
      groups.set(result.thread_id, {
        threadId: result.thread_id,
        threadTitle: displayThreadTitle(result.thread_title, result.thread_id),
        chatModel: result.chat_model,
        updatedAt: result.updated_at,
        results: [result],
      });
    });
    return Array.from(groups.values());
  }, [otherResults]);
  const statusCopy = useMemo(() => {
    if (!currentUserId) {
      return "Select a user before searching chats.";
    }
    if (deferredQuery.length === 0) {
      return "Search this chat and the rest of your local conversation archive.";
    }
    if (deferredQuery.length < 2) {
      return "Type at least 2 characters to search message content and chat titles.";
    }
    if (isFetching) {
      return "Searching local chats...";
    }
    if (isError) {
      return "Atlas could not search local chats.";
    }
    if (totalResults === 0) {
      return "No matching threads or messages in local history.";
    }
    return `${totalResults} match${totalResults === 1 ? "" : "es"} across local chats.`;
  }, [currentUserId, deferredQuery.length, isError, isFetching, totalResults]);

  useEffect(() => {
    setActiveResultIndex(0);
  }, [deferredQuery, open, totalResults]);

  useEffect(() => {
    resultsRef.current
      ?.querySelector<HTMLElement>('[data-search-active="true"]')
      ?.scrollIntoView?.({ block: "nearest" });
  }, [activeResultIndex]);

  const pickResult = (selected: SearchResultRef) => {
    const normalizedQuery = deferredQuery || query.trim();
    const result = selected.result;
    const meta =
      selected.scope === "current" && typeof result.history_index === "number"
        ? {
            historyIndices: currentThreadHistoryIndices,
            activePosition: Math.max(0, currentThreadHistoryIndices.indexOf(result.history_index)),
          }
        : undefined;
    addRecentSearchQuery(normalizedQuery);
    onPick(result, normalizedQuery, meta);
  };

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (!flatResults.length || deferredQuery.length < 2) {
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveResultIndex((current) => Math.min(flatResults.length - 1, current + 1));
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveResultIndex((current) => Math.max(0, current - 1));
      return;
    }
    if (event.key === "Enter") {
      if (event.target !== inputRef.current || event.nativeEvent.isComposing) {
        return;
      }
      event.preventDefault();
      const selected = flatResults[activeResultIndex];
      if (selected) {
        pickResult(selected);
      }
    }
  };

  if (!open) {
    return null;
  }

  return (
    <Dialog.Root onOpenChange={onOpenChange} open={open}>
      <Dialog.Portal>
        <Dialog.Overlay className="search-dialog-backdrop" />
        <Dialog.Content
          className="search-dialog"
          onKeyDown={handleKeyDown}
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            inputRef.current?.focus();
          }}
        >
        <div className="search-dialog-header">
          <div>
            <p className="workspace-section-label">Search</p>
            <Dialog.Title>Search chats</Dialog.Title>
          </div>
          <Dialog.Close asChild>
            <button
              aria-label="Close search"
              className="ghost-button icon-button"
              type="button"
            >
              <X size={16} />
            </button>
          </Dialog.Close>
        </div>

        <label className="search-input-shell">
          <Search size={18} />
          <input
            aria-activedescendant={
              activeResultKey ? searchResultDomId(activeResultKey) : undefined
            }
            aria-controls="chat-search-results"
            aria-label="Search chat history"
            className="search-input"
            onChange={(event) => setQuery(event.currentTarget.value)}
            placeholder="Search this chat and all chats"
            maxLength={1000}
            ref={inputRef}
            role="searchbox"
            value={query}
          />
          {query ? (
            <button
              aria-label="Clear search"
              className="ghost-button icon-button"
              onClick={() => setQuery("")}
              type="button"
            >
              <X size={14} />
            </button>
          ) : (
            <span className="search-shortcut-hint">Ctrl+K</span>
          )}
        </label>

        <Dialog.Description asChild>
          <p aria-live="polite" className="search-status-copy" role="status">
            {statusCopy}
          </p>
        </Dialog.Description>
        {isError ? (
          <div className="error-inline" role="alert">
            {error instanceof Error ? error.message : "Atlas could not search local chats."}
          </div>
        ) : null}

        {!deferredQuery && recentSearchQueries.length > 0 ? (
          <div className="search-recent-row">
            <span className="search-recent-label">Recent</span>
            <div className="search-recent-chips">
              {recentSearchQueries.map((item) => (
                <button
                  className="ghost-button search-recent-chip"
                  key={item}
                  onClick={() => setQuery(item)}
                  type="button"
                >
                  {item}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        <div className="search-dialog-sections" id="chat-search-results" ref={resultsRef}>
          <SearchSection
            activeResultKey={activeResultKey}
            emptyLabel={deferredQuery.length >= 2 ? "No hits in this chat." : "Start typing to search inside this chat."}
            onPick={(result) => pickResult({ result, scope: "current" })}
            query={deferredQuery}
            results={currentResults}
            sectionLabel="This chat"
            sectionTitle={currentThreadLabel}
          />
          <GroupedSearchSection
            activeResultKey={activeResultKey}
            emptyLabel={deferredQuery.length >= 2 ? "No hits in other chats." : "Search will also scan your other local chats."}
            groups={otherResultGroups}
            onPick={(result) => pickResult({ result, scope: "other" })}
            query={deferredQuery}
            sectionLabel="All chats"
            sectionTitle="Other conversations"
          />
        </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

type SearchSectionProps = {
  activeResultKey?: string | null;
  emptyLabel: string;
  onPick: (result: ChatSearchResult) => void;
  query: string;
  results: ChatSearchResult[];
  sectionLabel: string;
  sectionTitle: string;
};

function SearchSection({
  activeResultKey,
  emptyLabel,
  onPick,
  query,
  results,
  sectionLabel,
  sectionTitle,
}: SearchSectionProps) {
  return (
    <section className="search-section">
      <div className="search-section-head">
        <div>
          <p className="workspace-section-label">{sectionLabel}</p>
          <h3>{sectionTitle}</h3>
        </div>
        <span className="search-section-count">{results.length}</span>
      </div>

      {results.length > 0 ? (
        <div className="search-result-list">
          {results.map((result, index) => {
            const resultKey = searchResultKey(result);
            const active = activeResultKey === resultKey;
            return (
              <button
              className={`search-result-card${active ? " active" : ""}`}
              data-search-active={active ? "true" : undefined}
              id={searchResultDomId(resultKey)}
              key={`${resultKey}-${index}`}
              onClick={() => onPick(result)}
              type="button"
              >
              <div className="search-result-top">
                <div className="search-result-title-block">
                  <strong>{highlightSearchText(displayThreadTitle(result.thread_title, result.thread_id), query)}</strong>
                  <span className="search-result-badge">
                    {result.match_type === "thread"
                      ? "Chat"
                      : result.role === "assistant"
                        ? "Model"
                        : "User"}
                  </span>
                </div>
                <span className="search-result-meta">{formatSearchDate(result.updated_at)}</span>
              </div>
              <p className="search-result-snippet">{highlightSearchText(result.snippet, query)}</p>
              <div className="search-result-foot">
                <span>{result.chat_model || "Local model"}</span>
                <span>{result.history_index === null || result.history_index === undefined ? "Open chat" : "Jump to match"}</span>
              </div>
              </button>
            );
          })}
        </div>
      ) : (
        <div className="search-empty-state">{emptyLabel}</div>
      )}
    </section>
  );
}

type GroupedSearchSectionProps = {
  activeResultKey?: string | null;
  emptyLabel: string;
  groups: Array<{
    threadId: string;
    threadTitle: string;
    chatModel?: string;
    updatedAt?: string;
    results: ChatSearchResult[];
  }>;
  onPick: (result: ChatSearchResult) => void;
  query: string;
  sectionLabel: string;
  sectionTitle: string;
};

function GroupedSearchSection({
  activeResultKey,
  emptyLabel,
  groups,
  onPick,
  query,
  sectionLabel,
  sectionTitle,
}: GroupedSearchSectionProps) {
  return (
    <section className="search-section">
      <div className="search-section-head">
        <div>
          <p className="workspace-section-label">{sectionLabel}</p>
          <h3>{sectionTitle}</h3>
        </div>
        <span className="search-section-count">{groups.length}</span>
      </div>

      {groups.length > 0 ? (
        <div className="search-result-list">
          {groups.map((group) => (
            <div
              className={`search-thread-group${
                group.results.some((result) => searchResultKey(result) === activeResultKey)
                  ? " active"
                  : ""
              }`}
              key={group.threadId}
            >
              <div className="search-thread-group-head">
                <div className="search-thread-group-copy">
                  <strong>{highlightSearchText(group.threadTitle, query)}</strong>
                  <span>{group.chatModel || "Local model"}</span>
                </div>
                <span className="search-result-meta">{formatSearchDate(group.updatedAt)}</span>
              </div>
              <div className="search-thread-group-results">
                {group.results.map((result, index) => {
                  const resultKey = searchResultKey(result);
                  const active = activeResultKey === resultKey;
                  return (
                    <button
                    className={`search-result-card grouped${active ? " active" : ""}`}
                    data-search-active={active ? "true" : undefined}
                    id={searchResultDomId(resultKey)}
                    key={`${resultKey}-${index}`}
                    onClick={() => onPick(result)}
                    type="button"
                    >
                    <div className="search-result-top">
                      <div className="search-result-title-block">
                        <span className="search-result-badge">
                          {result.match_type === "thread"
                            ? "Chat"
                            : result.role === "assistant"
                              ? "Model"
                              : "User"}
                        </span>
                      </div>
                    </div>
                    <p className="search-result-snippet">{highlightSearchText(result.snippet, query)}</p>
                    <div className="search-result-foot">
                      <span>{result.history_index === null || result.history_index === undefined ? "Open chat" : "Jump to match"}</span>
                    </div>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="search-empty-state">{emptyLabel}</div>
      )}
    </section>
  );
}

function searchResultKey(result: ChatSearchResult) {
  return [
    result.thread_id,
    result.match_type,
    result.history_index ?? "thread",
    result.role ?? "",
  ].join(":");
}

function searchResultDomId(resultKey: string) {
  return `chat-search-result-${encodeURIComponent(resultKey)}`;
}

function highlightSearchText(value: string, query: string): ReactNode {
  if (!query) {
    return value;
  }
  const normalizedValue = value.toLocaleLowerCase();
  const normalizedQuery = query.toLocaleLowerCase();
  const parts: ReactNode[] = [];
  let cursor = 0;
  let key = 0;

  while (cursor < value.length) {
    const matchIndex = normalizedValue.indexOf(normalizedQuery, cursor);
    if (matchIndex < 0) {
      parts.push(value.slice(cursor));
      break;
    }
    if (matchIndex > cursor) {
      parts.push(value.slice(cursor, matchIndex));
    }
    const matchEnd = matchIndex + query.length;
    parts.push(
      <mark className="search-highlight" key={`match-${key}`}>
        {value.slice(matchIndex, matchEnd)}
      </mark>,
    );
    key += 1;
    cursor = matchEnd;
  }

  return parts;
}

function formatSearchDate(value?: string) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return "";
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
  }).format(date);
}
