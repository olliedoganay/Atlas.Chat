import type { QueryClient } from "@tanstack/react-query";

type CachedThread = {
  last_run_id?: string | null;
};

type CachedRun = {
  run_id?: string;
  user_id?: string;
};

export function removeLockedProfileCaches(queryClient: QueryClient, userId: string) {
  const runIds = new Set<string>();
  const cachedThreads = queryClient.getQueryData<CachedThread[]>(["threads", userId]) ?? [];
  for (const thread of cachedThreads) {
    if (thread.last_run_id) {
      runIds.add(thread.last_run_id);
    }
  }
  for (const [, runs] of queryClient.getQueriesData<CachedRun[]>({
    queryKey: ["thread-runs", userId],
  })) {
    for (const run of runs ?? []) {
      if (run.run_id) {
        runIds.add(run.run_id);
      }
    }
  }

  queryClient.removeQueries({ queryKey: ["threads", userId] });
  queryClient.removeQueries({ queryKey: ["thread-history", userId] });
  queryClient.removeQueries({ queryKey: ["thread-runs", userId] });
  queryClient.removeQueries({ queryKey: ["thread-context", userId] });
  queryClient.removeQueries({ queryKey: ["memories", userId] });
  queryClient.removeQueries({ queryKey: ["chat-search", userId] });
  for (const runId of runIds) {
    queryClient.removeQueries({ exact: true, queryKey: ["run", runId] });
  }
  queryClient.removeQueries({
    predicate: (query) => {
      if (query.queryKey[0] !== "run") {
        return false;
      }
      return (query.state.data as CachedRun | undefined)?.user_id === userId;
    },
  });
}

export function removeAllProfileCaches(queryClient: QueryClient) {
  const privateQueryRoots = new Set([
    "users",
    "threads",
    "thread-history",
    "thread-runs",
    "thread-context",
    "memories",
    "chat-search",
    "run",
  ]);
  queryClient.removeQueries({
    predicate: (query) => privateQueryRoots.has(String(query.queryKey[0] ?? "")),
  });
}
