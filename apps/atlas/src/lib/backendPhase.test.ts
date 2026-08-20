import { describe, expect, it } from "vitest";

import { resolveBackendPhase } from "./backendPhase";

describe("resolveBackendPhase", () => {
  it("returns online when a backend status payload exists", () => {
    expect(
      resolveBackendPhase({
        hasStatus: true,
        isError: false,
        isPending: false,
        isFetching: false,
        bootStartedAt: 10,
        now: 50_000,
      }),
    ).toBe("online");
  });

  it("returns offline when a refresh fails even if an older status payload is cached", () => {
    expect(
      resolveBackendPhase({
        hasStatus: true,
        isError: true,
        isPending: false,
        isFetching: false,
        bootStartedAt: 10,
        now: 50_000,
      }),
    ).toBe("offline");
  });

  it("returns starting while the first status query is pending", () => {
    expect(
      resolveBackendPhase({
        hasStatus: false,
        isError: false,
        isPending: true,
        isFetching: false,
        bootStartedAt: 10,
        now: 50_000,
      }),
    ).toBe("starting");
  });

  it("returns starting during the startup grace window", () => {
    expect(
      resolveBackendPhase({
        hasStatus: false,
        isError: false,
        isPending: false,
        isFetching: false,
        bootStartedAt: 1_000,
        now: 10_000,
      }),
    ).toBe("starting");
  });

  it("returns offline after the grace window expires without status", () => {
    expect(
      resolveBackendPhase({
        hasStatus: false,
        isError: false,
        isPending: false,
        isFetching: false,
        bootStartedAt: 1_000,
        now: 16_001,
      }),
    ).toBe("offline");
  });
});
