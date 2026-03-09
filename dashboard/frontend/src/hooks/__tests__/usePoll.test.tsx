import type { ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { usePoll } from "../usePoll";
import { ApiErrorProvider } from "../../context/ApiErrorContext";

function wrapper({ children }: { children: ReactNode }) {
  return <ApiErrorProvider>{children}</ApiErrorProvider>;
}

describe("usePoll", () => {
  it("does not restart polling when the fetcher identity changes after a rerender", async () => {
    const calls: AbortSignal[] = [];
    const fetcher = vi.fn((signal: AbortSignal) => {
      calls.push(signal);
      return Promise.resolve({ ok: true });
    });

    const { rerender } = renderHook(
      ({ version }) =>
        usePoll({
          fetcher: (signal) => fetcher(signal),
          pollKey: "poll:test",
          intervalMs: 10_000,
          staleAfterMs: 20_000,
        }),
      {
        initialProps: { version: 1 },
        wrapper,
      }
    );

    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
    expect(calls[0]?.aborted).toBe(false);

    rerender({ version: 2 });

    await act(async () => {
      await Promise.resolve();
    });

    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(calls[0]?.aborted).toBe(false);
  });

  it("keeps the error state visible after a failure instead of resetting to loading on retry setup", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn(() => Promise.reject(new Error("boom")));

    const { result } = renderHook(
      () =>
        usePoll({
          fetcher,
          pollKey: "poll:failure",
          intervalMs: 10_000,
          retryInitialMs: 1_000,
          retryMaxMs: 1_000,
      }),
      { wrapper }
    );

    await act(async () => {
      await Promise.resolve();
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.error?.message).toBe("boom");
    expect(result.current.failureCount).toBe(1);

    await act(async () => {
      vi.advanceTimersByTime(500);
      await Promise.resolve();
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.error?.message).toBe("boom");

    vi.useRealTimers();
  }, 10000);
});
