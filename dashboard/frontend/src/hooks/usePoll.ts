import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useApiErrors } from "../context/ApiErrorContext";

export interface UsePollOptions<T> {
  fetcher: (signal: AbortSignal) => Promise<T>;
  intervalMs: number;
  pollKey?: string;
  enabled?: boolean;
  immediate?: boolean;
  staleAfterMs?: number;
  retryInitialMs?: number;
  retryMaxMs?: number;
  circuitBreakerThreshold?: number;
  circuitBreakerCooldownMs?: number;
}

export interface UsePollResult<T> {
  data: T | null;
  error: Error | null;
  isLoading: boolean;
  isFetching: boolean;
  lastSuccessAt: number | null;
  isStale: boolean;
  failureCount: number;
  nextRetryInMs: number;
  circuitOpen: boolean;
  refresh: () => void;
}

export function usePoll<T>({
  fetcher,
  intervalMs,
  pollKey = "poll",
  enabled = true,
  immediate = true,
  staleAfterMs,
  retryInitialMs = 1000,
  retryMaxMs = 10000,
  circuitBreakerThreshold = 5,
  circuitBreakerCooldownMs = 30000,
}: UsePollOptions<T>): UsePollResult<T> {
  const { pushEvent } = useApiErrors();
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isFetching, setIsFetching] = useState(false);
  const [lastSuccessAt, setLastSuccessAt] = useState<number | null>(null);
  const [failureCount, setFailureCount] = useState(0);
  const [retryAt, setRetryAt] = useState<number | null>(null);
  const [nowMs, setNowMs] = useState<number>(Date.now());
  const [circuitOpenUntil, setCircuitOpenUntil] = useState<number | null>(null);

  const timerRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const stoppedRef = useRef(false);
  const failuresRef = useRef(0);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const clearAbort = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  const schedule = useCallback(
    (delayMs: number, run: () => void) => {
      clearTimer();
      timerRef.current = window.setTimeout(run, Math.max(0, delayMs));
    },
    [clearTimer]
  );

  const runRef = useRef<() => void>(() => {});

  useEffect(() => {
    stoppedRef.current = false;
    failuresRef.current = 0;
    setFailureCount(0);
    setRetryAt(null);
    setCircuitOpenUntil(null);
    setError(null);
    setIsLoading(enabled);
    if (!enabled) {
      setIsLoading(false);
      clearAbort();
      clearTimer();
      return;
    }

    runRef.current = () => {
      if (stoppedRef.current) {
        return;
      }
      if (circuitOpenUntil !== null && Date.now() < circuitOpenUntil) {
        const wait = Math.max(0, circuitOpenUntil - Date.now());
        setRetryAt(Date.now() + wait);
        schedule(wait, runRef.current);
        return;
      }

      clearAbort();
      const controller = new AbortController();
      abortRef.current = controller;
      setIsFetching(true);

      fetcher(controller.signal)
        .then((next) => {
          if (stoppedRef.current) {
            return;
          }
          setData(next);
          setError(null);
          setIsLoading(false);
          setIsFetching(false);
          setLastSuccessAt(Date.now());
          failuresRef.current = 0;
          setFailureCount(0);
          setRetryAt(null);
          setCircuitOpenUntil(null);
          schedule(intervalMs, runRef.current);
        })
        .catch((e: unknown) => {
          if (stoppedRef.current || controller.signal.aborted) {
            return;
          }
          const err = e instanceof Error ? e : new Error(String(e));
          setError(err);
          setIsLoading(false);
          setIsFetching(false);
          failuresRef.current += 1;
          setFailureCount(failuresRef.current);
          const baseBackoff = Math.min(
            retryInitialMs * 2 ** Math.max(0, failuresRef.current - 1),
            retryMaxMs
          );
          let delay = baseBackoff;
          let circuitOpen = false;
          if (circuitBreakerThreshold > 0 && failuresRef.current >= circuitBreakerThreshold) {
            delay = Math.max(baseBackoff, circuitBreakerCooldownMs);
            setCircuitOpenUntil(Date.now() + delay);
            circuitOpen = true;
          }
          setRetryAt(Date.now() + delay);
          pushEvent({
            key: pollKey,
            message: err.message || String(err),
            failureCount: failuresRef.current,
            nextRetryInMs: delay,
            circuitOpen,
          });
          schedule(delay, runRef.current);
        });
    };

    if (immediate) {
      runRef.current();
    } else {
      schedule(intervalMs, runRef.current);
    }

    return () => {
      stoppedRef.current = true;
      clearTimer();
      clearAbort();
    };
  }, [
    clearAbort,
    clearTimer,
    enabled,
    fetcher,
    immediate,
    intervalMs,
    retryInitialMs,
    retryMaxMs,
    schedule,
    circuitOpenUntil,
    circuitBreakerCooldownMs,
    circuitBreakerThreshold,
    pollKey,
    pushEvent,
  ]);

  const refresh = useCallback(() => {
    if (!enabled) {
      return;
    }
    clearTimer();
    runRef.current();
  }, [clearTimer, enabled]);

  const staleWindow = staleAfterMs ?? intervalMs * 2;
  const isStale = useMemo(() => {
    if (!enabled || lastSuccessAt === null) {
      return false;
    }
    return Date.now() - lastSuccessAt > staleWindow;
  }, [enabled, lastSuccessAt, staleWindow]);

  useEffect(() => {
    if (retryAt === null) {
      return;
    }
    const id = window.setInterval(() => setNowMs(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [retryAt]);

  const nextRetryInMs = useMemo(() => {
    if (retryAt === null) {
      return 0;
    }
    return Math.max(0, retryAt - nowMs);
  }, [retryAt, nowMs]);

  const circuitOpen = useMemo(() => {
    if (circuitOpenUntil === null) {
      return false;
    }
    return nowMs < circuitOpenUntil;
  }, [circuitOpenUntil, nowMs]);

  return {
    data,
    error,
    isLoading,
    isFetching,
    lastSuccessAt,
    isStale,
    failureCount,
    nextRetryInMs,
    circuitOpen,
    refresh,
  };
}
