import { useEffect, useMemo, useRef, useState } from "react";

export type SSEStatus = "idle" | "connecting" | "open" | "reconnecting" | "closed" | "error";

export interface UseSSEOptions {
  enabled: boolean;
  connect: () => EventSource;
  onMessage: (data: string, event: MessageEvent) => void;
  reconnectInitialMs?: number;
  reconnectMaxMs?: number;
  staleAfterMs?: number;
}

export interface UseSSEResult {
  status: SSEStatus;
  lastEventAt: number | null;
  reconnectAttempt: number;
  isStale: boolean;
  error: Error | null;
}

export function useSSE({
  enabled,
  connect,
  onMessage,
  reconnectInitialMs = 1000,
  reconnectMaxMs = 10000,
  staleAfterMs = 10000,
}: UseSSEOptions): UseSSEResult {
  const [status, setStatus] = useState<SSEStatus>("idle");
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const [error, setError] = useState<Error | null>(null);

  const sourceRef = useRef<EventSource | null>(null);
  const timerRef = useRef<number | null>(null);
  const stoppedRef = useRef(false);

  const connectRef = useRef(connect);
  connectRef.current = connect;
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    stoppedRef.current = false;
    if (!enabled) {
      setStatus("idle");
      setReconnectAttempt(0);
      setError(null);
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      if (sourceRef.current) {
        sourceRef.current.close();
        sourceRef.current = null;
      }
      return;
    }

    const cleanup = () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      if (sourceRef.current) {
        sourceRef.current.close();
        sourceRef.current = null;
      }
    };

    const open = (attempt: number) => {
      if (stoppedRef.current) {
        return;
      }
      setStatus(attempt === 0 ? "connecting" : "reconnecting");
      setReconnectAttempt(attempt);

      const es = connectRef.current();
      sourceRef.current = es;

      es.onopen = () => {
        if (stoppedRef.current) {
          return;
        }
        setStatus("open");
        setError(null);
        setReconnectAttempt(0);
      };

      es.onmessage = (event) => {
        if (stoppedRef.current) {
          return;
        }
        setLastEventAt(Date.now());
        onMessageRef.current(event.data, event);
      };

      es.onerror = () => {
        if (stoppedRef.current) {
          return;
        }
        setStatus("error");
        setError(new Error("SSE stream disconnected"));
        cleanup();
        const nextAttempt = attempt + 1;
        const delay = Math.min(reconnectInitialMs * 2 ** Math.max(0, nextAttempt - 1), reconnectMaxMs);
        timerRef.current = window.setTimeout(() => open(nextAttempt), delay);
      };
    };

    open(0);
    return () => {
      stoppedRef.current = true;
      cleanup();
    };
  }, [enabled, reconnectInitialMs, reconnectMaxMs]);

  const isStale = useMemo(() => {
    if (!enabled || lastEventAt === null) {
      return false;
    }
    return Date.now() - lastEventAt > staleAfterMs;
  }, [enabled, lastEventAt, staleAfterMs]);

  return { status, lastEventAt, reconnectAttempt, isStale, error };
}
