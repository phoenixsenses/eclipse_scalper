import React, { createContext, useCallback, useContext, useMemo, useState } from "react";

export interface ApiErrorEvent {
  id: string;
  ts: number;
  key: string;
  message: string;
  failureCount: number;
  nextRetryInMs: number;
  circuitOpen: boolean;
}

interface ApiErrorContextValue {
  events: ApiErrorEvent[];
  pushEvent: (evt: Omit<ApiErrorEvent, "id" | "ts">) => void;
  clear: () => void;
}

const Ctx = createContext<ApiErrorContextValue>({
  events: [],
  pushEvent: () => {},
  clear: () => {},
});

export function ApiErrorProvider({ children }: { children: React.ReactNode }) {
  const [events, setEvents] = useState<ApiErrorEvent[]>([]);

  const pushEvent = useCallback((evt: Omit<ApiErrorEvent, "id" | "ts">) => {
    setEvents((prev) => {
      const now = Date.now();
      const next: ApiErrorEvent = {
        id: `${evt.key}_${now}_${Math.random().toString(16).slice(2, 8)}`,
        ts: now,
        ...evt,
      };
      return [next, ...prev].slice(0, 100);
    });
  }, []);

  const clear = useCallback(() => setEvents([]), []);

  const value = useMemo<ApiErrorContextValue>(() => ({ events, pushEvent, clear }), [events, pushEvent, clear]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApiErrors() {
  return useContext(Ctx);
}

