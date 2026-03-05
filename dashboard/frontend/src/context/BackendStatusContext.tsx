import React, { createContext, useContext } from "react";

export interface BackendStatusValue {
  backendUp: boolean;
  backendMessage: string;
  lastSuccessAt: number | null;
  nextRetryInMs: number;
}

const BackendStatusContext = createContext<BackendStatusValue>({
  backendUp: true,
  backendMessage: "unknown",
  lastSuccessAt: null,
  nextRetryInMs: 0,
});

export function BackendStatusProvider({
  value,
  children,
}: {
  value: BackendStatusValue;
  children: React.ReactNode;
}) {
  return <BackendStatusContext.Provider value={value}>{children}</BackendStatusContext.Provider>;
}

export function useBackendStatus(): BackendStatusValue {
  return useContext(BackendStatusContext);
}
