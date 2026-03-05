import React, { createContext, useContext, useMemo, useState } from "react";
import type { DashboardAuthContext } from "../api/client";
import { getDashboardAuthContext, setDashboardAuthContext } from "../api/client";

interface AuthContextValue {
  auth: DashboardAuthContext;
  setAuth: React.Dispatch<React.SetStateAction<DashboardAuthContext>>;
}

const Ctx = createContext<AuthContextValue | null>(null);

export function DashboardAuthProvider({ children }: { children: React.ReactNode }) {
  const [auth, setAuthState] = useState<DashboardAuthContext>(() => getDashboardAuthContext());
  const setAuth: React.Dispatch<React.SetStateAction<DashboardAuthContext>> = (next) => {
    setAuthState((prev) => {
      const resolved = typeof next === "function" ? (next as (p: DashboardAuthContext) => DashboardAuthContext)(prev) : next;
      setDashboardAuthContext(resolved);
      return resolved;
    });
  };
  const value = useMemo<AuthContextValue>(() => ({ auth, setAuth }), [auth]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useDashboardAuth() {
  const v = useContext(Ctx);
  if (!v) {
    throw new Error("useDashboardAuth must be used within DashboardAuthProvider");
  }
  return v;
}

