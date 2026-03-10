import React, { createContext, useContext, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";

export interface LogNavOptions {
  file?: string;
  search?: string;
  pack?: string;
}

export interface NavContextValue {
  goToLogs: (opts?: LogNavOptions) => void;
  goToIncidents: (incidentId?: string) => void;
  goToTool: (action?: string) => void;
  goTo: (path: string, params?: Record<string, string>) => void;
}

const NavigationContext = createContext<NavContextValue | null>(null);

export function NavigationProvider({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  // Track last focus so panels can restore
  const _lastRef = useRef<string>("");

  const goToLogs = useCallback((opts?: LogNavOptions) => {
    _lastRef.current = "logs";
    const params = new URLSearchParams();
    if (opts?.file)   params.set("file", opts.file);
    if (opts?.search) params.set("search", opts.search);
    if (opts?.pack)   params.set("pack", opts.pack);
    const qs = params.toString();
    navigate(qs ? `/logs?${qs}` : "/logs");
  }, [navigate]);

  const goToIncidents = useCallback((incidentId?: string) => {
    _lastRef.current = "tower";
    const params = new URLSearchParams();
    if (incidentId) params.set("incident", incidentId);
    const qs = params.toString();
    navigate(qs ? `/tower?${qs}` : "/tower");
  }, [navigate]);

  const goToTool = useCallback((action?: string) => {
    _lastRef.current = "debug";
    const params = new URLSearchParams();
    if (action) params.set("action", action);
    const qs = params.toString();
    navigate(qs ? `/debug?${qs}` : "/debug");
  }, [navigate]);

  const goTo = useCallback((path: string, params?: Record<string, string>) => {
    if (!params || Object.keys(params).length === 0) {
      navigate(path);
      return;
    }
    const qs = new URLSearchParams(params).toString();
    navigate(`${path}?${qs}`);
  }, [navigate]);

  return (
    <NavigationContext.Provider value={{ goToLogs, goToIncidents, goToTool, goTo }}>
      {children}
    </NavigationContext.Provider>
  );
}

export function useNav(): NavContextValue {
  const ctx = useContext(NavigationContext);
  if (!ctx) throw new Error("useNav must be used inside NavigationProvider");
  return ctx;
}
