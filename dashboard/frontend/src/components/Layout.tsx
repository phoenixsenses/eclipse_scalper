import React, { useEffect, useMemo, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import GlossaryDrawer from "./GlossaryDrawer";
import OnboardingTour from "./OnboardingTour";
import { useDashboardAuth } from "../context/AuthContext";
import { usePoll } from "../hooks/usePoll";
import { api } from "../api/client";
import type { SecurityAuditEvent } from "../api/types";
import { BackendStatusProvider } from "../context/BackendStatusContext";
import { useApiErrors } from "../context/ApiErrorContext";

const NAV: { to: string; label: string }[] = [
  { to: "/",         label: "Genel Bakis / Overview"  },
  { to: "/tower",    label: "Kontrol Kulesi / Tower"  },
  { to: "/logs",     label: "Loglar / Logs"           },
  { to: "/trades",   label: "Islemler / Trades"       },
  { to: "/debug",    label: "Hata Ayikla / Debug"     },
  { to: "/settings", label: "Ayarlar / Settings"      },
];

const navStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 2,
  padding: "0 12px",
  height: 40,
  background: "var(--surface)",
  borderBottom: "1px solid var(--border)",
};

const logoStyle: React.CSSProperties = {
  fontWeight: 700,
  fontSize: 14,
  color: "var(--accent)",
  marginRight: 24,
  letterSpacing: "0.02em",
};

function NavItem({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      style={({ isActive }) => ({
        padding: "4px 10px",
        borderRadius: 4,
        fontSize: 13,
        color: isActive ? "var(--text)" : "var(--muted)",
        background: isActive ? "#21262d" : "transparent",
        transition: "background 0.1s",
      })}
    >
      {label}
    </NavLink>
  );
}

export default function Layout() {
  const navigate = useNavigate();
  const { auth, setAuth } = useDashboardAuth();
  const [glossaryOpen, setGlossaryOpen] = useState(false);
  const [opsOpen, setOpsOpen] = useState(false);
  const [apiPanelOpen, setApiPanelOpen] = useState(false);
  const [copyHint, setCopyHint] = useState<string>("");
  const { events: apiEvents, clear: clearApiEvents } = useApiErrors();
  const [denseMode, setDenseMode] = useState<boolean>(() => {
    try {
      return localStorage.getItem("eclipse.ui.dense") === "1";
    } catch {
      return false;
    }
  });
  const location = useLocation();
  const backendPoll = usePoll({
    fetcher: (signal) => api.health(signal),
    pollKey: "api:/health",
    intervalMs: 5000,
    staleAfterMs: 15000,
    retryInitialMs: 1000,
    retryMaxMs: 30000,
  });
  const backendUp = !backendPoll.error && !backendPoll.isStale;
  const backendBadge = backendUp ? "UP" : "DOWN";
  const backendMessage = backendPoll.error?.message ?? (backendPoll.isStale ? "stale" : "ok");
  const backendCommands = useMemo(
    () => [
      {
        label: "Start backend supervisor",
        tr: "Backend supervisor baslat",
        command:
          "powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\run_dashboard_backend_supervisor.ps1",
      },
      {
        label: "Start full dashboard",
        tr: "Tum dashboard stacki baslat",
        command: "powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\run_dashboard.ps1",
      },
      {
        label: "Check backend health",
        tr: "Backend health kontrol et",
        command: "Invoke-WebRequest http://127.0.0.1:8765/api/health -UseBasicParsing",
      },
    ],
    [],
  );
  const secPoll = usePoll<SecurityAuditEvent[]>({
    fetcher: (signal) => api.debugSecurityAudit(30, signal),
    pollKey: "api:/debug/security-audit",
    intervalMs: 15_000,
    staleAfterMs: 45_000,
    enabled: backendUp,
  });
  const secHealth = useMemo(() => {
    const rows = secPoll.data ?? [];
    const cutoff = Date.now() / 1000 - 15 * 60;
    const recent = rows.filter((r) => (r.ts ?? 0) >= cutoff);
    const byKind = (k: string) => recent.filter((r) => r.kind === k).length;
    return { authFailed: byKind("auth_failed"), roleDenied: byKind("role_denied"), rateLimited: byKind("rate_limited") };
  }, [secPoll.data]);
  const helpText = useMemo(() => {
    const p = location.pathname;
    if (p.startsWith("/tower")) return "Control Tower: one-screen runtime health, incidents, rate-limit, and quick operations.";
    if (p.startsWith("/logs")) return "Logs: select file, then filter by level/search to isolate issues fast.";
    if (p.startsWith("/trades")) return "Trades: inspect signal/stability/quality decisions and blockers.";
    if (p.startsWith("/debug")) return "Debug: run safe diagnostics (validate_env, preflight, status) from UI.";
    if (p.startsWith("/settings")) return "Settings: verify env/runtime config and backend health.";
    return "Overview: monitor health, data freshness, and entry gating status.";
  }, [location.pathname]);

  useEffect(() => {
    const root = document.body;
    if (denseMode) root.classList.add("dense");
    else root.classList.remove("dense");
    try {
      localStorage.setItem("eclipse.ui.dense", denseMode ? "1" : "0");
    } catch {
      // best effort
    }
  }, [denseMode]);

  useEffect(() => {
    if (!copyHint) return;
    const t = window.setTimeout(() => setCopyHint(""), 2000);
    return () => window.clearTimeout(t);
  }, [copyHint]);

  async function copyCommand(value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setCopyHint("Command copied");
    } catch {
      setCopyHint("Copy failed");
    }
  }

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <nav style={navStyle}>
        <span style={logoStyle}>ECLIPSE SCALPER</span>
        {NAV.map((n) => <NavItem key={n.to} {...n} />)}
        <span
          style={{
            marginLeft: "auto",
            padding: "4px 8px",
            borderRadius: 999,
            border: "1px solid var(--border)",
            background: backendUp ? "#1d3a25" : "#3a1d1d",
            color: "var(--text)",
            fontSize: 11,
          }}
          title={`backend=${backendMessage}${!backendUp && backendPoll.nextRetryInMs > 0 ? ` | retry=${Math.ceil(backendPoll.nextRetryInMs / 1000)}s` : ""}`}
        >
          API {backendBadge}
          {!backendUp && backendPoll.nextRetryInMs > 0 ? ` (${Math.ceil(backendPoll.nextRetryInMs / 1000)}s)` : ""}
        </span>
        <button
          onClick={() => setDenseMode((v) => !v)}
          style={{
            marginLeft: 8,
            padding: "4px 8px",
            borderRadius: 4,
            border: "1px solid var(--border)",
            background: denseMode ? "var(--surface-2)" : "transparent",
            color: denseMode ? "var(--text)" : "var(--muted)",
            cursor: "pointer",
            fontSize: 11,
          }}
          title="Toggle dense table layout"
        >
          {denseMode ? "Dense ON" : "Dense OFF"}
        </button>
        <button
          onClick={() => setOpsOpen(true)}
          style={{
            marginLeft: 8,
            padding: "4px 8px",
            borderRadius: 4,
            border: "1px solid var(--border)",
            background: "transparent",
            color: "var(--muted)",
            cursor: "pointer",
            fontSize: 11,
          }}
          title="Open Ops Command Palette"
        >
          OPS
        </button>
        <button
          onClick={() => setGlossaryOpen(true)}
          style={{
            marginLeft: 8,
            padding: "4px 8px",
            borderRadius: 4,
            border: "1px solid var(--border)",
            background: "transparent",
            color: "var(--muted)",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          Glossary
        </button>
        <button
          onClick={() => setApiPanelOpen((v) => !v)}
          style={{
            marginLeft: 8,
            padding: "4px 8px",
            borderRadius: 999,
            border: "1px solid var(--border)",
            background: apiEvents.length > 0 ? "#3a1d1d" : "transparent",
            color: apiEvents.length > 0 ? "var(--text)" : "var(--muted)",
            cursor: "pointer",
            fontSize: 11,
          }}
          title="API failure panel"
        >
          API ERR {apiEvents.length}
        </button>
        <button
          onClick={() => setAuth((p) => ({ ...p, role: p.role === "viewer" ? "admin" : "viewer" }))}
          title={`operator=${auth.operator} role=${auth.role}`}
          style={{
            marginLeft: 8,
            padding: "4px 8px",
            borderRadius: 999,
            border: "1px solid var(--border)",
            background: auth.role === "viewer" ? "#3a1d1d" : "#1d3a25",
            color: "var(--text)",
            cursor: "pointer",
            fontSize: 11,
          }}
        >
          SEC {auth.role.toUpperCase()} | A:{secHealth.authFailed} R:{secHealth.roleDenied} L:{secHealth.rateLimited}
        </button>
      </nav>
      <div
        style={{
          padding: "8px 16px",
          borderBottom: "1px solid var(--border)",
          background: "var(--bg)",
          color: "var(--muted)",
          fontSize: 12,
        }}
      >
        Help: {helpText}
        <div className="legend-row" style={{ marginTop: 6 }}>
          <span className="legend-chip">Healthy</span>
          <span className="legend-chip">Degraded</span>
          <span className="legend-chip">Critical</span>
          <span className="legend-chip">Paper Mode</span>
          <span className="legend-chip">Regime</span>
        </div>
      </div>
      {!backendUp && (
        <div
          style={{
            padding: "8px 16px",
            borderBottom: "1px solid var(--border)",
            background: "#3a1d1d",
            color: "var(--text)",
            fontSize: 12,
          }}
        >
          API degraded: backend unreachable ({backendMessage}). Auto-retry in {Math.max(1, Math.ceil(backendPoll.nextRetryInMs / 1000))}s.
          {backendPoll.lastSuccessAt ? ` Last success: ${new Date(backendPoll.lastSuccessAt).toLocaleTimeString()}` : " No successful poll yet."}
          <div className="recovery-panel">
            <div className="recovery-title">Quick Recovery / Hizli Kurtarma</div>
            <div className="recovery-steps">
              <div>1. Open a new PowerShell window in repo root.</div>
              <div>2. Run one command below (copy button).</div>
              <div>3. Wait until top-right API badge turns UP.</div>
            </div>
            <div className="recovery-cmd-grid">
              {backendCommands.map((c) => (
                <div key={c.command} className="recovery-cmd-card">
                  <div className="recovery-cmd-head">
                    <span>{c.label}</span>
                    <span style={{ color: "var(--muted)", fontSize: 10 }}>{c.tr}</span>
                  </div>
                  <code className="recovery-code">{c.command}</code>
                  <button className="guide-toggle" onClick={() => copyCommand(c.command)}>
                    Copy
                  </button>
                </div>
              ))}
            </div>
            <div className="recovery-next">
              <div className="recovery-next-title">After Backend Is UP / Backend UP olduktan sonra</div>
              <div className="recovery-next-actions">
                <button
                  className={`guide-toggle recovery-next-btn${backendUp ? " ready" : ""}`}
                  onClick={() => navigate("/tower")}
                  title={backendUp ? "Backend is up: proceed" : "Wait for API UP, then proceed"}
                >
                  Open Control Tower
                </button>
                <button
                  className={`guide-toggle recovery-next-btn${backendUp ? " ready" : ""}`}
                  onClick={() => navigate("/logs")}
                  title={backendUp ? "Backend is up: proceed" : "Wait for API UP, then proceed"}
                >
                  Open Logs
                </button>
                <button
                  className={`guide-toggle recovery-next-btn${backendUp ? " ready" : ""}`}
                  onClick={() => navigate("/debug?auto=triage")}
                  title={backendUp ? "Backend is up: proceed" : "Wait for API UP, then proceed"}
                >
                  Run Debug Triage
                </button>
              </div>
            </div>
            {copyHint ? <div className="recovery-copy-hint">{copyHint}</div> : null}
          </div>
        </div>
      )}
      {backendUp && (
        <div
          style={{
            padding: "6px 16px",
            borderBottom: "1px solid var(--border)",
            background: "rgba(20, 52, 31, 0.55)",
          }}
        >
          <button
            className="recovery-ready-badge"
            title="Backend reachable and polling healthy. Open Control Tower."
            onClick={() => navigate("/tower")}
          >
            API UP - safe to continue
          </button>
        </div>
      )}
      <BackendStatusProvider
        value={{
          backendUp,
          backendMessage,
          lastSuccessAt: backendPoll.lastSuccessAt,
          nextRetryInMs: backendPoll.nextRetryInMs,
        }}
      >
        <main style={{ flex: 1, padding: 14, maxWidth: 1280, margin: "0 auto", width: "100%" }}>
          <Outlet />
        </main>
      </BackendStatusProvider>
      <GlossaryDrawer open={glossaryOpen} onClose={() => setGlossaryOpen(false)} />
      <OnboardingTour />
      {apiPanelOpen && (
        <div
          onClick={() => setApiPanelOpen(false)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 9998,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="card"
            style={{ width: 760, maxWidth: "94vw", maxHeight: "80vh", overflowY: "auto" }}
          >
            <div className="card-title">API Failure Panel (last 10)</div>
            <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
              <button className="guide-toggle" onClick={() => clearApiEvents()}>Clear</button>
            </div>
            {apiEvents.length === 0 ? (
              <div style={{ color: "var(--muted)", fontSize: 12 }}>No recent API failures.</div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>time</th>
                    <th>endpoint</th>
                    <th>message</th>
                    <th>fails</th>
                    <th>retry_s</th>
                    <th>circuit</th>
                  </tr>
                </thead>
                <tbody>
                  {apiEvents.slice(0, 10).map((e) => (
                    <tr key={e.id}>
                      <td style={{ color: "var(--muted)", fontSize: 12 }}>{new Date(e.ts).toLocaleTimeString()}</td>
                      <td>{e.key}</td>
                      <td style={{ color: "var(--muted)", fontSize: 12 }}>{e.message}</td>
                      <td>{e.failureCount}</td>
                      <td>{Math.ceil(e.nextRetryInMs / 1000)}</td>
                      <td>{e.circuitOpen ? "OPEN" : "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}
      {opsOpen && (
        <div
          onClick={() => setOpsOpen(false)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 9999,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="card"
            style={{ width: 620, maxWidth: "92vw", display: "flex", flexDirection: "column", gap: 8 }}
          >
            <div className="card-title">Ops Command Palette</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              <button className="guide-toggle" onClick={() => { setOpsOpen(false); navigate("/debug?auto=triage"); }}>Run Triage Macro</button>
              <button className="guide-toggle" onClick={() => { setOpsOpen(false); navigate("/logs?pack=No%20Match"); }}>Logs: No Match</button>
              <button className="guide-toggle" onClick={() => { setOpsOpen(false); navigate("/logs?pack=Regime"); }}>Logs: Regime</button>
              <button className="guide-toggle" onClick={() => { setOpsOpen(false); navigate("/logs?pack=Shutdown"); }}>Logs: Shutdown</button>
              <button className="guide-toggle" onClick={() => { setOpsOpen(false); navigate("/logs?pack=Timeout"); }}>Logs: Timeout</button>
              <button className="guide-toggle" onClick={() => { setOpsOpen(false); navigate("/logs?ops=start_incident"); }}>Start Incident Session</button>
              <button className="guide-toggle" onClick={() => { setOpsOpen(false); navigate("/logs?ops=stop_incident"); }}>Stop Incident Session</button>
              <button className="guide-toggle" onClick={() => { setOpsOpen(false); navigate("/logs?ops=export_bundle"); }}>Export Bundle</button>
              <button className="guide-toggle" onClick={() => { setOpsOpen(false); navigate("/logs?ops=copy_summary"); }}>Copy Triage Summary</button>
              <button className="guide-toggle" onClick={() => { setOpsOpen(false); navigate("/logs?ops=force_recover"); }}>Force Recover</button>
              <button className="guide-toggle" onClick={() => { setOpsOpen(false); navigate("/logs?ops=keep_fallback_on"); }}>Keep Fallback ON</button>
              <button className="guide-toggle" onClick={() => { setOpsOpen(false); navigate("/logs?ops=keep_fallback_off"); }}>Keep Fallback OFF</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
