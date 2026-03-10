import React, { useEffect, useMemo, useState, useCallback } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import GlossaryDrawer from "./GlossaryDrawer";
import OnboardingTour from "./OnboardingTour";
import { useDashboardAuth } from "../context/AuthContext";
import { usePoll } from "../hooks/usePoll";
import { api } from "../api/client";
import type { SecurityAuditEvent } from "../api/types";
import { BackendStatusProvider } from "../context/BackendStatusContext";
import { useApiErrors } from "../context/ApiErrorContext";
import { NavigationProvider } from "../context/NavigationContext";
import CommandPalette from "./CommandPalette";

// ─── Nav structure ────────────────────────────────────────────
interface NavItem { to: string; label: string; icon: string; desc: string; }

const NAV: { section: string; items: NavItem[] }[] = [
  {
    section: "MONITOR",
    items: [
      { to: "/",         label: "Status",    icon: "◉", desc: "System health overview" },
      { to: "/live",     label: "Live",      icon: "▶", desc: "Real-time microstructure" },
      { to: "/research", label: "Research",  icon: "◆", desc: "Pockets & watchboard" },
    ],
  },
  {
    section: "OPERATIONS",
    items: [
      { to: "/tower",    label: "Incidents", icon: "!", desc: "Incident inbox & runbooks" },
      { to: "/debug",    label: "Tools",     icon: "⚙", desc: "Debug actions & history" },
      { to: "/recovery", label: "Recovery",  icon: "↺", desc: "Guided recovery wizard" },
    ],
  },
  {
    section: "DATA",
    items: [
      { to: "/logs",     label: "Logs",      icon: "≡", desc: "Log browser & streamer" },
      { to: "/trades",   label: "Signals",   icon: "~", desc: "Trade signals & events" },
    ],
  },
  {
    section: "SYSTEM",
    items: [
      { to: "/settings", label: "Settings",  icon: "⚙", desc: "Config & diagnostics" },
    ],
  },
];

const PAGE_TITLES: Record<string, string> = {
  "/":         "Status Overview",
  "/live":     "Live Monitor",
  "/research": "Research",
  "/tower":    "Control Tower — Incidents",
  "/debug":    "Tools",
  "/recovery": "Recovery Wizard",
  "/logs":     "Logs",
  "/trades":   "Signals & Events",
  "/settings": "Settings",
};

// ─── Sidebar ──────────────────────────────────────────────────
function Sidebar({
  collapsed,
  mobileOpen,
  onToggle,
  onMobileClose,
  backendUp,
  backendMessage,
  incidentCount,
}: {
  collapsed: boolean;
  mobileOpen: boolean;
  onToggle: () => void;
  onMobileClose: () => void;
  backendUp: boolean;
  backendMessage: string;
  incidentCount: number;
}) {
  const sidebarClass = [
    "sidebar",
    collapsed ? "collapsed" : "",
    mobileOpen ? "mobile-open" : "",
  ].filter(Boolean).join(" ");

  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          onClick={onMobileClose}
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)", zIndex: 199 }}
        />
      )}

      <aside className={sidebarClass}>
        {/* Logo */}
        <div className="sidebar-logo">
          <div className="sidebar-logo-mark">E</div>
          {!collapsed && (
            <div className="sidebar-logo-text">
              Eclipse <span>Scalper</span>
            </div>
          )}
          <button className="sidebar-collapse-btn" onClick={onToggle} title={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
            {collapsed ? "»" : "«"}
          </button>
        </div>

        {/* Nav sections */}
        <nav className="sidebar-nav">
          {NAV.map((section) => (
            <div key={section.section} className="sidebar-section">
              <div className="sidebar-section-label">{section.section}</div>
              {section.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === "/"}
                  className={({ isActive }) =>
                    "sidebar-nav-item self-help" + (isActive ? " active" : "")
                  }
                  data-help={item.desc}
                  onClick={mobileOpen ? onMobileClose : undefined}
                >
                  <span className="sidebar-nav-icon">{item.icon}</span>
                  <span className="sidebar-nav-label">{item.label}</span>
                  {item.to === "/tower" && incidentCount > 0 && (
                    <span className="sidebar-badge">{incidentCount > 99 ? "99+" : incidentCount}</span>
                  )}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        {/* Footer: API status */}
        <div className="sidebar-footer">
          <div
            className={`sidebar-api-pill ${backendUp ? "up" : "down"}`}
            title={backendMessage}
          >
            <span className="sidebar-api-dot" />
            {!collapsed && <span>API {backendUp ? "LIVE" : "DOWN"}</span>}
          </div>
        </div>
      </aside>
    </>
  );
}

// ─── TopBar ───────────────────────────────────────────────────
function TopBar({
  title,
  backendUp,
  onOpenCmd,
  onOpenMobileSidebar,
  onToggleDense,
  denseMode,
  apiErrCount,
  onOpenApiPanel,
  auth,
  onToggleRole,
}: {
  title: string;
  backendUp: boolean;
  onOpenCmd: () => void;
  onOpenMobileSidebar: () => void;
  onToggleDense: () => void;
  denseMode: boolean;
  apiErrCount: number;
  onOpenApiPanel: () => void;
  auth: { role: string; operator: string };
  onToggleRole: () => void;
}) {
  return (
    <header className="topbar">
      {/* Mobile hamburger */}
      <button
        onClick={onOpenMobileSidebar}
        style={{
          display: "none",
          background: "none",
          border: "1px solid var(--border)",
          borderRadius: 5,
          color: "var(--muted)",
          width: 30,
          height: 30,
          cursor: "pointer",
          fontSize: 16,
          alignItems: "center",
          justifyContent: "center",
        }}
        className="mobile-menu-btn"
        title="Open navigation"
      >
        ☰
      </button>

      <div className="topbar-title">{title}</div>

      {/* Command palette trigger */}
      <button className="topbar-cmd-trigger self-help" data-help="Search pages, tools, log filters" onClick={onOpenCmd}>
        <span>Search</span>
        <kbd>Ctrl</kbd><kbd>K</kbd>
      </button>

      <div className="topbar-divider" />

      <button
        className={`topbar-btn self-help${denseMode ? "" : ""}`}
        data-help="Dense mode: compact rows"
        onClick={onToggleDense}
        title="Toggle dense layout"
        style={{ background: denseMode ? "var(--surface-3)" : undefined, color: denseMode ? "var(--text)" : undefined }}
      >
        {denseMode ? "Dense" : "Compact"}
      </button>

      {apiErrCount > 0 && (
        <button
          className="topbar-btn danger-pill self-help"
          data-help="Recent API failures — click to inspect"
          onClick={onOpenApiPanel}
        >
          {apiErrCount} err{apiErrCount !== 1 ? "s" : ""}
        </button>
      )}

      <button
        className={`topbar-btn self-help ${auth.role === "admin" ? "ok-pill" : "warn-pill"}`}
        data-help="Toggle viewer/admin role for write operations"
        onClick={onToggleRole}
        title={`operator=${auth.operator} role=${auth.role}`}
      >
        {auth.role.toUpperCase()}
      </button>
    </header>
  );
}

// ─── StatusBar ────────────────────────────────────────────────
function StatusBar({
  backendUp,
  backendMessage,
  nextRetryInMs,
  lastSuccessAt,
  secHealth,
  onOpenGlossary,
  onOpenOps,
}: {
  backendUp: boolean;
  backendMessage: string;
  nextRetryInMs: number;
  lastSuccessAt: number | null;
  secHealth: { authFailed: number; roleDenied: number; rateLimited: number };
  onOpenGlossary: () => void;
  onOpenOps: () => void;
}) {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <footer className="statusbar">
      <div className={`statusbar-item ${backendUp ? "up" : "down"}`} title={backendMessage}>
        <span className="statusbar-dot" />
        <span>
          {backendUp
            ? "API LIVE"
            : `API DOWN${nextRetryInMs > 0 ? ` (${Math.ceil(nextRetryInMs / 1000)}s)` : ""}`}
        </span>
      </div>

      {lastSuccessAt && (
        <div className="statusbar-item" title="Last successful API poll">
          last {new Date(lastSuccessAt).toLocaleTimeString()}
        </div>
      )}

      {(secHealth.authFailed > 0 || secHealth.roleDenied > 0) && (
        <div className="statusbar-item warn" title="Security events in last 15 min">
          auth:{secHealth.authFailed} deny:{secHealth.roleDenied} rate:{secHealth.rateLimited}
        </div>
      )}

      <div className="statusbar-item accent" style={{ cursor: "pointer" }} onClick={onOpenOps} title="Quick ops palette">
        OPS
      </div>
      <div className="statusbar-item" style={{ cursor: "pointer" }} onClick={onOpenGlossary} title="Glossary">
        GLO
      </div>
      <div className="statusbar-item" style={{ marginLeft: "auto", borderRight: "none" }}>
        {now.toLocaleTimeString()}
      </div>
    </footer>
  );
}

// ─── Layout (root) ────────────────────────────────────────────
export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { auth, setAuth } = useDashboardAuth();
  const { events: apiEvents, clear: clearApiEvents } = useApiErrors();

  const [sidebarCollapsed, setSidebarCollapsed]   = useState(() => window.innerWidth < 768);
  const [mobileOpen, setMobileOpen]               = useState(false);
  const [cmdOpen, setCmdOpen]                     = useState(false);
  const [glossaryOpen, setGlossaryOpen]           = useState(false);
  const [opsOpen, setOpsOpen]                     = useState(false);
  const [apiPanelOpen, setApiPanelOpen]           = useState(false);
  const [copyHint, setCopyHint]                   = useState("");
  const [incidentCount, setIncidentCount]         = useState(0);

  const [denseMode, setDenseMode] = useState<boolean>(() => {
    try { return localStorage.getItem("eclipse.ui.dense") === "1"; } catch { return false; }
  });

  // Backend health polling
  const backendPoll = usePoll({
    fetcher: (signal) => api.health(signal),
    pollKey: "api:/health",
    intervalMs: 5000,
    staleAfterMs: 15000,
    retryInitialMs: 1000,
    retryMaxMs: 30000,
  });
  const backendUp = !backendPoll.error && !backendPoll.isStale;
  const backendMessage = backendPoll.error?.message ?? (backendPoll.isStale ? "stale" : "ok");

  // Security audit polling
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

  // Incident count polling (for sidebar badge)
  const incidentPoll = usePoll({
    fetcher: (signal) => api.debugIncidents(10, signal),
    pollKey: "api:/debug/incidents/badge",
    intervalMs: 15_000,
    staleAfterMs: 45_000,
    enabled: backendUp,
  });
  useEffect(() => {
    const rows = incidentPoll.data ?? [];
    const open = rows.filter((r) => r.status !== "resolved" && r.status !== "muted").length;
    setIncidentCount(open);
  }, [incidentPoll.data]);

  // Dense mode
  useEffect(() => {
    document.body.classList.toggle("dense", denseMode);
    try { localStorage.setItem("eclipse.ui.dense", denseMode ? "1" : "0"); } catch { /**/ }
  }, [denseMode]);

  // Ctrl+K shortcut
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setCmdOpen((v) => !v);
      }
      if (e.key === "Escape" && cmdOpen) setCmdOpen(false);
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [cmdOpen]);

  // Copy hint auto-clear
  useEffect(() => {
    if (!copyHint) return;
    const t = window.setTimeout(() => setCopyHint(""), 2000);
    return () => window.clearTimeout(t);
  }, [copyHint]);

  async function copyCommand(value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setCopyHint("Copied!");
    } catch {
      setCopyHint("Copy failed");
    }
  }

  const toggleDense = useCallback(() => setDenseMode((v) => !v), []);
  const toggleRole  = useCallback(() => setAuth((p) => ({ ...p, role: p.role === "viewer" ? "admin" : "viewer" })), [setAuth]);

  const title = PAGE_TITLES[location.pathname] ?? "Eclipse Scalper";

  const backendCommands = [
    { label: "Start backend supervisor", tr: "Backend supervisor baslat", command: "powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\run_dashboard_backend_supervisor.ps1" },
    { label: "Start full dashboard",     tr: "Tum dashboard stacki baslat", command: "powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\run_dashboard.ps1" },
    { label: "Check backend health",     tr: "Backend health kontrol et", command: "Invoke-WebRequest http://127.0.0.1:8765/api/health -UseBasicParsing" },
  ];

  return (
    <NavigationProvider>
      <div className="app-shell">
        {/* Sidebar */}
        <Sidebar
          collapsed={sidebarCollapsed}
          mobileOpen={mobileOpen}
          onToggle={() => setSidebarCollapsed((v) => !v)}
          onMobileClose={() => setMobileOpen(false)}
          backendUp={backendUp}
          backendMessage={backendMessage}
          incidentCount={incidentCount}
        />

        {/* Main area */}
        <div className="main-area">
          {/* Top bar */}
          <TopBar
            title={title}
            backendUp={backendUp}
            onOpenCmd={() => setCmdOpen(true)}
            onOpenMobileSidebar={() => setMobileOpen(true)}
            onToggleDense={toggleDense}
            denseMode={denseMode}
            apiErrCount={apiEvents.length}
            onOpenApiPanel={() => setApiPanelOpen(true)}
            auth={auth}
            onToggleRole={toggleRole}
          />

          {/* Backend down banner */}
          {!backendUp && (
            <div style={{ padding: "10px 16px", borderBottom: "1px solid rgba(255,64,64,0.2)", background: "rgba(255,64,64,0.06)", fontSize: 12 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--red)", display: "inline-block" }} />
                <span style={{ color: "var(--red)", fontWeight: 700 }}>API degraded</span>
                <span style={{ color: "var(--muted)" }}>
                  {backendMessage}
                  {backendPoll.nextRetryInMs > 0 ? ` — retry in ${Math.ceil(backendPoll.nextRetryInMs / 1000)}s` : ""}
                </span>
              </div>
              <div className="recovery-cmd-grid">
                {backendCommands.map((c) => (
                  <div key={c.command} className="recovery-cmd-card">
                    <div className="recovery-cmd-head">
                      <span>{c.label}</span>
                    </div>
                    <code className="recovery-code">{c.command}</code>
                    <div style={{ display: "flex", gap: 6 }}>
                      <button className="guide-toggle" onClick={() => copyCommand(c.command)}>Copy</button>
                      <button className="guide-toggle" onClick={() => navigate("/recovery")}>Recovery page</button>
                    </div>
                  </div>
                ))}
              </div>
              {copyHint && <div style={{ marginTop: 6, color: "var(--green)", fontSize: 11 }}>{copyHint}</div>}
            </div>
          )}

          {/* Page content */}
          <BackendStatusProvider value={{
            backendUp,
            backendMessage,
            lastSuccessAt: backendPoll.lastSuccessAt,
            nextRetryInMs: backendPoll.nextRetryInMs,
          }}>
            <main className="content-area">
              <Outlet />
            </main>
          </BackendStatusProvider>

          {/* Status bar */}
          <StatusBar
            backendUp={backendUp}
            backendMessage={backendMessage}
            nextRetryInMs={backendPoll.nextRetryInMs}
            lastSuccessAt={backendPoll.lastSuccessAt}
            secHealth={secHealth}
            onOpenGlossary={() => setGlossaryOpen(true)}
            onOpenOps={() => setOpsOpen(true)}
          />
        </div>
      </div>

      {/* Command palette */}
      {cmdOpen && <CommandPalette onClose={() => setCmdOpen(false)} />}

      {/* Glossary drawer */}
      <GlossaryDrawer open={glossaryOpen} onClose={() => setGlossaryOpen(false)} />
      <OnboardingTour />

      {/* API error panel */}
      {apiPanelOpen && (
        <div
          onClick={() => setApiPanelOpen(false)}
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 9998 }}
        >
          <div onClick={(e) => e.stopPropagation()} className="card" style={{ width: 720, maxWidth: "94vw", maxHeight: "80vh", overflowY: "auto" }}>
            <div className="card-title">API Failures (last 10)</div>
            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              <button className="guide-toggle" onClick={() => clearApiEvents()}>Clear all</button>
              <button className="guide-toggle" onClick={() => setApiPanelOpen(false)}>Close</button>
            </div>
            {apiEvents.length === 0 ? (
              <div style={{ color: "var(--muted)", fontSize: 12 }}>No recent API failures.</div>
            ) : (
              <table>
                <thead><tr><th>time</th><th>endpoint</th><th>message</th><th>fails</th><th>retry_s</th><th>circuit</th></tr></thead>
                <tbody>
                  {apiEvents.slice(0, 10).map((e) => (
                    <tr key={e.id}>
                      <td style={{ color: "var(--muted)", fontSize: 11, fontFamily: "var(--font-mono)" }}>{new Date(e.ts).toLocaleTimeString()}</td>
                      <td style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{e.key}</td>
                      <td style={{ color: "var(--muted)", fontSize: 11 }}>{e.message}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{e.failureCount}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{Math.ceil(e.nextRetryInMs / 1000)}</td>
                      <td>{e.circuitOpen ? <span className="badge badge-red">OPEN</span> : "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      {/* Ops quick palette */}
      {opsOpen && (
        <div
          onClick={() => setOpsOpen(false)}
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 9999 }}
        >
          <div onClick={(e) => e.stopPropagation()} className="card" style={{ width: 600, maxWidth: "92vw", display: "flex", flexDirection: "column", gap: 10 }}>
            <div className="card-title">Quick Ops</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {[
                { label: "Run Triage Macro",       action: () => { setOpsOpen(false); navigate("/debug?auto=triage"); } },
                { label: "Logs: Regime",            action: () => { setOpsOpen(false); navigate("/logs?pack=Regime"); } },
                { label: "Logs: Shutdown",          action: () => { setOpsOpen(false); navigate("/logs?pack=Shutdown"); } },
                { label: "Logs: Timeout",           action: () => { setOpsOpen(false); navigate("/logs?pack=Timeout"); } },
                { label: "Logs: No Match",          action: () => { setOpsOpen(false); navigate("/logs?pack=No%20Match"); } },
                { label: "Start Incident Session",  action: () => { setOpsOpen(false); navigate("/logs?ops=start_incident"); } },
                { label: "Stop Incident Session",   action: () => { setOpsOpen(false); navigate("/logs?ops=stop_incident"); } },
                { label: "Export Bundle",           action: () => { setOpsOpen(false); navigate("/logs?ops=export_bundle"); } },
                { label: "Copy Triage Summary",     action: () => { setOpsOpen(false); navigate("/logs?ops=copy_summary"); } },
                { label: "Force Recover",           action: () => { setOpsOpen(false); navigate("/logs?ops=force_recover"); } },
              ].map((op) => (
                <button key={op.label} className="guide-toggle" onClick={op.action}>{op.label}</button>
              ))}
            </div>
            <div style={{ fontSize: 11, color: "var(--muted)" }}>Or press Ctrl+K for full search palette.</div>
          </div>
        </div>
      )}
    </NavigationProvider>
  );
}
