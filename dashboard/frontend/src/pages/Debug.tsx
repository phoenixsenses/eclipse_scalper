import React, { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api/client";
import type {
  AutoRunbookPolicy,
  ControlActionInfo,
  ControlActionResult,
  ControlHistoryItem,
  IncidentAuditEvent,
  IncidentInboxItem,
  SecurityAuditEvent,
  IncidentHint as ApiIncidentHint,
  RunbookSessionDetail,
  RunbookSessionSummary,
  SessionTimelineEvent,
} from "../api/types";
import AsyncState from "../components/AsyncState";
import PageGuide from "../components/PageGuide";
import TermTip from "../components/TermTip";
import { usePoll } from "../hooks/usePoll";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useDashboardAuth } from "../context/AuthContext";

function fmtTs(ts?: number | null): string {
  if (!ts) return "-";
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return "-";
  }
}

function isResearchFitnessIncident(inc: IncidentInboxItem): boolean {
  return inc.type === "data_research_fitness" || inc.incident_id === "data_research_fitness";
}

function incidentAccent(level?: string): string {
  const key = String(level || "").toUpperCase();
  if (key === "CRITICAL" || key === "ERROR") return "var(--red)";
  if (key === "WARNING") return "var(--yellow)";
  if (key === "INFO") return "var(--accent)";
  return "var(--border)";
}

interface RunbookStep {
  action: string;
  label: string;
  why: string;
}

const RUNBOOK_STEPS: RunbookStep[] = [
  {
    action: "validate_env",
    label: "Validate Environment",
    why: "Confirms required env/config and connectivity basics.",
  },
  {
    action: "preflight_check",
    label: "Preflight Check",
    why: "Verifies DB/data freshness and critical readiness gates.",
  },
  {
    action: "paper_trade_status",
    label: "Paper Trade Status",
    why: "Shows live runtime state and recent paper operation health.",
  },
  {
    action: "incident_bundle",
    label: "Incident Bundle",
    why: "Captures diagnostics snapshot for root-cause analysis.",
  },
];

function stepRecommendation(action: string): string {
  if (action === "validate_env") return "Fix missing env/dependency issues first, then rerun the session.";
  if (action === "preflight_check") return "Check collector/DB freshness and resolve preflight FAIL reasons.";
  if (action === "paper_trade_status") return "Review logs for runtime blockers (`reason=...`) and gate failures.";
  if (action === "incident_bundle") return "Open generated bundle path and inspect shutdown/errors timeline.";
  return "Inspect logs and rerun the guided session.";
}

interface IncidentHint {
  title: string;
  detail: string;
  file: string;
  query: string;
  level: "ALL" | "INFO" | "WARNING" | "ERROR" | "CRITICAL";
  confidence?: number;
  suggested_command?: string | null;
}

interface RunbookStatus {
  ts: string;
  ok: boolean;
  failed_action?: string;
}

interface TriageSummary {
  ok: boolean;
  failedAction?: string;
  sessionId?: string;
  durationSec?: number;
  ts: string;
}

const RUNBOOK_STATUS_KEY = "eclipse.debug.last_runbook.v1";
const TRIAGE_STATUS_KEY = "eclipse.debug.last_triage.v1";
const INCIDENT_FILTERS_KEY = "eclipse.debug.incident_filters.v1";
const STABILIZE_MACRO_KEY = "eclipse.debug.stabilize_macro.v1";
const TRIAGE_STEPS = ["validate_env", "preflight_check", "paper_trade_status"];
const INCIDENT_PLAYBOOKS: Record<string, string[]> = {
  dependency_missing: [
    ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt",
    "python -m tools.validate_env",
  ],
  exchange_timeout: [
    "python -m tools.incident_bundle",
    "python -m tools.paper_trade_status",
  ],
  data_freshness: [
    "python -m tools.preflight_check",
    "python -m tools.db_maintenance",
  ],
  regime_gate: [
    "python -m tools.paper_trade_status",
    "python -m tools.paper_trade_summary",
  ],
  signal_no_match: [
    "python -m tools.paper_trade_summary",
    "python -m tools.incident_bundle",
  ],
  shutdown_event: [
    "python -m tools.incident_bundle",
    "python -m tools.paper_trade_status",
  ],
};

type MacroPresetName = "quick" | "full" | "no-export" | "custom";
interface MacroStepsState {
  ackFiltered: boolean;
  autoRun: boolean;
  exportMd: boolean;
  refresh: boolean;
}
const MACRO_PRESETS: Record<Exclude<MacroPresetName, "custom">, MacroStepsState> = {
  quick: { ackFiltered: true, autoRun: false, exportMd: false, refresh: true },
  full: { ackFiltered: true, autoRun: true, exportMd: true, refresh: true },
  "no-export": { ackFiltered: true, autoRun: true, exportMd: false, refresh: true },
};

function detectIncidentHint(output: string): IncidentHint | null {
  const t = output.toLowerCase();
  if (t.includes("modulenotfounderror") || t.includes("no module named")) {
    return {
      title: "Dependency missing",
      detail: "Python module/dependency missing at runtime.",
      file: "paper_trading.log",
      query: "ModuleNotFoundError",
      level: "ERROR",
      confidence: 0.95,
      suggested_command: ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt",
    };
  }
  if (t.includes("requesttimeout") || t.includes("exchange timeout")) {
    return {
      title: "Exchange timeout",
      detail: "Exchange/network request timed out during bootstrap or runtime.",
      file: "paper_trading.log",
      query: "RequestTimeout",
      level: "WARNING",
      confidence: 0.85,
      suggested_command: "python -m tools.incident_bundle",
    };
  }
  if (t.includes("db stale") || t.includes("preflight fail")) {
    return {
      title: "Data freshness issue",
      detail: "Collector lag or DB freshness gate failed.",
      file: "microstructure_collector.log",
      query: "stale",
      level: "WARNING",
      confidence: 0.9,
      suggested_command: "python -m tools.preflight_check",
    };
  }
  if (t.includes("regime_mismatch") || t.includes("regime=unknown")) {
    return {
      title: "Regime gating block",
      detail: "Regime gate blocked entries or regime state unresolved.",
      file: "paper_trading.log",
      query: "REGIME",
      level: "INFO",
      confidence: 0.75,
      suggested_command: "python -m tools.paper_trade_status",
    };
  }
  if (t.includes("no_match") || t.includes("signal not present")) {
    return {
      title: "Signal no-match",
      detail: "Market state did not satisfy pocket thresholds at evaluation time.",
      file: "paper_trading.log",
      query: "no_match_detail",
      level: "INFO",
      confidence: 0.75,
      suggested_command: "python -m tools.paper_trade_summary",
    };
  }
  if (t.includes("shutdown") || t.includes("offline")) {
    return {
      title: "Shutdown event",
      detail: "Runtime reported shutdown/offline transition.",
      file: "paper_trading.log",
      query: "SHUTDOWN",
      level: "CRITICAL",
      confidence: 0.9,
      suggested_command: "python -m tools.incident_bundle",
    };
  }
  return null;
}

export default function Debug() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [runningAction, setRunningAction] = useState<string | null>(null);
  const [result, setResult] = useState<ControlActionResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [guidedRunning, setGuidedRunning] = useState(false);
  const [guidedResults, setGuidedResults] = useState<Record<string, ControlActionResult>>({});
  const [guidedError, setGuidedError] = useState<string | null>(null);
  const [triageRunning, setTriageRunning] = useState(false);
  const [triageSummary, setTriageSummary] = useState<TriageSummary | null>(() => {
    try {
      const raw = localStorage.getItem(TRIAGE_STATUS_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw) as TriageSummary;
      if (!parsed || typeof parsed.ok !== "boolean" || typeof parsed.ts !== "string") return null;
      return parsed;
    } catch {
      return null;
    }
  });
  const [selectedSession, setSelectedSession] = useState<RunbookSessionDetail | null>(null);
  const [compareAId, setCompareAId] = useState<string>(() => searchParams.get("compareA") ?? "");
  const [compareBId, setCompareBId] = useState<string>(() => searchParams.get("compareB") ?? "");
  const [compareA, setCompareA] = useState<RunbookSessionDetail | null>(null);
  const [compareB, setCompareB] = useState<RunbookSessionDetail | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareOnlyFailed, setCompareOnlyFailed] = useState(false);
  const [compareLinkCopied, setCompareLinkCopied] = useState(false);
  const [timeline, setTimeline] = useState<SessionTimelineEvent[]>([]);
  const [sessionAlert, setSessionAlert] = useState<string | null>(null);
  const [sessionTagDraft, setSessionTagDraft] = useState("");
  const [sessionNoteDraft, setSessionNoteDraft] = useState("");
  const [sessionMetaSaving, setSessionMetaSaving] = useState(false);
  const [incidents, setIncidents] = useState<IncidentInboxItem[]>([]);
  const [incidentLoading, setIncidentLoading] = useState(false);
  const [incidentError, setIncidentError] = useState<string | null>(null);
  const [incidentStatusFilter, setIncidentStatusFilter] = useState<"all" | "active" | "resolved">("active");
  const [incidentTypeFilter, setIncidentTypeFilter] = useState<string>("all");
  const [bulkPreviewEligible, setBulkPreviewEligible] = useState<number | null>(null);
  const [incidentAudit, setIncidentAudit] = useState<IncidentAuditEvent[]>([]);
  const [securityAudit, setSecurityAudit] = useState<SecurityAuditEvent[]>([]);
  const [macroSteps, setMacroSteps] = useState<MacroStepsState>(MACRO_PRESETS.full);
  const [macroPreset, setMacroPreset] = useState<MacroPresetName>("full");
  const [macroOwner, setMacroOwner] = useState<string>("");
  const [macroUpdatedTs, setMacroUpdatedTs] = useState<number | null>(null);
  const [policy, setPolicy] = useState<AutoRunbookPolicy>({
    enabled: false,
    min_level: "WARNING",
    cooldown_sec: 900,
    last_run_ts_by_type: {},
  });
  const [policySaving, setPolicySaving] = useState(false);
  const { auth: authCtx } = useDashboardAuth();
  const [lastRunbook, setLastRunbook] = useState<RunbookStatus | null>(() => {
    try {
      const raw = localStorage.getItem(RUNBOOK_STATUS_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw) as RunbookStatus;
      if (!parsed || typeof parsed.ts !== "string" || typeof parsed.ok !== "boolean") return null;
      return parsed;
    } catch {
      return null;
    }
  });

  const fetchActions = useCallback((signal: AbortSignal) => api.debugActions(signal), []);
  const fetchHistory = useCallback((signal: AbortSignal) => api.debugHistory(40, signal), []);
  const fetchSessions = useCallback((signal: AbortSignal) => api.debugSessions(30, signal), []);

  const actionsPoll = usePoll<ControlActionInfo[]>({
    fetcher: fetchActions,
    pollKey: "api:/debug/actions",
    intervalMs: 60_000,
    staleAfterMs: 180_000,
  });

  const historyPoll = usePoll<ControlHistoryItem[]>({
    fetcher: fetchHistory,
    pollKey: "api:/debug/history",
    intervalMs: 10_000,
    staleAfterMs: 30_000,
  });
  const sessionsPoll = usePoll<RunbookSessionSummary[]>({
    fetcher: fetchSessions,
    pollKey: "api:/debug/sessions",
    intervalMs: 15_000,
    staleAfterMs: 45_000,
  });

  const actions = actionsPoll.data ?? [];
  const history = useMemo(() => [...(historyPoll.data ?? [])].reverse(), [historyPoll.data]);
  const sessions = useMemo(() => [...(sessionsPoll.data ?? [])], [sessionsPoll.data]);
  const availableActions = useMemo(() => new Set(actions.map((a) => a.action)), [actions]);
  const selectedSessionId = selectedSession?.session_id ?? "";
  const writeLocked = authCtx.role === "viewer";

  const detectPreset = useCallback((steps: MacroStepsState): MacroPresetName => {
    const names: Array<Exclude<MacroPresetName, "custom">> = ["quick", "full", "no-export"];
    for (const n of names) {
      const p = MACRO_PRESETS[n];
      if (
        p.ackFiltered === steps.ackFiltered &&
        p.autoRun === steps.autoRun &&
        p.exportMd === steps.exportMd &&
        p.refresh === steps.refresh
      ) {
        return n;
      }
    }
    return "custom";
  }, []);

  const applyMacroPreset = useCallback((name: Exclude<MacroPresetName, "custom">) => {
    setMacroPreset(name);
    setMacroSteps(MACRO_PRESETS[name]);
  }, []);

  const updateMacroStep = useCallback(
    (patch: Partial<MacroStepsState>) => {
      setMacroSteps((prev) => ({ ...prev, ...patch }));
    },
    []
  );

  useEffect(() => {
    const d = detectPreset(macroSteps);
    if (d !== macroPreset) {
      setMacroPreset(d);
    }
  }, [macroSteps, macroPreset, detectPreset]);

  useEffect(() => {
    setSessionTagDraft(selectedSession?.tag ?? "");
    setSessionNoteDraft(selectedSession?.note ?? "");
  }, [selectedSessionId, selectedSession?.tag, selectedSession?.note]);

  useEffect(() => {
    let active = true;
    if (!selectedSessionId) {
      setTimeline([]);
      return () => {
        active = false;
      };
    }
    api.debugSessionTimeline(selectedSessionId)
      .then((rows) => {
        if (!active) return;
        setTimeline(rows);
      })
      .catch(() => {
        if (!active) return;
        setTimeline([]);
      });
    return () => {
      active = false;
    };
  }, [selectedSessionId]);

  useEffect(() => {
    let active = true;
    async function fetchIncidentData() {
      setIncidentLoading(true);
      setIncidentError(null);
      try {
        const [rows, pol, auditRows, secRows] = await Promise.all([
          api.debugIncidents(50),
          api.debugIncidentPolicy(),
          api.debugIncidentAudit(20),
          api.debugSecurityAudit(20),
        ]);
        if (!active) return;
        setIncidents(rows);
        setPolicy(pol);
        setIncidentAudit(auditRows);
        setSecurityAudit(secRows);
      } catch (err) {
        if (!active) return;
        setIncidentError(String(err));
      } finally {
        if (active) setIncidentLoading(false);
      }
    }
    fetchIncidentData();
    const timer = window.setInterval(fetchIncidentData, 15_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let active = true;
    api.previewBulkDebugIncidents({
      incident_type: incidentTypeFilter === "all" ? undefined : incidentTypeFilter,
      status_scope: incidentStatusFilter === "all" ? "all" : incidentStatusFilter,
    })
      .then((res) => {
        if (!active) return;
        setBulkPreviewEligible(res.eligible ?? 0);
      })
      .catch(() => {
        if (!active) return;
        setBulkPreviewEligible(null);
      });
    return () => {
      active = false;
    };
  }, [incidentStatusFilter, incidentTypeFilter, incidents.length]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(INCIDENT_FILTERS_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as { status?: string; type?: string };
      if (parsed.status === "all" || parsed.status === "active" || parsed.status === "resolved") {
        setIncidentStatusFilter(parsed.status);
      }
      if (typeof parsed.type === "string" && parsed.type) {
        setIncidentTypeFilter(parsed.type);
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(INCIDENT_FILTERS_KEY, JSON.stringify({ status: incidentStatusFilter, type: incidentTypeFilter }));
    } catch {
      // ignore
    }
  }, [incidentStatusFilter, incidentTypeFilter]);


  useEffect(() => {
    try {
      const raw = localStorage.getItem(STABILIZE_MACRO_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as Partial<MacroStepsState> & { preset?: MacroPresetName };
      if (parsed.preset && parsed.preset !== "custom" && MACRO_PRESETS[parsed.preset]) {
        setMacroPreset(parsed.preset);
        setMacroSteps(MACRO_PRESETS[parsed.preset]);
        return;
      }
      const next: MacroStepsState = {
        ackFiltered: typeof parsed.ackFiltered === "boolean" ? parsed.ackFiltered : MACRO_PRESETS.full.ackFiltered,
        autoRun: typeof parsed.autoRun === "boolean" ? parsed.autoRun : MACRO_PRESETS.full.autoRun,
        exportMd: typeof parsed.exportMd === "boolean" ? parsed.exportMd : MACRO_PRESETS.full.exportMd,
        refresh: typeof parsed.refresh === "boolean" ? parsed.refresh : MACRO_PRESETS.full.refresh,
      };
      setMacroSteps(next);
      setMacroPreset(detectPreset(next));
    } catch {
      // ignore
    }
  }, [detectPreset]);

  useEffect(() => {
    let active = true;
    api.debugMacroPreset()
      .then((cfg) => {
        if (!active) return;
        const next: MacroStepsState = {
          ackFiltered: !!cfg.ackFiltered,
          autoRun: !!cfg.autoRun,
          exportMd: !!cfg.exportMd,
          refresh: !!cfg.refresh,
        };
        setMacroSteps(next);
        setMacroPreset(detectPreset(next));
        setMacroOwner(cfg.owner ?? "");
        setMacroUpdatedTs(cfg.updated_ts ?? null);
      })
      .catch(() => {
        // fallback stays local
      });
    return () => {
      active = false;
    };
  }, [detectPreset]);

  useEffect(() => {
    try {
      localStorage.setItem(STABILIZE_MACRO_KEY, JSON.stringify({ preset: macroPreset, ...macroSteps }));
    } catch {
      // ignore
    }
    api
      .patchDebugMacroPreset({
        preset: macroPreset,
        ackFiltered: macroSteps.ackFiltered,
        autoRun: macroSteps.autoRun,
        exportMd: macroSteps.exportMd,
        refresh: macroSteps.refresh,
        owner: macroOwner || undefined,
      })
      .then((cfg) => {
        setMacroOwner(cfg.owner ?? "");
        setMacroUpdatedTs(cfg.updated_ts ?? null);
      })
      .catch(() => {
        // keep local fallback
      });
  }, [macroSteps, macroPreset, macroOwner]);

  useEffect(() => {
    const qA = searchParams.get("compareA") ?? "";
    const qB = searchParams.get("compareB") ?? "";
    if (qA !== compareAId) setCompareAId(qA);
    if (qB !== compareBId) setCompareBId(qB);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  useEffect(() => {
    const next = new URLSearchParams(searchParams);
    if (compareAId) next.set("compareA", compareAId);
    else next.delete("compareA");
    if (compareBId) next.set("compareB", compareBId);
    else next.delete("compareB");
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [compareAId, compareBId, searchParams, setSearchParams]);

  async function onRun(action: string) {
    setRunningAction(action);
    setRunError(null);
    try {
      const res = await api.runDebugAction(action);
      setResult(res);
      historyPoll.refresh();
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        setRunError("Control endpoints disabled. Set DASHBOARD_CONTROL_ENABLED=1.");
      } else {
        setRunError(String(err));
      }
      setResult(null);
    } finally {
      setRunningAction(null);
    }
  }

  async function runPreflightFromIncident() {
    await onRun("preflight_check");
  }

  async function runGuidedSession() {
    setGuidedRunning(true);
    setGuidedError(null);
    setGuidedResults({});
    for (const step of RUNBOOK_STEPS) {
      if (!availableActions.has(step.action)) {
        setGuidedError(`Missing action from backend: ${step.action}`);
        const status: RunbookStatus = { ts: new Date().toISOString(), ok: false, failed_action: step.action };
        setLastRunbook(status);
        try { localStorage.setItem(RUNBOOK_STATUS_KEY, JSON.stringify(status)); } catch {}
        setRunningAction(null);
        setGuidedRunning(false);
        return;
      }
    }
    try {
      setRunningAction("runbook");
      const session = await api.runDebugRunbook(RUNBOOK_STEPS.map((s) => s.action));
      setSelectedSession(session);
      const stepMap: Record<string, ControlActionResult> = {};
      for (const s of session.steps ?? []) {
        stepMap[s.action] = s;
      }
      setGuidedResults(stepMap);
      setResult(session.steps?.length ? session.steps[session.steps.length - 1] : null);
      if (!session.ok && session.failed_action) {
        setGuidedError(`Stopped at ${session.failed_action}. ${stepRecommendation(session.failed_action)}`);
      }
      const status: RunbookStatus = { ts: new Date().toISOString(), ok: session.ok, failed_action: session.failed_action ?? undefined };
      setLastRunbook(status);
      try { localStorage.setItem(RUNBOOK_STATUS_KEY, JSON.stringify(status)); } catch {}
    } catch (err) {
      setGuidedError(`Runbook failed: ${err instanceof Error ? err.message : String(err)}`);
      const status: RunbookStatus = { ts: new Date().toISOString(), ok: false };
      setLastRunbook(status);
      try { localStorage.setItem(RUNBOOK_STATUS_KEY, JSON.stringify(status)); } catch {}
    } finally {
      historyPoll.refresh();
      sessionsPoll.refresh();
      setRunningAction(null);
      setGuidedRunning(false);
    }
  }

  async function runTriageMacro() {
    setTriageRunning(true);
    setGuidedError(null);
    for (const action of TRIAGE_STEPS) {
      if (!availableActions.has(action)) {
        const miss: TriageSummary = {
          ok: false,
          failedAction: action,
          ts: new Date().toISOString(),
        };
        setTriageSummary(miss);
        try { localStorage.setItem(TRIAGE_STATUS_KEY, JSON.stringify(miss)); } catch {}
        setTriageRunning(false);
        return;
      }
    }
    try {
      setRunningAction("triage_macro");
      const session = await api.runDebugRunbook(TRIAGE_STEPS);
      setSelectedSession(session);
      const next: TriageSummary = {
        ok: session.ok,
        failedAction: session.failed_action ?? undefined,
        sessionId: session.session_id,
        durationSec: session.duration_sec,
        ts: new Date().toISOString(),
      };
      setTriageSummary(next);
      try { localStorage.setItem(TRIAGE_STATUS_KEY, JSON.stringify(next)); } catch {}
      if (!session.ok && session.failed_action) {
        setGuidedError(`Triage macro failed at ${session.failed_action}. ${stepRecommendation(session.failed_action)}`);
      }
    } catch (err) {
      const failed: TriageSummary = {
        ok: false,
        ts: new Date().toISOString(),
      };
      setTriageSummary(failed);
      try { localStorage.setItem(TRIAGE_STATUS_KEY, JSON.stringify(failed)); } catch {}
      setGuidedError(`Triage macro failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      sessionsPoll.refresh();
      historyPoll.refresh();
      setRunningAction(null);
      setTriageRunning(false);
    }
  }

  useEffect(() => {
    const auto = searchParams.get("auto");
    if (auto !== "triage") return;
    if (!triageRunning && !runningAction) {
      void runTriageMacro();
    }
    const next = new URLSearchParams(searchParams);
    next.delete("auto");
    setSearchParams(next, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  function openLogsForFailedStep(step?: string) {
    const s = (step || "").toLowerCase();
    if (s.includes("preflight")) return navigate("/logs?pack=Timeout");
    if (s.includes("status")) return navigate("/logs?pack=No%20Match");
    if (s.includes("validate")) return navigate("/logs?pack=Shutdown");
    return navigate("/logs");
  }

  const firstFailedStep = useMemo(() => {
    for (const step of RUNBOOK_STEPS) {
      const r = guidedResults[step.action];
      if (r && !r.ok) return step;
    }
    return null;
  }, [guidedResults]);
  function mapApiIncidentHint(h: ApiIncidentHint | null | undefined): IncidentHint | null {
    if (!h || !h.title || !h.detail || !h.file || !h.query || !h.level) return null;
    const lvl = h.level;
    if (lvl !== "ALL" && lvl !== "INFO" && lvl !== "WARNING" && lvl !== "ERROR" && lvl !== "CRITICAL") {
      return null;
    }
    return {
      title: h.title,
      detail: h.detail,
      file: h.file,
      query: h.query,
      level: lvl,
      confidence: h.confidence ?? undefined,
      suggested_command: h.suggested_command ?? undefined,
    };
  }

  const incidentHint = useMemo(() => {
    const fromSession = mapApiIncidentHint(selectedSession?.incident_hint);
    if (fromSession) return fromSession;
    const source = guidedError || result?.output || runError || "";
    return source ? detectIncidentHint(source) : null;
  }, [guidedError, result?.output, runError, selectedSession]);

  const incidentCommand = useMemo(() => {
    if (!incidentHint) return null;
    if (incidentHint.suggested_command) return incidentHint.suggested_command;
    if (incidentHint.query === "ModuleNotFoundError") return ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt";
    if (incidentHint.query === "RequestTimeout") return "python -m tools.incident_bundle";
    if (incidentHint.query === "stale") return "python -m tools.preflight_check";
    if (incidentHint.query === "REGIME") return "python -m tools.paper_trade_status";
    if (incidentHint.query === "no_match_detail") return "python -m tools.paper_trade_summary";
    if (incidentHint.query === "SHUTDOWN") return "python -m tools.incident_bundle";
    return null;
  }, [incidentHint]);
  const incidentPlaybook = useMemo(() => {
    const key = selectedSession?.incident_hint?.type ?? incidentHint?.title?.toLowerCase().replace(/\s+/g, "_");
    if (!key) return [];
    return INCIDENT_PLAYBOOKS[key] ?? [];
  }, [selectedSession?.incident_hint?.type, incidentHint?.title]);

  async function runFromIncident() {
    if (!incidentHint) return;
    setGuidedRunning(true);
    setGuidedError(null);
    try {
      const session = await api.runDebugRunbookFromIncident(
        incidentHint.file,
        incidentHint.query,
        incidentHint.level,
        RUNBOOK_STEPS.map((s) => s.action)
      );
      await loadSession(session.session_id);
      sessionsPoll.refresh();
      historyPoll.refresh();
    } catch (err) {
      setGuidedError(`Incident runbook failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setGuidedRunning(false);
    }
  }

  async function loadSession(sessionId: string) {
    try {
      const detail = await api.debugSessionDetail(sessionId);
      setSelectedSession(detail);
      const stepMap: Record<string, ControlActionResult> = {};
      for (const s of detail.steps ?? []) {
        stepMap[s.action] = s;
      }
      setGuidedResults(stepMap);
      setGuidedError(detail.ok ? null : `Stopped at ${detail.failed_action ?? "unknown"}.`);
      setResult(detail.steps?.length ? detail.steps[detail.steps.length - 1] : null);
    } catch (err) {
      setRunError(`Failed to load session ${sessionId}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function runCompare() {
    if (!compareAId || !compareBId) {
      setCompareError("Select two sessions to compare.");
      return;
    }
    setCompareError(null);
    setCompareLoading(true);
    try {
      const [a, b] = await Promise.all([
        api.debugSessionDetail(compareAId),
        api.debugSessionDetail(compareBId),
      ]);
      setCompareA(a);
      setCompareB(b);
    } catch (err) {
      setCompareError(`Compare failed: ${err instanceof Error ? err.message : String(err)}`);
      setCompareA(null);
      setCompareB(null);
    } finally {
      setCompareLoading(false);
    }
  }

  useEffect(() => {
    if (!selectedSessionId) return;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const latest = await api.debugSessionDetail(selectedSessionId);
        if (cancelled) return;
        const prev = selectedSession;
        setSelectedSession(latest);
        if (prev) {
          const prevFail = prev.steps?.filter((s) => !s.ok).length ?? 0;
          const nextFail = latest.steps?.filter((s) => !s.ok).length ?? 0;
          const prevType = prev.incident_hint?.type ?? "";
          const nextType = latest.incident_hint?.type ?? "";
          const prevSnips = prev.log_snippets?.length ?? 0;
          const nextSnips = latest.log_snippets?.length ?? 0;
          if (nextFail > prevFail) {
            setSessionAlert(`Session updated: failed steps ${prevFail} -> ${nextFail}`);
          } else if (prevType !== nextType && nextType) {
            setSessionAlert(`Incident changed: ${prevType || "-"} -> ${nextType}`);
          } else if (nextSnips > prevSnips) {
            setSessionAlert(`Session updated: log snippets ${prevSnips} -> ${nextSnips}`);
          }
        }
      } catch {
        // keep polling; errors are transient
      }
    }, 15_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [selectedSessionId, selectedSession]);

  async function saveSessionMeta() {
    if (!selectedSessionId) return;
    setSessionMetaSaving(true);
    try {
      const updated = await api.patchDebugSession(selectedSessionId, {
        tag: sessionTagDraft,
        note: sessionNoteDraft,
      });
      setSelectedSession(updated);
      sessionsPoll.refresh();
    } catch (err) {
      setRunError(`Failed to save session meta: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSessionMetaSaving(false);
    }
  }

  async function runIncident(incidentId: string) {
    try {
      const session = await api.runDebugIncidentRunbook(incidentId);
      await loadSession(session.session_id);
      sessionsPoll.refresh();
      historyPoll.refresh();
      setSessionAlert(`Incident runbook executed: ${incidentId}`);
    } catch (err) {
      setRunError(`Incident runbook failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function patchIncident(incidentId: string, action: string, incidentType?: string) {
    if (action === "mute_type") {
      const ok = window.confirm(`Mute incident type "${incidentType ?? "-"}"?`);
      if (!ok) return;
    }
    try {
      await api.patchDebugIncident(incidentId, {
        action,
        incident_type: incidentType,
        snooze_minutes: 60,
      });
      const rows = await api.debugIncidents(50);
      setIncidents(rows);
    } catch (err) {
      setIncidentError(`Incident update failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function savePolicy() {
    setPolicySaving(true);
    try {
      const next = await api.patchDebugIncidentPolicy(policy);
      setPolicy(next);
    } catch (err) {
      setIncidentError(`Policy save failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPolicySaving(false);
    }
  }

  async function runAutoPolicyOnce() {
    try {
      const res = await api.runAutoRunbookOnce();
      if (res.ran && res.session_id) {
        await loadSession(res.session_id);
        sessionsPoll.refresh();
      }
      const rows = await api.debugIncidents(50);
      setIncidents(rows);
      setSessionAlert(`Auto-run result: ${res.reason ?? (res.ran ? "executed" : "skipped")}`);
    } catch (err) {
      setIncidentError(`Auto-run failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function refreshIncidents() {
    try {
      const [rows, auditRows, secRows] = await Promise.all([
        api.debugIncidents(50),
        api.debugIncidentAudit(20),
        api.debugSecurityAudit(20),
      ]);
      setIncidents(rows);
      setIncidentAudit(auditRows);
      setSecurityAudit(secRows);
    } catch (err) {
      setIncidentError(`Incident refresh failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function bulkIncident(action: "ack" | "resolve") {
    if (action === "resolve") {
      const token = window.prompt('Type "RESOLVE" to confirm bulk resolve');
      if (token !== "RESOLVE") return;
    }
    try {
      await api.bulkDebugIncidents({
        action,
        incident_type: incidentTypeFilter === "all" ? undefined : incidentTypeFilter,
        status_scope: incidentStatusFilter === "all" ? "all" : incidentStatusFilter,
      });
      await refreshIncidents();
    } catch (err) {
      setIncidentError(`Bulk incident action failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function undoIncidentAction() {
    try {
      const res = await api.undoDebugIncidents();
      await refreshIncidents();
      setSessionAlert(`Undo: ${String((res as { reason?: unknown }).reason ?? "done")}`);
    } catch (err) {
      setIncidentError(`Undo failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function runStabilizeMacro() {
    const token = window.prompt('Type "STABILIZE" to run macro');
    if (token !== "STABILIZE") return;
    try {
      if (macroSteps.ackFiltered) {
        await api.bulkDebugIncidents({
          action: "ack",
          incident_type: incidentTypeFilter === "all" ? undefined : incidentTypeFilter,
          status_scope: incidentStatusFilter === "all" ? "all" : incidentStatusFilter,
        });
      }
      if (macroSteps.autoRun) {
        await runAutoPolicyOnce();
      }
      if (macroSteps.exportMd) {
        exportSessionMarkdown();
      }
      if (macroSteps.refresh) {
        await refreshIncidents();
      }
      setSessionAlert(
        `Stabilize macro completed (${[
          macroSteps.ackFiltered ? "ack" : null,
          macroSteps.autoRun ? "auto-run" : null,
          macroSteps.exportMd ? "export-md" : null,
          macroSteps.refresh ? "refresh" : null,
        ]
          .filter(Boolean)
          .join(", ") || "no-op"}).`
      );
    } catch (err) {
      setIncidentError(`Stabilize macro failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  function exportIncidentOpsBundle() {
    const payload = {
      exported_at: new Date().toISOString(),
      filters: { incidentStatusFilter, incidentTypeFilter },
      incidents: filteredIncidents,
      incident_audit: incidentAudit,
      selected_session: selectedSession,
      compare: {
        compareAId,
        compareBId,
        summary: compareSummary,
      },
      timeline,
    };
    downloadTextFile(
      `incident_ops_bundle_${new Date().toISOString().replace(/[:.]/g, "-")}.json`,
      JSON.stringify(payload, null, 2)
    );
  }

  const incidentTypeOptions = useMemo(() => {
    const types = Array.from(new Set(incidents.map((x) => x.type).filter(Boolean))).sort();
    return ["all", ...types];
  }, [incidents]);

  const filteredIncidents = useMemo(() => {
    return incidents.filter((inc) => {
      if (incidentStatusFilter === "active" && inc.status === "resolved") return false;
      if (incidentStatusFilter === "resolved" && inc.status !== "resolved") return false;
      if (incidentTypeFilter !== "all" && inc.type !== incidentTypeFilter) return false;
      return true;
    });
  }, [incidents, incidentStatusFilter, incidentTypeFilter]);

  const incidentSla = useMemo(() => {
    const nowSec = Date.now() / 1000;
    const active = incidents.filter((x) => x.status !== "resolved");
    const newOnes = incidents.filter((x) => x.status === "new");
    const acked = incidents.filter((x) => x.ack_ts && x.ts);
    const resolved = incidents.filter((x) => x.resolved_ts && x.ts);
    const avg = (vals: number[]) => (vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null);
    const avgNewAge = avg(newOnes.map((x) => Math.max(0, nowSec - (x.ts || nowSec))));
    const avgAckLag = avg(acked.map((x) => Math.max(0, (x.ack_ts || 0) - (x.ts || 0))));
    const avgResolveLag = avg(resolved.map((x) => Math.max(0, (x.resolved_ts || 0) - (x.ts || 0))));
    return {
      total: incidents.length,
      active: active.length,
      avgNewAge,
      avgAckLag,
      avgResolveLag,
    };
  }, [incidents]);
  const securityHealth = useMemo(() => {
    const cutoff = Date.now() / 1000 - 15 * 60;
    const recent = securityAudit.filter((x) => (x.ts ?? 0) >= cutoff);
    const byKind = (k: string) => recent.filter((x) => x.kind === k).length;
    return {
      recentCount: recent.length,
      authFailed: byKind("auth_failed"),
      roleDenied: byKind("role_denied"),
      rateLimited: byKind("rate_limited"),
      replay: byKind("idempotency_replay"),
    };
  }, [securityAudit]);

  const compareSummary = useMemo(() => {
    if (!compareA || !compareB) return null;
    const stepStatusMap = (s: RunbookSessionDetail) =>
      Object.fromEntries((s.steps ?? []).map((x) => [x.action, x.ok ? "pass" : "fail"]));
    const sa = stepStatusMap(compareA);
    const sb = stepStatusMap(compareB);
    const allActions = Array.from(new Set([...Object.keys(sa), ...Object.keys(sb)]));
    const stepRows = allActions.map((action) => ({
      action,
      a: sa[action] ?? "-",
      b: sb[action] ?? "-",
      changed: (sa[action] ?? "-") !== (sb[action] ?? "-"),
    }));
    const diffs = stepRows.filter((x) => x.changed);
    return {
      aOk: compareA.ok,
      bOk: compareB.ok,
      aFailed: compareA.failed_action ?? "-",
      bFailed: compareB.failed_action ?? "-",
      aIncident: compareA.incident_hint?.type ?? "-",
      bIncident: compareB.incident_hint?.type ?? "-",
      aSnips: compareA.log_snippets?.length ?? 0,
      bSnips: compareB.log_snippets?.length ?? 0,
      stepRows,
      stepDiffs: diffs,
    };
  }, [compareA, compareB]);

  async function copyCommand(cmd: string) {
    try {
      await navigator.clipboard.writeText(cmd);
    } catch {
      // ignore clipboard failures
    }
  }

  function downloadTextFile(filename: string, content: string) {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  function exportSessionJson() {
    const payload = {
      exported_at: new Date().toISOString(),
      guided_error: guidedError,
      run_error: runError,
      incident_hint: incidentHint,
      guided_results: guidedResults,
      last_result: result,
      recent_history: history.slice(0, 40),
    };
    downloadTextFile(
      `debug_session_${new Date().toISOString().replace(/[:.]/g, "-")}.json`,
      JSON.stringify(payload, null, 2)
    );
  }

  function exportSessionMarkdown() {
    const ts = new Date().toISOString();
    const lines: string[] = [];
    lines.push("# Debug Session Export");
    lines.push("");
    lines.push(`- exported_at: ${ts}`);
    lines.push(`- guided_error: ${guidedError ?? "-"}`);
    lines.push(`- run_error: ${runError ?? "-"}`);
    if (incidentHint) {
      lines.push(`- incident_title: ${incidentHint.title}`);
      lines.push(`- incident_detail: ${incidentHint.detail}`);
      lines.push(`- incident_log_file: ${incidentHint.file}`);
      lines.push(`- incident_query: ${incidentHint.query}`);
      lines.push("");
    }
    lines.push("## Guided Steps");
    for (const step of RUNBOOK_STEPS) {
      const r = guidedResults[step.action];
      lines.push(
        `- ${step.label} (${step.action}): ${
          !r ? "pending" : r.ok ? "pass" : `fail (exit=${r.exit_code})`
        }`
      );
    }
    lines.push("");
    if (result) {
      lines.push("## Last Action");
      lines.push(`- action: ${result.action}`);
      lines.push(`- ok: ${result.ok}`);
      lines.push(`- exit_code: ${result.exit_code}`);
      lines.push(`- duration_sec: ${result.duration_sec.toFixed(2)}`);
      lines.push("");
      lines.push("```text");
      lines.push(result.output || "(no output)");
      lines.push("```");
      lines.push("");
    }
    lines.push("## Recent History");
    for (const h of history.slice(0, 20)) {
      lines.push(
        `- ${fmtTs(h.ts)} | ${h.action} | ${
          h.ok == null ? "n/a" : h.ok ? "ok" : "fail"
        } | exit=${h.exit_code ?? "-"} | duration=${h.duration_sec ?? "-"}`
      );
    }
    lines.push("");
    downloadTextFile(
      `debug_session_${new Date().toISOString().replace(/[:.]/g, "-")}.md`,
      lines.join("\n")
    );
  }

  function exportCompareJson() {
    if (!compareA || !compareB || !compareSummary) return;
    const payload = {
      exported_at: new Date().toISOString(),
      compare_a_id: compareAId,
      compare_b_id: compareBId,
      compare_a: compareA,
      compare_b: compareB,
      summary: compareSummary,
    };
    downloadTextFile(
      `debug_compare_${compareAId || "A"}_vs_${compareBId || "B"}_${new Date().toISOString().replace(/[:.]/g, "-")}.json`,
      JSON.stringify(payload, null, 2)
    );
  }

  function exportCompareMarkdown() {
    if (!compareA || !compareB || !compareSummary) return;
    const lines: string[] = [];
    lines.push("# Debug Session Compare");
    lines.push("");
    lines.push(`- exported_at: ${new Date().toISOString()}`);
    lines.push(`- A: ${compareAId}`);
    lines.push(`- B: ${compareBId}`);
    lines.push(`- A ok: ${String(compareSummary.aOk)} (failed=${compareSummary.aFailed}, incident=${compareSummary.aIncident})`);
    lines.push(`- B ok: ${String(compareSummary.bOk)} (failed=${compareSummary.bFailed}, incident=${compareSummary.bIncident})`);
    lines.push(`- Step differences: ${compareSummary.stepDiffs.length}`);
    lines.push("");
    lines.push("## Step Status");
    for (const row of compareSummary.stepRows) {
      lines.push(`- ${row.action}: A=${row.a}, B=${row.b}${row.changed ? " [changed]" : ""}`);
    }
    downloadTextFile(
      `debug_compare_${compareAId || "A"}_vs_${compareBId || "B"}_${new Date().toISOString().replace(/[:.]/g, "-")}.md`,
      lines.join("\n")
    );
  }

  async function copyCompareLink() {
    const params = new URLSearchParams();
    if (compareAId) params.set("compareA", compareAId);
    if (compareBId) params.set("compareB", compareBId);
    const shareUrl = `${window.location.origin}${window.location.pathname}${params.toString() ? `?${params.toString()}` : ""}`;
    await copyCommand(shareUrl);
    setCompareLinkCopied(true);
    window.setTimeout(() => setCompareLinkCopied(false), 1400);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <PageGuide
        icon="🛠️"
        titleTr="Hata Ayıklama Merkezi"
        titleEn="Debug Control Center"
        subtitleTr="Tanı komutlarını güvenli şekilde çalıştır, incident akışını takip et, aksiyon geçmişini izle."
        subtitleEn="Run safe diagnostics, process incidents, and track operational actions."
        items={[
          {
            icon: "🧪",
            titleTr: "Guided Session",
            titleEn: "Guided Session",
            descTr: "Sıralı kontrol: env → preflight → status → incident bundle.",
            descEn: "Step chain: env → preflight → status → incident bundle.",
          },
          {
            icon: "🚨",
            titleTr: "Incident Inbox",
            titleEn: "Incident Inbox",
            descTr: "Olayları filtrele, bulk aksiyon uygula, geri al.",
            descEn: "Filter incidents, run bulk actions, and undo safely.",
          },
          {
            icon: "🔐",
            titleTr: "Yetki",
            titleEn: "Authorization",
            descTr: "Viewer rolü write aksiyonları kilitler.",
            descEn: "Viewer role locks write actions by design.",
          },
        ]}
      />

      <div className="card">
        <div className="card-title">Guided Debug Session</div>
        <div style={{ color: "var(--muted)", marginBottom: 10 }}>
          Recommended sequence: validate, then preflight, then status, then incident capture.
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
          <button
            onClick={runTriageMacro}
            disabled={writeLocked || triageRunning || Boolean(runningAction) || guidedRunning}
            style={{
              padding: "6px 10px",
              borderRadius: 4,
              border: "1px solid var(--yellow)",
              background: triageRunning ? "var(--surface-3)" : "transparent",
              color: "var(--yellow)",
              cursor: triageRunning ? "not-allowed" : "pointer",
            }}
          >
            {triageRunning ? "Running triage..." : "Run Triage Macro"}
          </button>
          <button
            onClick={runGuidedSession}
            disabled={writeLocked || guidedRunning || Boolean(runningAction) || actionsPoll.isLoading}
            style={{
              padding: "6px 10px",
              borderRadius: 4,
              border: "1px solid var(--accent)",
              background: guidedRunning ? "var(--surface-3)" : "transparent",
              color: "var(--accent)",
              cursor: guidedRunning ? "not-allowed" : "pointer",
            }}
          >
            {guidedRunning ? "Running guided session..." : "Run Guided Session"}
          </button>
          {firstFailedStep && (
            <span style={{ color: "var(--yellow)", fontSize: 12 }}>
              Next action: {stepRecommendation(firstFailedStep.action)}
            </span>
          )}
          {lastRunbook && (
            <span style={{ color: lastRunbook.ok ? "var(--green)" : "var(--yellow)", fontSize: 12, marginLeft: "auto" }}>
              last runbook: {lastRunbook.ok ? "pass" : `fail (${lastRunbook.failed_action ?? "unknown"})`} @ {lastRunbook.ts}
            </span>
          )}
        </div>
        {triageSummary && (
          <div
            className="card"
            style={{
              marginBottom: 10,
              borderColor: triageSummary.ok ? "var(--green)" : "var(--yellow)",
            }}
          >
            <div className="card-title">Triage Macro Summary</div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontSize: 12 }}>
              <span style={{ color: triageSummary.ok ? "var(--green)" : "var(--yellow)", fontWeight: 700 }}>
                {triageSummary.ok ? "PASS" : "FAIL"}
              </span>
              <span style={{ color: "var(--muted)" }}>ts={triageSummary.ts}</span>
              {triageSummary.sessionId && <span>session={triageSummary.sessionId}</span>}
              {triageSummary.durationSec != null && <span>duration={triageSummary.durationSec.toFixed(2)}s</span>}
              {triageSummary.failedAction && <span>failed={triageSummary.failedAction}</span>}
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
              {!triageSummary.ok ? (
                <>
                  <button
                    onClick={() => openLogsForFailedStep(triageSummary.failedAction)}
                    style={{
                      padding: "3px 8px",
                      borderRadius: 4,
                      border: "1px solid var(--border)",
                      background: "transparent",
                      color: "var(--muted)",
                      cursor: "pointer",
                      fontSize: 11,
                    }}
                  >
                    Open failed step logs
                  </button>
                  <button
                    onClick={() => navigate("/debug")}
                    style={{
                      padding: "3px 8px",
                      borderRadius: 4,
                      border: "1px solid var(--accent)",
                      background: "transparent",
                      color: "var(--accent)",
                      cursor: "pointer",
                      fontSize: 11,
                    }}
                  >
                    Open Incident Inbox
                  </button>
                </>
              ) : (
                <button
                  onClick={() => navigate("/logs?pack=No%20Match")}
                  style={{
                    padding: "3px 8px",
                    borderRadius: 4,
                    border: "1px solid var(--border)",
                    background: "transparent",
                    color: "var(--muted)",
                    cursor: "pointer",
                    fontSize: 11,
                  }}
                >
                  Go to Logs (No Match)
                </button>
              )}
            </div>
          </div>
        )}
        {guidedError && (
          <div style={{ color: "var(--red)", fontSize: 12, marginBottom: 10 }}>
            {guidedError}
          </div>
        )}
        <table>
          <thead>
            <tr>
              <th><TermTip term="Step" tr="Runbook adimi." en="Runbook step." /></th>
              <th><TermTip term="Action" tr="Calistirilan komut kimligi." en="Executed action id." /></th>
              <th><TermTip term="Status" tr="Adim sonucu (pass/fail)." en="Step outcome (pass/fail)." /></th>
              <th><TermTip term="Why" tr="Bu adimin amaci." en="Purpose of this step." /></th>
            </tr>
          </thead>
          <tbody>
            {RUNBOOK_STEPS.map((step) => {
              const r = guidedResults[step.action];
              let status = "pending";
              let color = "var(--muted)";
              if (guidedRunning && runningAction === step.action) {
                status = "running";
                color = "var(--accent)";
              } else if (r) {
                status = r.ok ? "pass" : "fail";
                color = r.ok ? "var(--green)" : "var(--red)";
              }
              return (
                <tr key={step.action}>
                  <td>{step.label}</td>
                  <td style={{ color: "var(--muted)" }}>{step.action}</td>
                  <td style={{ color, fontWeight: 700 }}>{status}</td>
                  <td style={{ color: "var(--muted)" }}>{step.why}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="card">
        <div className="card-title">Debug Controls</div>
        <div style={{ color: "var(--muted)", marginBottom: 10 }}>
          Safe action runner (whitelisted commands only). Useful for live diagnostics and incident triage.
        </div>
        <AsyncState
          loading={actionsPoll.isLoading}
          error={actionsPoll.error}
          isEmpty={actions.length === 0}
          emptyText="No debug actions available"
        >
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {actions.map((a) => (
              <button
                key={a.action}
                onClick={() => onRun(a.action)}
                disabled={writeLocked || Boolean(runningAction) || guidedRunning}
                title={a.description}
                style={{
                  padding: "6px 10px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: runningAction === a.action ? "var(--surface-3)" : "transparent",
                  color: "var(--text)",
                  cursor: runningAction ? "not-allowed" : "pointer",
                }}
              >
                {runningAction === a.action ? `Running ${a.action}...` : a.action}
              </button>
            ))}
          </div>
        </AsyncState>
        <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
          <button
            onClick={exportSessionJson}
            style={{
              padding: "6px 10px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: "pointer",
            }}
          >
            Export Session JSON
          </button>
          <button
            onClick={exportSessionMarkdown}
            style={{
              padding: "6px 10px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: "pointer",
            }}
          >
            Export Session MD
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-title">API Auth & Security</div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
          <span style={{ color: "var(--muted)", fontSize: 12 }}>
            operator={authCtx.operator || "-"} role={authCtx.role} apiKey={authCtx.apiKey ? "set" : "empty"} idempotency={authCtx.idempotency ? "on" : "off"}
          </span>
          <Link to="/settings" style={{ fontSize: 12 }}>
            Manage in Settings
          </Link>
        </div>
        <div style={{ marginTop: 8, color: "var(--muted)", fontSize: 12 }}>
          write mode: {writeLocked ? "LOCKED (viewer)" : `ENABLED (${authCtx.role})`}
          {" | "}security(15m): auth_fail={securityHealth.authFailed}, role_denied={securityHealth.roleDenied}, rate_limited={securityHealth.rateLimited}, replay={securityHealth.replay}
        </div>
      </div>

      {runError && (
        <div className="card" style={{ borderColor: "var(--red)", color: "var(--red)" }}>
          {runError}
        </div>
      )}
      {sessionAlert && (
        <div className="card" style={{ borderColor: "var(--yellow)", color: "var(--yellow)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span>{sessionAlert}</span>
            <button
              onClick={() => setSessionAlert(null)}
              style={{
                marginLeft: "auto",
                padding: "3px 8px",
                borderRadius: 4,
                border: "1px solid var(--border)",
                background: "transparent",
                color: "var(--muted)",
                cursor: "pointer",
                fontSize: 12,
              }}
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {incidentHint && (
        <div className="card" style={{ borderLeft: "3px solid var(--yellow)" }}>
          <div className="card-title" style={{ marginBottom: 6 }}>Incident Summary</div>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>{incidentHint.title}</div>
          <div style={{ color: "var(--muted)", fontSize: 12, marginBottom: 8 }}>
            {incidentHint.detail}
          </div>
          {incidentHint.confidence != null && (
            <div style={{ color: "var(--muted)", fontSize: 12, marginBottom: 8 }}>
              confidence: {(incidentHint.confidence * 100).toFixed(0)}%
            </div>
          )}
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <Link
              to={`/logs?file=${encodeURIComponent(incidentHint.file)}&q=${encodeURIComponent(incidentHint.query)}&level=${incidentHint.level}`}
              style={{ fontSize: 12 }}
            >
              Open related logs
            </Link>
            <Link
              to={`/trades?tab=signals&q=${encodeURIComponent(incidentHint.query)}`}
              style={{ fontSize: 12 }}
            >
              Open related trades
            </Link>
            {incidentCommand && (
              <button
                onClick={() => copyCommand(incidentCommand)}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: "var(--muted)",
                  cursor: "pointer",
                  fontSize: 12,
                }}
                title={incidentCommand}
              >
                Copy suggested command
              </button>
            )}
            {selectedSession?.failed_action && (
              <button
                onClick={() => onRun(selectedSession.failed_action as string)}
                disabled={writeLocked}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: "var(--muted)",
                  cursor: "pointer",
                  fontSize: 12,
                }}
              >
                Re-run failed step
              </button>
            )}
            <button
              onClick={runFromIncident}
              disabled={writeLocked || guidedRunning}
              style={{
                padding: "4px 8px",
                borderRadius: 4,
                border: "1px solid var(--border)",
                background: "transparent",
                color: "var(--muted)",
                cursor: guidedRunning ? "not-allowed" : "pointer",
                fontSize: 12,
              }}
            >
              Runbook From Incident
            </button>
          </div>
          {incidentPlaybook.length > 0 && (
            <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6 }}>
              {incidentPlaybook.map((cmd) => (
                <button
                  key={cmd}
                  onClick={() => copyCommand(cmd)}
                  style={{
                    padding: "4px 8px",
                    borderRadius: 999,
                    border: "1px solid var(--border)",
                    background: "transparent",
                    color: "var(--muted)",
                    cursor: "pointer",
                    fontSize: 11,
                  }}
                  title={cmd}
                >
                  Copy: {cmd}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {selectedSession && (
        <div className="card">
          <div className="card-title">Session Metadata</div>
          <div style={{ display: "grid", gap: 8 }}>
            <label style={{ color: "var(--muted)", fontSize: 12 }}>
              Tag
              <input
                value={sessionTagDraft}
                onChange={(e) => setSessionTagDraft(e.target.value)}
                placeholder="network / deps / regime / data"
                style={{
                  width: "100%",
                  marginTop: 4,
                  background: "transparent",
                  border: "1px solid var(--border)",
                  color: "var(--text)",
                  borderRadius: 4,
                  padding: "6px 8px",
                }}
              />
            </label>
            <label style={{ color: "var(--muted)", fontSize: 12 }}>
              Note
              <textarea
                value={sessionNoteDraft}
                onChange={(e) => setSessionNoteDraft(e.target.value)}
                rows={3}
                style={{
                  width: "100%",
                  marginTop: 4,
                  background: "transparent",
                  border: "1px solid var(--border)",
                  color: "var(--text)",
                  borderRadius: 4,
                  padding: "6px 8px",
                }}
              />
            </label>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <button
                onClick={saveSessionMeta}
                disabled={writeLocked || sessionMetaSaving}
                style={{
                  padding: "5px 10px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: "var(--muted)",
                  cursor: sessionMetaSaving ? "not-allowed" : "pointer",
                  fontSize: 12,
                }}
              >
                {sessionMetaSaving ? "Saving..." : "Save"}
              </button>
              <span style={{ color: "var(--muted)", fontSize: 12 }}>
                updated: {fmtTs(selectedSession.updated_ts ?? null)}
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">Last Action Output</div>
        {!result ? (
          <div style={{ color: "var(--muted)" }}>Run an action to see output.</div>
        ) : (
          <>
            <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 8 }}>
              <span className={`badge ${result.ok ? "badge-green" : "badge-red"}`}>
                {result.ok ? "SUCCESS" : "FAILED"}
              </span>
              <span style={{ color: "var(--muted)" }}>
                action={result.action} exit={result.exit_code} duration={result.duration_sec.toFixed(2)}s
              </span>
            </div>
            <pre style={{ whiteSpace: "pre-wrap", fontSize: 12, color: "var(--text)" }}>
              {result.output || "(no output)"}
            </pre>
          </>
        )}
      </div>

      <div className="card">
        <div className="card-title">Recent Sessions</div>
        <AsyncState
          loading={sessionsPoll.isLoading}
          error={sessionsPoll.error}
          isEmpty={sessions.length === 0}
          emptyText="No runbook sessions yet"
        >
          <table>
            <thead>
              <tr>
                <th><TermTip term="Session" tr="Debug oturum kimligi." en="Debug session id." /></th>
                <th><TermTip term="Time" tr="Oturum baslangic zamani." en="Session start time." /></th>
                <th><TermTip term="Status" tr="Oturum sonucu." en="Session result status." /></th>
                <th><TermTip term="Failed action" tr="Basarisiz kalan adim." en="Failed step/action." /></th>
                <th><TermTip term="Tag" tr="Operator etiketi." en="Operator tag label." /></th>
                <th><TermTip term="Note" tr="Operator notu." en="Operator note text." /></th>
                <th><TermTip term="Open" tr="Oturum detayini ac." en="Open full session detail." /></th>
                <th>A</th>
                <th>B</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.session_id}>
                  <td style={{ color: "var(--muted)" }}>{s.session_id}</td>
                  <td>{fmtTs(s.started_ts ?? null)}</td>
                  <td>{s.ok ? "pass" : "fail"}</td>
                  <td>{s.failed_action ?? "-"}</td>
                  <td>{s.tag ?? "-"}</td>
                  <td style={{ color: "var(--muted)", fontSize: 12 }}>{s.note_preview ?? "-"}</td>
                  <td>
                    <button
                      onClick={() => loadSession(s.session_id)}
                      style={{
                        padding: "3px 8px",
                        borderRadius: 4,
                        border: "1px solid var(--border)",
                        background: "transparent",
                        color: "var(--muted)",
                        cursor: "pointer",
                        fontSize: 12,
                      }}
                    >
                      Load
                    </button>
                  </td>
                  <td>
                    <button
                      onClick={() => setCompareAId(s.session_id)}
                      style={{
                        padding: "3px 8px",
                        borderRadius: 4,
                        border: "1px solid var(--border)",
                        background: compareAId === s.session_id ? "var(--accent-dim)" : "transparent",
                        color: "var(--muted)",
                        cursor: "pointer",
                        fontSize: 12,
                      }}
                    >
                      A
                    </button>
                  </td>
                  <td>
                    <button
                      onClick={() => setCompareBId(s.session_id)}
                      style={{
                        padding: "3px 8px",
                        borderRadius: 4,
                        border: "1px solid var(--border)",
                        background: compareBId === s.session_id ? "var(--accent-dim)" : "transparent",
                        color: "var(--muted)",
                        cursor: "pointer",
                        fontSize: 12,
                      }}
                    >
                      B
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </AsyncState>
      </div>

      <div className="card">
        <div className="card-title">Incident Inbox</div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 10 }}>
          <label style={{ color: "var(--muted)", fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}>
            <input
              type="checkbox"
              checked={policy.enabled}
              onChange={(e) => setPolicy((p) => ({ ...p, enabled: e.target.checked }))}
            />
            auto-run policy
          </label>
          <label style={{ color: "var(--muted)", fontSize: 12 }}>
            min level
            <select
              value={policy.min_level}
              onChange={(e) => setPolicy((p) => ({ ...p, min_level: e.target.value }))}
              style={{ marginLeft: 6, background: "transparent", color: "var(--text)", border: "1px solid var(--border)" }}
            >
              <option value="INFO">INFO</option>
              <option value="WARNING">WARNING</option>
              <option value="ERROR">ERROR</option>
              <option value="CRITICAL">CRITICAL</option>
            </select>
          </label>
          <label style={{ color: "var(--muted)", fontSize: 12 }}>
            cooldown (sec)
            <input
              type="number"
              min={60}
              step={60}
              value={policy.cooldown_sec}
              onChange={(e) => setPolicy((p) => ({ ...p, cooldown_sec: Number(e.target.value || 900) }))}
              style={{
                marginLeft: 6,
                width: 90,
                background: "transparent",
                color: "var(--text)",
                border: "1px solid var(--border)",
                borderRadius: 4,
                padding: "2px 6px",
              }}
            />
          </label>
          <button
            onClick={savePolicy}
            disabled={writeLocked || policySaving}
            style={{
              padding: "4px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: policySaving ? "not-allowed" : "pointer",
              fontSize: 12,
            }}
          >
            {policySaving ? "Saving..." : "Save Policy"}
          </button>
          <button
            onClick={runAutoPolicyOnce}
            disabled={writeLocked}
            style={{
              padding: "4px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            Run Auto Once
          </button>
          <button
            onClick={exportIncidentOpsBundle}
            style={{
              padding: "4px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            Export Ops Bundle
          </button>
          <label style={{ color: "var(--muted)", fontSize: 12 }}>
            status
            <select
              value={incidentStatusFilter}
              onChange={(e) => setIncidentStatusFilter(e.target.value as "all" | "active" | "resolved")}
              style={{ marginLeft: 6, background: "transparent", color: "var(--text)", border: "1px solid var(--border)" }}
            >
              <option value="all">all</option>
              <option value="active">active</option>
              <option value="resolved">resolved</option>
            </select>
          </label>
          <label style={{ color: "var(--muted)", fontSize: 12 }}>
            type
            <select
              value={incidentTypeFilter}
              onChange={(e) => setIncidentTypeFilter(e.target.value)}
              style={{ marginLeft: 6, background: "transparent", color: "var(--text)", border: "1px solid var(--border)" }}
            >
              {incidentTypeOptions.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </label>
          <button
            onClick={() => bulkIncident("ack")}
            disabled={writeLocked}
            style={{
              padding: "4px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            Ack all (filtered)
          </button>
          <button
            onClick={() => bulkIncident("resolve")}
            disabled={writeLocked}
            style={{
              padding: "4px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            Resolve all (filtered)
          </button>
          <button
            onClick={undoIncidentAction}
            disabled={writeLocked}
            style={{
              padding: "4px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            Undo (60s)
          </button>
          <button
            onClick={runStabilizeMacro}
            disabled={writeLocked}
            style={{
              padding: "4px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            Stabilize Run
          </button>
          <label style={{ color: "var(--muted)", fontSize: 12 }}>
            preset
            <select
              value={macroPreset}
              onChange={(e) => {
                const v = e.target.value as MacroPresetName;
                if (v === "custom") return;
                applyMacroPreset(v);
              }}
              style={{ marginLeft: 6, background: "transparent", color: "var(--text)", border: "1px solid var(--border)" }}
            >
              <option value="quick">quick</option>
              <option value="full">full</option>
              <option value="no-export">no-export</option>
              <option value="custom">custom</option>
            </select>
          </label>
          <label style={{ color: "var(--muted)", fontSize: 12 }}>
            owner
            <input
              value={macroOwner}
              onChange={(e) => setMacroOwner(e.target.value)}
              placeholder="local"
              style={{
                marginLeft: 6,
                width: 110,
                background: "transparent",
                color: "var(--text)",
                border: "1px solid var(--border)",
                borderRadius: 4,
                padding: "2px 6px",
              }}
            />
          </label>
          <label style={{ color: "var(--muted)", fontSize: 12, display: "flex", alignItems: "center", gap: 4 }}>
            <input
              type="checkbox"
              checked={macroSteps.ackFiltered}
              onChange={(e) => updateMacroStep({ ackFiltered: e.target.checked })}
            />
            ack
          </label>
          <label style={{ color: "var(--muted)", fontSize: 12, display: "flex", alignItems: "center", gap: 4 }}>
            <input
              type="checkbox"
              checked={macroSteps.autoRun}
              onChange={(e) => updateMacroStep({ autoRun: e.target.checked })}
            />
            auto-run
          </label>
          <label style={{ color: "var(--muted)", fontSize: 12, display: "flex", alignItems: "center", gap: 4 }}>
            <input
              type="checkbox"
              checked={macroSteps.exportMd}
              onChange={(e) => updateMacroStep({ exportMd: e.target.checked })}
            />
            export-md
          </label>
          <label style={{ color: "var(--muted)", fontSize: 12, display: "flex", alignItems: "center", gap: 4 }}>
            <input
              type="checkbox"
              checked={macroSteps.refresh}
              onChange={(e) => updateMacroStep({ refresh: e.target.checked })}
            />
            refresh
          </label>
          <span style={{ color: "var(--muted)", fontSize: 12 }}>
            eligible (filtered): {bulkPreviewEligible == null ? "-" : bulkPreviewEligible}
          </span>
          <span style={{ color: "var(--muted)", fontSize: 12 }}>
            preset by: {macroOwner || "-"} @ {fmtTs(macroUpdatedTs)}
          </span>
          <span style={{ color: "var(--muted)", fontSize: 12 }}>
            SLA active={incidentSla.active}/{incidentSla.total}
            {" | "}new_age_avg={incidentSla.avgNewAge == null ? "-" : `${incidentSla.avgNewAge.toFixed(0)}s`}
            {" | "}ack_avg={incidentSla.avgAckLag == null ? "-" : `${incidentSla.avgAckLag.toFixed(0)}s`}
            {" | "}resolve_avg={incidentSla.avgResolveLag == null ? "-" : `${incidentSla.avgResolveLag.toFixed(0)}s`}
          </span>
        </div>
        {incidentError && <div style={{ color: "var(--red)", fontSize: 12, marginBottom: 8 }}>{incidentError}</div>}
        <AsyncState
          loading={incidentLoading}
          error={null}
          isEmpty={filteredIncidents.length === 0}
          emptyText="No incidents"
        >
          <table>
            <thead>
              <tr>
                <th><TermTip term="Time" tr="Incident olusma zamani." en="Incident timestamp." /></th>
                <th><TermTip term="Type" tr="Incident tur sinifi." en="Incident type class." /></th>
                <th><TermTip term="Level" tr="Ciddiyet seviyesi." en="Severity level." /></th>
                <th><TermTip term="Status" tr="Ack/resolve durumu." en="Ack/resolve status." /></th>
                <th><TermTip term="Failed" tr="Bagli failed action." en="Related failed action." /></th>
                <th><TermTip term="Actions" tr="Uygulanabilir operasyonlar." en="Available operations." /></th>
              </tr>
            </thead>
              <tbody>
                {filteredIncidents.map((inc) => (
                  <tr
                    key={inc.incident_id}
                    style={
                      isResearchFitnessIncident(inc)
                        ? {
                            boxShadow: `inset 3px 0 0 ${incidentAccent(inc.level)}`,
                            background: "color-mix(in srgb, var(--panel) 88%, var(--accent) 12%)",
                          }
                        : undefined
                    }
                  >
                    <td style={{ color: "var(--muted)" }}>{fmtTs(inc.ts ?? null)}</td>
                    <td>
                      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                        <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                          <span>{inc.type}</span>
                          {isResearchFitnessIncident(inc) && (
                            <span className={`badge ${String(inc.level).toUpperCase() === "ERROR" ? "badge-red" : "badge-yellow"}`}>
                              FITNESS
                            </span>
                          )}
                        </div>
                        {isResearchFitnessIncident(inc) && (
                          <div style={{ fontSize: 11, color: "var(--muted)", maxWidth: 420 }}>
                            {inc.detail || inc.query || "Research fitness degraded"}
                          </div>
                        )}
                      </div>
                    </td>
                    <td>{inc.level}</td>
                    <td>{inc.status}{inc.muted ? " (muted)" : ""}</td>
                    <td>
                      {isResearchFitnessIncident(inc) ? (
                        <span style={{ color: "var(--muted)", fontSize: 11 }}>
                          review research coverage
                        </span>
                      ) : (
                        inc.failed_action ?? "-"
                      )}
                    </td>
                    <td style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      <button onClick={() => runIncident(inc.incident_id)} disabled={writeLocked} style={{ fontSize: 11 }}>Run now</button>
                      <button onClick={() => patchIncident(inc.incident_id, "ack")} disabled={writeLocked} style={{ fontSize: 11 }}>Ack</button>
                      <button onClick={() => patchIncident(inc.incident_id, "snooze_type", inc.type)} disabled={writeLocked} style={{ fontSize: 11 }}>Snooze type</button>
                      {inc.muted ? (
                        <button onClick={() => patchIncident(inc.incident_id, "unmute_type", inc.type)} disabled={writeLocked} style={{ fontSize: 11 }}>Unmute type</button>
                      ) : (
                        <button onClick={() => patchIncident(inc.incident_id, "mute_type", inc.type)} disabled={writeLocked} style={{ fontSize: 11 }}>Mute type</button>
                      )}
                      {isResearchFitnessIncident(inc) && (
                        <>
                          <button onClick={() => navigate("/research")} style={{ fontSize: 11 }}>
                            Open Research
                          </button>
                          <button
                            onClick={() => void runPreflightFromIncident()}
                            disabled={writeLocked || runningAction === "preflight_check"}
                            style={{ fontSize: 11 }}
                          >
                            {runningAction === "preflight_check" ? "Running..." : "Run Preflight"}
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
        </AsyncState>
        <div style={{ marginTop: 10 }}>
          <div style={{ color: "var(--muted)", fontSize: 12, marginBottom: 6 }}>Recent Incident Audit</div>
          {incidentAudit.length === 0 ? (
            <div style={{ color: "var(--muted)", fontSize: 12 }}>No audit events.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th><TermTip term="Time" tr="Audit olay zamani." en="Audit event time." /></th>
                  <th><TermTip term="Operator" tr="Aksiyonu yapan kisi/sistem." en="Actor/operator id." /></th>
                  <th><TermTip term="Kind" tr="Audit kategori tipi." en="Audit category kind." /></th>
                  <th><TermTip term="Action" tr="Yapilan incident aksiyonu." en="Applied incident action." /></th>
                  <th><TermTip term="Type" tr="Hedef incident turu." en="Target incident type." /></th>
                  <th><TermTip term="Updated" tr="Guncellenen incident sayisi." en="Count of updated incidents." /></th>
                </tr>
              </thead>
              <tbody>
                {incidentAudit.map((a, idx) => (
                  <tr key={`${a.ts}-${idx}`}>
                    <td style={{ color: "var(--muted)" }}>{fmtTs(a.ts ?? null)}</td>
                    <td>{a.operator ?? "-"}</td>
                    <td>{a.kind ?? "-"}</td>
                    <td>{typeof a.action === "string" ? a.action : JSON.stringify(a.action ?? "")}</td>
                    <td>{a.incident_type ?? "-"}</td>
                    <td>{a.updated ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div style={{ marginTop: 10 }}>
          <div style={{ color: "var(--muted)", fontSize: 12, marginBottom: 6 }}>Recent Security Audit</div>
          {securityAudit.length === 0 ? (
            <div style={{ color: "var(--muted)", fontSize: 12 }}>No security events.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th><TermTip term="Time" tr="Guvenlik olayi zamani." en="Security event time." /></th>
                  <th><TermTip term="Kind" tr="Reddedilen/limitlenen olay tipi." en="Denied/limited event kind." /></th>
                  <th><TermTip term="Role" tr="Istekteki rol." en="Request role." /></th>
                  <th><TermTip term="Operator" tr="Istek operator kimligi." en="Request operator id." /></th>
                  <th><TermTip term="Path" tr="Hedef API yolu." en="Target API path." /></th>
                  <th><TermTip term="Detail" tr="Ek guvenlik detayi." en="Additional security detail." /></th>
                </tr>
              </thead>
              <tbody>
                {securityAudit.map((a, idx) => (
                  <tr key={`${a.ts}-${a.kind}-${idx}`}>
                    <td style={{ color: "var(--muted)" }}>{fmtTs(a.ts ?? null)}</td>
                    <td>{a.kind}</td>
                    <td>{a.role ?? "-"}</td>
                    <td>{a.operator ?? "-"}</td>
                    <td style={{ color: "var(--muted)", fontSize: 12 }}>{a.path ?? "-"}</td>
                    <td style={{ color: "var(--muted)", fontSize: 12 }}>{a.detail ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-title">Session Compare</div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 8 }}>
          <span style={{ color: "var(--muted)", fontSize: 12 }}>A: {compareAId || "-"}</span>
          <span style={{ color: "var(--muted)", fontSize: 12 }}>B: {compareBId || "-"}</span>
          <button
            onClick={runCompare}
            disabled={compareLoading}
            style={{
              padding: "5px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: compareLoading ? "not-allowed" : "pointer",
              fontSize: 12,
            }}
          >
            {compareLoading ? "Comparing..." : "Compare"}
          </button>
          <button
            onClick={() => {
              setCompareAId("");
              setCompareBId("");
              setCompareA(null);
              setCompareB(null);
              setCompareError(null);
            }}
            style={{
              padding: "5px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            Clear
          </button>
          <label style={{ color: "var(--muted)", fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}>
            <input
              type="checkbox"
              checked={compareOnlyFailed}
              onChange={(e) => setCompareOnlyFailed(e.target.checked)}
            />
            only failed steps
          </label>
          <button
            onClick={exportCompareJson}
            disabled={!compareSummary}
            style={{
              padding: "5px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: compareSummary ? "pointer" : "not-allowed",
              fontSize: 12,
            }}
          >
            Export Compare JSON
          </button>
          <button
            onClick={exportCompareMarkdown}
            disabled={!compareSummary}
            style={{
              padding: "5px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: compareSummary ? "pointer" : "not-allowed",
              fontSize: 12,
            }}
          >
            Export Compare MD
          </button>
          <button
            onClick={copyCompareLink}
            disabled={!compareAId && !compareBId}
            style={{
              padding: "5px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: compareAId || compareBId ? "pointer" : "not-allowed",
              fontSize: 12,
            }}
            title="Copy compare URL"
          >
            Copy Share Link
          </button>
          {compareLinkCopied && (
            <span style={{ color: "var(--green)", fontSize: 12 }}>Copied</span>
          )}
        </div>
        {compareError && <div style={{ color: "var(--red)", fontSize: 12, marginBottom: 8 }}>{compareError}</div>}
        {compareSummary ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ color: "var(--muted)", fontSize: 12 }}>
              A: ok={String(compareSummary.aOk)} failed={compareSummary.aFailed} incident={compareSummary.aIncident} snippets={compareSummary.aSnips}
            </div>
            <div style={{ color: "var(--muted)", fontSize: 12 }}>
              B: ok={String(compareSummary.bOk)} failed={compareSummary.bFailed} incident={compareSummary.bIncident} snippets={compareSummary.bSnips}
            </div>
            <div style={{ fontSize: 12, color: "var(--muted)" }}>
              Step differences: {compareSummary.stepDiffs.length}
            </div>
            {compareSummary.stepRows.length > 0 && (
              <table>
                <thead>
                  <tr>
                    <th><TermTip term="Action" tr="Karsilastirilan adim." en="Compared action/step." /></th>
                    <th><TermTip term="A" tr="A oturumu sonucu." en="Session A outcome." /></th>
                    <th><TermTip term="B" tr="B oturumu sonucu." en="Session B outcome." /></th>
                    <th><TermTip term="Changed" tr="Iki oturum farkli mi." en="Whether result changed." /></th>
                  </tr>
                </thead>
                <tbody>
                  {compareSummary.stepRows
                    .filter((d) => {
                      if (!compareOnlyFailed) return true;
                      return d.a === "fail" || d.b === "fail";
                    })
                    .map((d) => (
                    <tr key={d.action}>
                      <td>{d.action}</td>
                      <td>{d.a}</td>
                      <td>{d.b}</td>
                      <td>{d.changed ? "yes" : "no"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ) : (
          <div style={{ color: "var(--muted)", fontSize: 12 }}>Pick two sessions from Recent Sessions using A/B, then click Compare.</div>
        )}
      </div>

      {selectedSession?.log_snippets && selectedSession.log_snippets.length > 0 && (
        <div className="card">
          <div className="card-title">Log Snippets (Session)</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {selectedSession.log_snippets.map((sn, idx) => (
              <div key={`${sn.file}-${sn.line_no}-${idx}`} style={{ border: "1px solid var(--border)", borderRadius: 4, padding: 8 }}>
                <div style={{ color: "var(--muted)", fontSize: 11, marginBottom: 6 }}>
                  {sn.file}:{sn.line_no}
                </div>
                {sn.context_before?.map((ln, i) => (
                  <div key={`b-${i}`} style={{ fontSize: 11, color: "var(--muted)", whiteSpace: "pre-wrap" }}>
                    {ln}
                  </div>
                ))}
                <div style={{ fontSize: 12, color: "var(--yellow)", whiteSpace: "pre-wrap", fontWeight: 700 }}>
                  {sn.match}
                </div>
                {sn.context_after?.map((ln, i) => (
                  <div key={`a-${i}`} style={{ fontSize: 11, color: "var(--muted)", whiteSpace: "pre-wrap" }}>
                    {ln}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      {selectedSession && (
        <div className="card">
          <div className="card-title">Session Timeline</div>
          {timeline.length === 0 ? (
            <div style={{ color: "var(--muted)", fontSize: 12 }}>No timeline events.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th><TermTip term="Time" tr="Timeline olay zamani." en="Timeline event timestamp." /></th>
                  <th><TermTip term="Kind" tr="Olay kategorisi." en="Event category kind." /></th>
                  <th><TermTip term="Title" tr="Kisa olay basligi." en="Short event title." /></th>
                  <th><TermTip term="Status" tr="Olay durum etiketi." en="Event status label." /></th>
                  <th><TermTip term="Detail" tr="Detay aciklama." en="Detailed context." /></th>
                </tr>
              </thead>
              <tbody>
                {timeline.map((ev, idx) => (
                  <tr key={`${ev.kind}-${ev.ts}-${idx}`}>
                    <td style={{ color: "var(--muted)" }}>{fmtTs(ev.ts ?? null)}</td>
                    <td>{ev.kind}</td>
                    <td>{ev.title}</td>
                    <td>{ev.status ?? "-"}</td>
                    <td style={{ color: "var(--muted)", fontSize: 12 }}>{ev.detail ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <div className="card">
        <div className="card-title">Action History</div>
        <AsyncState
          loading={historyPoll.isLoading}
          error={historyPoll.error}
          isEmpty={history.length === 0}
          emptyText="No history yet"
        >
          <table>
            <thead>
              <tr>
                <th><TermTip term="Time" tr="Komut calisma zamani." en="Action execution time." /></th>
                <th><TermTip term="Action" tr="Cagrilan debug aksiyonu." en="Invoked debug action." /></th>
                <th><TermTip term="Status" tr="Komut basari durumu." en="Action success status." /></th>
                <th><TermTip term="Exit" tr="Process cikis kodu." en="Process exit code." /></th>
                <th><TermTip term="Duration" tr="Calisma suresi (ms)." en="Execution duration (ms)." /></th>
              </tr>
            </thead>
            <tbody>
              {history.map((h, idx) => (
                <tr key={`${h.ts}-${h.action}-${idx}`}>
                  <td style={{ color: "var(--muted)" }}>{fmtTs(h.ts)}</td>
                  <td>{h.action}</td>
                  <td>{h.ok == null ? "-" : h.ok ? "ok" : "fail"}</td>
                  <td>{h.exit_code == null ? "-" : h.exit_code}</td>
                  <td>{h.duration_sec == null ? "-" : `${h.duration_sec.toFixed(2)}s`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </AsyncState>
      </div>
    </div>
  );
}
