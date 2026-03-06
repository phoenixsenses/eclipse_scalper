import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type {
  AutoRunbookPolicy,
  ControlActionResult,
  DiagConnectivityResponse,
  IncidentInboxItem,
  OpsHealthResponse,
  RunbookSessionDetail,
  RuntimeStatus,
  SessionTimelineEvent,
  SupervisorStatusResponse,
} from "../api/types";
import AsyncState from "../components/AsyncState";
import DegradedBanner, { type DegradedMode } from "../components/DegradedBanner";
import PageGuide from "../components/PageGuide";
import { usePoll } from "../hooks/usePoll";
import { useBackendStatus } from "../context/BackendStatusContext";
import { useDashboardAuth } from "../context/AuthContext";
import { useApiErrors } from "../context/ApiErrorContext";

interface IncidentSessionEvent {
  ts: number;
  action: string;
  detail?: string;
}

interface StepCompareRow {
  action: string;
  a: string;
  b: string;
  changed: boolean;
}

interface OpsAckState {
  data_ack_ts?: number;
  network_ack_ts?: number;
}

interface RootCauseItem {
  key: string;
  label: string;
  confidence: number;
  evidence: string[];
  action: string;
}

const OPS_ACK_TTL_MS = 30 * 60 * 1000;

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function parseRateLimitFromLines(lines: string[]): { used?: number; cap?: number; pct?: number } {
  const re = /\[RATE_LIMIT\]\s+used_1m=(\d+)\s+cap_1m=(\d+)\s+usage_pct=([\d.]+)/i;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const m = re.exec(lines[i]);
    if (!m) continue;
    return { used: Number(m[1]), cap: Number(m[2]), pct: Number(m[3]) };
  }
  return {};
}

export default function ControlTower() {
  const navigate = useNavigate();
  const backend = useBackendStatus();
  const { auth } = useDashboardAuth();
  const { events: apiErrors } = useApiErrors();
  const writeLocked = auth.role === "viewer";

  const fetchRuntime = useCallback((signal: AbortSignal) => api.runtime(signal), []);
  const fetchIncidents = useCallback((signal: AbortSignal) => api.debugIncidents(20, signal), []);
  const fetchPolicy = useCallback((signal: AbortSignal) => api.debugIncidentPolicy(signal), []);
  const fetchRate = useCallback(async (signal: AbortSignal) => {
    const res = await api.logTail("paper_trading.log", 300, signal);
    return parseRateLimitFromLines(res.lines ?? []);
  }, []);

  const runtimePoll = usePoll<RuntimeStatus>({
    fetcher: fetchRuntime,
    pollKey: "api:/runtime:tower",
    intervalMs: 3000,
    staleAfterMs: 10000,
    enabled: backend.backendUp,
  });
  const incidentPoll = usePoll<IncidentInboxItem[]>({
    fetcher: fetchIncidents,
    pollKey: "api:/debug/incidents:tower",
    intervalMs: 15000,
    staleAfterMs: 45000,
    enabled: backend.backendUp,
  });
  const policyPoll = usePoll<AutoRunbookPolicy>({
    fetcher: fetchPolicy,
    pollKey: "api:/debug/incidents-policy",
    intervalMs: 20000,
    staleAfterMs: 60000,
    enabled: backend.backendUp,
  });
  const ratePoll = usePoll<{ used?: number; cap?: number; pct?: number }>({
    fetcher: fetchRate,
    pollKey: "api:/logs/tail:paper_trading",
    intervalMs: 20000,
    staleAfterMs: 60000,
    enabled: backend.backendUp,
  });
  const fetchOpsHealth = useCallback((signal: AbortSignal) => api.opsHealth(signal), []);
  const opsPoll = usePoll<OpsHealthResponse>({
    fetcher: fetchOpsHealth,
    pollKey: "api:/ops/health",
    intervalMs: 15000,
    staleAfterMs: 45000,
    enabled: backend.backendUp,
  });
  const fetchSupervisor = useCallback((signal: AbortSignal) => api.supervisorStatus(signal), []);
  const supervisorPoll = usePoll<SupervisorStatusResponse>({
    fetcher: fetchSupervisor,
    pollKey: "api:/ops/supervisor",
    intervalMs: 10000,
    staleAfterMs: 30000,
    enabled: backend.backendUp,
  });

  const [incidentSessionId, setIncidentSessionId] = useState("");
  const [incidentTimeline, setIncidentTimeline] = useState<IncidentSessionEvent[]>([]);
  const [lastTriage, setLastTriage] = useState<{ ok: boolean; ts: string; failedAction?: string } | null>(null);
  const [policyDraft, setPolicyDraft] = useState<AutoRunbookPolicy>({
    enabled: false,
    min_level: "WARNING",
    cooldown_sec: 900,
    last_run_ts_by_type: {},
  });
  const [policySaving, setPolicySaving] = useState(false);
  const [policyMsg, setPolicyMsg] = useState<string>("");
  const [autoRunMsg, setAutoRunMsg] = useState<string>("");
  const [selectedIncident, setSelectedIncident] = useState<IncidentInboxItem | null>(null);
  const [sessionDetail, setSessionDetail] = useState<RunbookSessionDetail | null>(null);
  const [sessionTimeline, setSessionTimeline] = useState<SessionTimelineEvent[]>([]);
  const [sessionLoading, setSessionLoading] = useState(false);
  const [sessionError, setSessionError] = useState<string>("");
  const [compareAId, setCompareAId] = useState("");
  const [compareBId, setCompareBId] = useState("");
  const [compareA, setCompareA] = useState<RunbookSessionDetail | null>(null);
  const [compareB, setCompareB] = useState<RunbookSessionDetail | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState("");
  const [compareOnlyFailed, setCompareOnlyFailed] = useState(false);
  const [opsActionRunning, setOpsActionRunning] = useState<string>("");
  const [opsActionResult, setOpsActionResult] = useState<ControlActionResult | null>(null);
  const [opsActionError, setOpsActionError] = useState("");
  const [diagRunning, setDiagRunning] = useState(false);
  const [diagData, setDiagData] = useState<DiagConnectivityResponse | null>(null);
  const [diagError, setDiagError] = useState("");
  const [opsAck, setOpsAck] = useState<OpsAckState>(() => {
    try {
      const raw = localStorage.getItem("eclipse.tower.ops_ack.v1");
      if (!raw) return {};
      return JSON.parse(raw) as OpsAckState;
    } catch {
      return {};
    }
  });
  const [nowTs, setNowTs] = useState<number>(Date.now());

  useEffect(() => {
    const sync = () => {
      setIncidentSessionId(readJson<string>("eclipse.logs.incident_session.v1", ""));
      setIncidentTimeline(readJson<IncidentSessionEvent[]>("eclipse.logs.incident_timeline.v1", []));
      const tri = readJson<{ ok: boolean; ts: string; failedAction?: string } | null>("eclipse.debug.last_triage.v1", null);
      setLastTriage(tri);
    };
    sync();
    const t = window.setInterval(sync, 5000);
    window.addEventListener("focus", sync);
    return () => {
      window.clearInterval(t);
      window.removeEventListener("focus", sync);
    };
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("eclipse.tower.ops_ack.v1", JSON.stringify(opsAck));
    } catch {
      // ignore storage failures
    }
  }, [opsAck]);

  useEffect(() => {
    const id = window.setInterval(() => setNowTs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    setOpsAck((prev) => {
      const next: OpsAckState = { ...prev };
      let changed = false;
      if (next.data_ack_ts && nowTs - next.data_ack_ts > OPS_ACK_TTL_MS) {
        delete next.data_ack_ts;
        changed = true;
      }
      if (next.network_ack_ts && nowTs - next.network_ack_ts > OPS_ACK_TTL_MS) {
        delete next.network_ack_ts;
        changed = true;
      }
      return changed ? next : prev;
    });
  }, [nowTs]);

  useEffect(() => {
    if (policyPoll.data) {
      setPolicyDraft(policyPoll.data);
    }
  }, [policyPoll.data]);

  const loadSession = useCallback(async (sessionId: string) => {
    setSessionLoading(true);
    setSessionError("");
    try {
      const [detail, timeline] = await Promise.all([
        api.debugSessionDetail(sessionId),
        api.debugSessionTimeline(sessionId),
      ]);
      setSessionDetail(detail);
      setSessionTimeline(timeline);
    } catch (err) {
      setSessionDetail(null);
      setSessionTimeline([]);
      setSessionError(err instanceof Error ? err.message : String(err));
    } finally {
      setSessionLoading(false);
    }
  }, []);

  const runIncidentNow = useCallback(async (incidentId: string) => {
    setSessionLoading(true);
    setSessionError("");
    try {
      const detail = await api.runDebugIncidentRunbook(incidentId);
      setSessionDetail(detail);
      const timeline = await api.debugSessionTimeline(detail.session_id);
      setSessionTimeline(timeline);
      incidentPoll.refresh();
      setAutoRunMsg(`Runbook executed: ${detail.session_id}`);
    } catch (err) {
      setSessionError(err instanceof Error ? err.message : String(err));
    } finally {
      setSessionLoading(false);
    }
  }, [incidentPoll]);

  const savePolicy = useCallback(async () => {
    setPolicySaving(true);
    setPolicyMsg("");
    try {
      const next = await api.patchDebugIncidentPolicy({
        enabled: !!policyDraft.enabled,
        min_level: policyDraft.min_level || "WARNING",
        cooldown_sec: Math.max(30, Number(policyDraft.cooldown_sec || 900)),
        last_run_ts_by_type: policyDraft.last_run_ts_by_type ?? {},
      });
      setPolicyDraft(next);
      setPolicyMsg("Policy saved.");
      policyPoll.refresh();
    } catch (err) {
      setPolicyMsg(`Save failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPolicySaving(false);
    }
  }, [policyDraft, policyPoll]);

  const runAutoOnce = useCallback(async () => {
    setAutoRunMsg("");
    try {
      const res = await api.runAutoRunbookOnce();
      if (res.ran) {
        setAutoRunMsg(`Auto-run executed: incident=${res.incident_id ?? "-"} session=${res.session_id ?? "-"}`);
        if (res.session_id) {
          await loadSession(res.session_id);
        }
      } else {
        setAutoRunMsg(`Auto-run skipped: ${res.reason ?? "no eligible incidents"}`);
      }
      incidentPoll.refresh();
      policyPoll.refresh();
    } catch (err) {
      setAutoRunMsg(`Auto-run failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [incidentPoll, loadSession, policyPoll]);

  const loadCompare = useCallback(async () => {
    if (!compareAId || !compareBId) {
      setCompareError("Select A and B sessions first.");
      return;
    }
    setCompareLoading(true);
    setCompareError("");
    try {
      const [a, b] = await Promise.all([
        api.debugSessionDetail(compareAId),
        api.debugSessionDetail(compareBId),
      ]);
      setCompareA(a);
      setCompareB(b);
    } catch (err) {
      setCompareError(err instanceof Error ? err.message : String(err));
      setCompareA(null);
      setCompareB(null);
    } finally {
      setCompareLoading(false);
    }
  }, [compareAId, compareBId]);

  const activeIncidents = useMemo(
    () => (incidentPoll.data ?? []).filter((x) => (x.status || "").toLowerCase() === "active"),
    [incidentPoll.data]
  );
  const mode: DegradedMode = useMemo(() => {
    if (runtimePoll.error || incidentPoll.error) return "down";
    if (runtimePoll.isStale || incidentPoll.isStale) return "degraded";
    return "ok";
  }, [incidentPoll.error, incidentPoll.isStale, runtimePoll.error, runtimePoll.isStale]);

  const compareRows = useMemo<StepCompareRow[]>(() => {
    if (!compareA || !compareB) return [];
    const aMap = new Map(compareA.steps.map((s) => [s.action, s]));
    const bMap = new Map(compareB.steps.map((s) => [s.action, s]));
    const actions = Array.from(new Set([...aMap.keys(), ...bMap.keys()]));
    return actions.map((action) => {
      const a = aMap.get(action);
      const b = bMap.get(action);
      const aVal = a ? (a.ok ? "ok" : "fail") : "-";
      const bVal = b ? (b.ok ? "ok" : "fail") : "-";
      return { action, a: aVal, b: bVal, changed: aVal !== bVal };
    });
  }, [compareA, compareB]);

  const opsBreaches = useMemo(() => {
    const rowsData: string[] = [];
    const rowsNet: string[] = [];
    const ops = opsPoll.data;
    if (!ops) return { data: rowsData, network: rowsNet };

    const diskFree = Number(ops.data_integrity?.disk_free_gb ?? 0);
    const diskMin = Number(ops.thresholds?.disk_free_min_gb ?? 10);
    if (ops.data_integrity?.disk_free_gb != null && diskFree < diskMin) {
      rowsData.push(`Disk low: ${diskFree.toFixed(2)}GB < ${diskMin.toFixed(2)}GB`);
    }
    const walMb = Number(ops.data_integrity?.wal_size_bytes ?? 0) / (1024 * 1024);
    const walWarn = Number(ops.thresholds?.wal_warn_mb ?? 2048);
    if (walMb > walWarn) {
      rowsData.push(`WAL high: ${walMb.toFixed(1)}MB > ${walWarn.toFixed(1)}MB`);
    }
    const backupAge = Number(ops.data_integrity?.backup_age_sec ?? 0);
    const backupWarn = Number(ops.thresholds?.backup_warn_sec ?? 86400);
    if (ops.data_integrity?.backup_age_sec != null && backupAge > backupWarn) {
      rowsData.push(`Backup stale: ${(backupAge / 3600).toFixed(2)}h > ${(backupWarn / 3600).toFixed(2)}h`);
    }

    const reconnects = Number(ops.network?.reconnects_last_5m ?? 0);
    const reconnectWarn = Number(ops.thresholds?.reconnect_warn_5m ?? 10);
    if (reconnects >= reconnectWarn) {
      rowsNet.push(`Reconnect storm: ${reconnects} >= ${reconnectWarn}`);
    }
    const usage = Number(ops.network?.usage_pct ?? 0);
    const usageWarn = Number(ops.thresholds?.rate_warn_pct ?? 80);
    if (ops.network?.usage_pct != null && usage >= usageWarn) {
      rowsNet.push(`Rate pressure: ${usage.toFixed(1)}% >= ${usageWarn.toFixed(1)}%`);
    }

    return { data: rowsData, network: rowsNet };
  }, [opsPoll.data]);

  const trend = useMemo(() => {
    const h = (opsPoll.data?.history ?? []).slice(-120);
    const disk: number[] = [];
    const walMb: number[] = [];
    const reconnects: number[] = [];
    const usagePct: number[] = [];
    for (const r of h) {
      const d = Number(r.data_integrity?.disk_free_gb);
      const w = Number(r.data_integrity?.wal_size_bytes);
      const rc = Number(r.network?.reconnects_last_5m);
      const up = Number(r.network?.usage_pct);
      if (Number.isFinite(d)) disk.push(d);
      if (Number.isFinite(w)) walMb.push(w / (1024 * 1024));
      if (Number.isFinite(rc)) reconnects.push(rc);
      if (Number.isFinite(up)) usagePct.push(up);
    }
    const direction = (arr: number[]) => {
      if (arr.length < 2) return "flat";
      const first = arr[0];
      const last = arr[arr.length - 1];
      if (Math.abs(last - first) < 1e-9) return "flat";
      return last > first ? "up" : "down";
    };
    return {
      disk,
      walMb,
      reconnects,
      usagePct,
      diskDir: direction(disk),
      walDir: direction(walMb),
      reconnectDir: direction(reconnects),
      usageDir: direction(usagePct),
    };
  }, [opsPoll.data?.history]);

  const renderSpark = (arr: number[], warnAt?: number, invertGood?: boolean) => {
    const values = arr.slice(-30);
    if (!values.length) {
      return <div style={{ fontSize: 11, color: "var(--muted)" }}>no data</div>;
    }
    const max = Math.max(...values, 1);
    return (
      <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 40, border: "1px solid var(--border)", borderRadius: 4, padding: "2px 4px" }}>
        {values.map((v, i) => {
          const h = Math.max(2, Math.min(36, Math.round((v / max) * 36)));
          let color = "var(--accent)";
          if (warnAt != null && Number.isFinite(warnAt)) {
            const bad = invertGood ? v <= warnAt : v >= warnAt;
            if (bad) color = "var(--yellow)";
          }
          return <span key={`sp_${i}`} style={{ width: 5, height: h, background: color, borderRadius: 2, display: "inline-block" }} title={v.toFixed(2)} />;
        })}
      </div>
    );
  };

  const dataAckActive = !!opsAck.data_ack_ts && (nowTs - opsAck.data_ack_ts) <= OPS_ACK_TTL_MS;
  const netAckActive = !!opsAck.network_ack_ts && (nowTs - opsAck.network_ack_ts) <= OPS_ACK_TTL_MS;
  const dataAckLeftSec = dataAckActive ? Math.max(0, Math.ceil((OPS_ACK_TTL_MS - (nowTs - Number(opsAck.data_ack_ts))) / 1000)) : 0;
  const netAckLeftSec = netAckActive ? Math.max(0, Math.ceil((OPS_ACK_TTL_MS - (nowTs - Number(opsAck.network_ack_ts))) / 1000)) : 0;
  const rootCauseSummary = useMemo<RootCauseItem[]>(() => {
    const items: RootCauseItem[] = [];
    const add = (item: RootCauseItem) => items.push(item);
    const incidentTypes = (incidentPoll.data ?? []).map((x) => String(x.type || "").toLowerCase());
    const apiMsgs = (apiErrors ?? []).slice(0, 20).map((e) => `${e.key} ${e.message}`.toLowerCase());

    const netEvidence: string[] = [];
    if (opsBreaches.network.length > 0) netEvidence.push(...opsBreaches.network);
    if (incidentTypes.some((t) => t.includes("timeout") || t.includes("exchange"))) netEvidence.push("incident includes timeout/exchange");
    if (apiMsgs.some((m) => m.includes("/runtime") || m.includes("/health") || m.includes("network error"))) {
      netEvidence.push("api network/runtime fetch failures");
    }
    if (netEvidence.length > 0) {
      add({
        key: "network_instability",
        label: "Network / Exchange Instability",
        confidence: Math.min(0.98, 0.45 + netEvidence.length * 0.12),
        evidence: netEvidence.slice(0, 4),
        action: "Run Connectivity Check -> then Run Preflight -> inspect Logs: Timeout",
      });
    }

    const dataEvidence: string[] = [];
    if (opsBreaches.data.length > 0) dataEvidence.push(...opsBreaches.data);
    if (incidentTypes.some((t) => t.includes("data") || t.includes("stale"))) dataEvidence.push("incident includes data freshness/stale");
    if (diagData?.hints?.some((h) => h.toLowerCase().includes("stale") || h.toLowerCase().includes("missing"))) {
      dataEvidence.push(...(diagData.hints || []).slice(0, 2));
    }
    if (dataEvidence.length > 0) {
      add({
        key: "data_integrity_pressure",
        label: "Data Integrity / Storage Pressure",
        confidence: Math.min(0.98, 0.42 + dataEvidence.length * 0.13),
        evidence: dataEvidence.slice(0, 4),
        action: "Run DB Maintenance -> verify backup age + disk free -> rerun Connectivity Check",
      });
    }

    const appEvidence: string[] = [];
    if (apiMsgs.some((m) => m.includes("/debug/") || m.includes("/logs/"))) appEvidence.push("debug/log API errors observed");
    if (incidentTypes.some((t) => t.includes("shutdown"))) appEvidence.push("shutdown-class incident present");
    if (lastTriage && !lastTriage.ok) appEvidence.push(`last triage failed at ${lastTriage.failedAction || "unknown_step"}`);
    if (appEvidence.length > 0) {
      add({
        key: "app_control_path",
        label: "App Control / Runtime Path Issue",
        confidence: Math.min(0.95, 0.35 + appEvidence.length * 0.14),
        evidence: appEvidence.slice(0, 4),
        action: "Run Triage Macro -> open Session Compare (A/B) -> Export Bug Bundle",
      });
    }

    if (items.length === 0) {
      add({
        key: "no_clear_issue",
        label: "No strong anomaly in current window",
        confidence: 0.55,
        evidence: ["No sustained threshold breach or high-severity incident detected"],
        action: "Keep monitoring; run Connectivity Check if UI/API flaps again",
      });
    }

    return items.sort((a, b) => b.confidence - a.confidence).slice(0, 3);
  }, [apiErrors, diagData?.hints, incidentPoll.data, lastTriage, opsBreaches.data, opsBreaches.network]);

  const runOpsAction = useCallback(async (action: string) => {
    setOpsActionRunning(action);
    setOpsActionError("");
    setOpsActionResult(null);
    try {
      const res = await api.runDebugAction(action);
      setOpsActionResult(res);
      if (action === "db_maintenance") {
        opsPoll.refresh();
      }
    } catch (err) {
      setOpsActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setOpsActionRunning("");
    }
  }, [opsPoll]);

  const runConnectivityCheck = useCallback(async () => {
    setDiagRunning(true);
    setDiagError("");
    try {
      const res = await api.diagConnectivity();
      setDiagData(res);
    } catch (err) {
      setDiagError(err instanceof Error ? err.message : String(err));
      setDiagData(null);
    } finally {
      setDiagRunning(false);
    }
  }, []);

  const exportBugBundle = useCallback(() => {
    const payload = {
      exported_at: new Date().toISOString(),
      backend: {
        up: backend.backendUp,
        message: backend.backendMessage,
        lastSuccessAt: backend.lastSuccessAt,
      },
      runtime: runtimePoll.data ?? null,
      ops_health: opsPoll.data ?? null,
      diag_connectivity: diagData,
      active_incident: selectedIncident,
      selected_session_id: sessionDetail?.session_id ?? null,
      api_errors: apiErrors.slice(0, 50),
    };
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `dashboard_bug_bundle_${ts}.json`;
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, [apiErrors, backend.backendMessage, backend.backendUp, backend.lastSuccessAt, diagData, opsPoll.data, runtimePoll.data, selectedIncident, sessionDetail?.session_id]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <PageGuide
        icon="TOWER"
        titleTr="Operasyon Kontrol Kulesi"
        titleEn="Execution Control Tower"
        subtitleTr="Kritik runtime sinyalleri ve incident aksiyonlari tek ekranda."
        subtitleEn="Single-screen view for runtime risk and incident actions."
        items={[
          { icon: "1", titleTr: "Durum", titleEn: "Status", descTr: "Runtime + rate limit + incident yogunlugu.", descEn: "Runtime + rate limit + incident pressure." },
          { icon: "2", titleTr: "Aksiyon", titleEn: "Action", descTr: "Triage, log pack, recover tek tik.", descEn: "One-click triage, log packs, recover." },
          { icon: "3", titleTr: "Iz", titleEn: "Trace", descTr: "Session ve triage gecmisi.", descEn: "Session and triage traceability." },
        ]}
      />

      <DegradedBanner
        mode={mode}
        message={runtimePoll.error?.message ?? incidentPoll.error?.message ?? (!backend.backendUp ? `Backend unreachable (${backend.backendMessage})` : undefined)}
      />
      {!backend.backendUp && (
        <div className="card" style={{ borderStyle: "dashed" }}>
          <div className="card-title">Backend Connection</div>
          <div style={{ color: "var(--yellow)", fontSize: 12 }}>
            Backend is down. Heavy polls are paused. Auto-retry in {Math.max(1, Math.ceil(backend.nextRetryInMs / 1000))}s.
          </div>
        </div>
      )}

      <AsyncState loading={runtimePoll.isLoading} error={runtimePoll.error} loadingText="Loading control tower...">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
          <div className="card">
            <div className="card-title">Runtime</div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>
              {(runtimePoll.data?.collector?.alive ?? false) ? "ALIVE" : "DOWN"}
            </div>
            <div style={{ color: "var(--muted)", fontSize: 11 }}>
              collector_uptime={runtimePoll.data?.collector?.uptime_sec ?? "-"}s
            </div>
          </div>
          <div className="card">
            <div className="card-title">Rate Limit</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: (ratePoll.data?.pct ?? 0) >= 80 ? "var(--yellow)" : "var(--text)" }}>
              {ratePoll.data?.pct != null ? `${ratePoll.data.pct.toFixed(1)}%` : "-"}
            </div>
            <div style={{ color: "var(--muted)", fontSize: 11 }}>
              used={ratePoll.data?.used ?? "-"} cap={ratePoll.data?.cap ?? "-"}
            </div>
          </div>
          <div className="card">
            <div className="card-title">Incidents (active)</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: activeIncidents.length > 0 ? "var(--yellow)" : "var(--green)" }}>
              {activeIncidents.length}
            </div>
          </div>
          <div className="card">
            <div className="card-title">Last Triage</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: lastTriage?.ok ? "var(--green)" : "var(--yellow)" }}>
              {lastTriage ? (lastTriage.ok ? "PASS" : "FAIL") : "-"}
            </div>
            <div style={{ color: "var(--muted)", fontSize: 11 }}>
              {lastTriage?.ts ?? "no triage yet"}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-title self-help" data-help="Hizli aksiyonlar: tek tikla triage, log pack ve recovery adimlari.">Quick Actions</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            <button className="guide-toggle self-help" data-help="Debug triage makrosunu otomatik baslatir." onClick={() => navigate("/debug?auto=triage")}>Run Triage Macro</button>
            <button className="guide-toggle self-help" data-help="No-match kok nedenlerini filtreli log paketinde acar." onClick={() => navigate("/logs?pack=No%20Match")}>Logs: No Match</button>
            <button className="guide-toggle self-help" data-help="Regime uyumsuzlugu log paketini acar." onClick={() => navigate("/logs?pack=Regime")}>Logs: Regime</button>
            <button className="guide-toggle self-help" data-help="Shutdown olaylari log paketini acar." onClick={() => navigate("/logs?pack=Shutdown")}>Logs: Shutdown</button>
            <button className="guide-toggle self-help" data-help="Timeout/network kaynakli olaylari filtreler." onClick={() => navigate("/logs?pack=Timeout")}>Logs: Timeout</button>
            <button className="guide-toggle self-help" data-help="Log sayfasinda fallback/degrade kilidini sifirlar." onClick={() => navigate("/logs?ops=force_recover")}>Force Recover</button>
            <button className="guide-toggle self-help" data-help="Log fallback modunu manuel acik tutar." onClick={() => navigate("/logs?ops=keep_fallback_on")}>Keep Fallback ON</button>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Root Cause Summary (Top 3)</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {rootCauseSummary.map((rc) => (
              <div
                key={rc.key}
                style={{
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  padding: 8,
                  background: "var(--surface-1)",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", gap: 10, marginBottom: 4 }}>
                  <div style={{ fontSize: 13, fontWeight: 700 }}>{rc.label}</div>
                  <span className={`badge ${rc.confidence >= 0.8 ? "badge-yellow" : "badge-blue"}`}>
                    conf={Math.round(rc.confidence * 100)}%
                  </span>
                </div>
                <ul style={{ margin: "0 0 6px 0", paddingLeft: 16, color: "var(--muted)", fontSize: 12 }}>
                  {rc.evidence.map((e, i) => <li key={`${rc.key}_${i}`}>{e}</li>)}
                </ul>
                <div style={{ fontSize: 12 }}>
                  next: <span style={{ color: "var(--text)" }}>{rc.action}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 10 }}>
          <div className="card">
            <div className="card-title">Ops Health: Data Integrity</div>
            <AsyncState loading={opsPoll.isLoading} error={opsPoll.error}>
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
                <span className={`badge ${(opsPoll.data?.status?.data_integrity || "ok") === "critical" ? "badge-red" : (opsPoll.data?.status?.data_integrity || "ok") === "warning" ? "badge-yellow" : "badge-green"}`}>
                  {(opsPoll.data?.status?.data_integrity || "ok").toUpperCase()}
                </span>
                <span style={{ fontSize: 12, color: "var(--muted)" }}>
                  why: {(opsPoll.data?.data_integrity?.disk_free_gb ?? 0) < (opsPoll.data?.thresholds?.disk_free_min_gb ?? 10)
                    ? "low disk"
                    : (Number(opsPoll.data?.data_integrity?.backup_age_sec ?? 0) > Number(opsPoll.data?.thresholds?.backup_warn_sec ?? 86400))
                      ? "backup stale"
                      : (Number(opsPoll.data?.data_integrity?.wal_size_bytes ?? 0) / (1024 * 1024)) > Number(opsPoll.data?.thresholds?.wal_warn_mb ?? 2048)
                        ? "wal growth"
                        : "within thresholds"}
                </span>
              </div>
              <div style={{ fontSize: 12, color: "var(--muted)", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                <span>disk_free_gb: {opsPoll.data?.data_integrity?.disk_free_gb ?? "-"}</span>
                <span>disk_total_gb: {opsPoll.data?.data_integrity?.disk_total_gb ?? "-"}</span>
                <span>db_size_gb: {opsPoll.data?.data_integrity?.db_size_bytes ? (opsPoll.data.data_integrity.db_size_bytes / (1024 ** 3)).toFixed(2) : "-"}</span>
                <span>wal_mb: {opsPoll.data?.data_integrity?.wal_size_bytes ? (opsPoll.data.data_integrity.wal_size_bytes / (1024 ** 2)).toFixed(2) : "0.00"}</span>
                <span>backup_count: {opsPoll.data?.data_integrity?.backup_count ?? 0}</span>
                <span>backup_age_h: {opsPoll.data?.data_integrity?.backup_age_sec ? (opsPoll.data.data_integrity.backup_age_sec / 3600).toFixed(2) : "-"}</span>
              </div>
              {opsBreaches.data.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 11, color: "var(--yellow)", marginBottom: 4 }}>breaches</div>
                  <ul style={{ margin: 0, paddingLeft: 16, color: "var(--muted)", fontSize: 12 }}>
                    {opsBreaches.data.map((b, i) => <li key={`d_${i}`}>{b}</li>)}
                  </ul>
                  <div style={{ marginTop: 6, display: "flex", gap: 8, alignItems: "center" }}>
                    <button className="guide-toggle" onClick={() => setOpsAck((p) => ({ ...p, data_ack_ts: Date.now() }))}>Acknowledge</button>
                    {dataAckActive && opsAck.data_ack_ts && (
                      <span style={{ fontSize: 11, color: "var(--muted)" }}>
                        acked @ {new Date(opsAck.data_ack_ts).toLocaleTimeString()} | expires in {Math.floor(dataAckLeftSec / 60)}m {dataAckLeftSec % 60}s
                      </span>
                    )}
                  </div>
                </div>
              )}
            </AsyncState>
          </div>

          <div className="card">
            <div className="card-title">Ops Health: Network Resilience</div>
            <AsyncState loading={opsPoll.isLoading} error={opsPoll.error}>
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
                <span className={`badge ${(opsPoll.data?.status?.network || "ok") === "warning" ? "badge-yellow" : "badge-green"}`}>
                  {(opsPoll.data?.status?.network || "ok").toUpperCase()}
                </span>
                <span style={{ fontSize: 12, color: "var(--muted)" }}>
                  why: {Number(opsPoll.data?.network?.reconnects_last_5m ?? 0) >= Number(opsPoll.data?.thresholds?.reconnect_warn_5m ?? 10)
                    ? "reconnect storm"
                    : Number(opsPoll.data?.network?.usage_pct ?? 0) >= Number(opsPoll.data?.thresholds?.rate_warn_pct ?? 80)
                      ? "rate pressure"
                      : "within thresholds"}
                </span>
              </div>
              <div style={{ fontSize: 12, color: "var(--muted)", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                <span>collector_connected: {String(opsPoll.data?.network?.collector_connected ?? "-")}</span>
                <span>reconnects_5m: {opsPoll.data?.network?.reconnects_last_5m ?? 0}</span>
                <span>errors_5m: {opsPoll.data?.network?.errors_last_5m ?? 0}</span>
                <span>rate_usage_pct: {opsPoll.data?.network?.usage_pct != null ? `${Number(opsPoll.data.network.usage_pct).toFixed(1)}%` : "-"}</span>
              </div>
              <div style={{ marginTop: 8 }}>
                <div style={{ color: "var(--muted)", fontSize: 11, marginBottom: 4 }}>rate trend (last samples)</div>
                <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 44, border: "1px solid var(--border)", borderRadius: 4, padding: "2px 4px" }}>
                  {(opsPoll.data?.network?.samples ?? []).slice(-20).map((s, i) => {
                    const h = Math.max(2, Math.min(40, Math.round((Number(s.usage_pct) / 100) * 40)));
                    const color = Number(s.usage_pct) >= Number(opsPoll.data?.thresholds?.rate_warn_pct ?? 80) ? "var(--yellow)" : "var(--accent)";
                    return <span key={`bar_${i}`} style={{ width: 6, height: h, background: color, borderRadius: 2, display: "inline-block" }} title={`${Number(s.usage_pct).toFixed(1)}%`} />;
                  })}
                </div>
              </div>
              {opsBreaches.network.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 11, color: "var(--yellow)", marginBottom: 4 }}>breaches</div>
                  <ul style={{ margin: 0, paddingLeft: 16, color: "var(--muted)", fontSize: 12 }}>
                    {opsBreaches.network.map((b, i) => <li key={`n_${i}`}>{b}</li>)}
                  </ul>
                  <div style={{ marginTop: 6, display: "flex", gap: 8, alignItems: "center" }}>
                    <button className="guide-toggle" onClick={() => setOpsAck((p) => ({ ...p, network_ack_ts: Date.now() }))}>Acknowledge</button>
                    {netAckActive && opsAck.network_ack_ts && (
                      <span style={{ fontSize: 11, color: "var(--muted)" }}>
                        acked @ {new Date(opsAck.network_ack_ts).toLocaleTimeString()} | expires in {Math.floor(netAckLeftSec / 60)}m {netAckLeftSec % 60}s
                      </span>
                    )}
                  </div>
                </div>
              )}
            </AsyncState>
          </div>
        </div>

        <div className="card">
          <div className="card-title self-help" data-help="Bakim, preflight ve baglanti testlerini buradan calistirirsin.">Ops Run Actions</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
            <button className="guide-toggle self-help" data-help="DB checkpoint/backup/disk kontrollerini tetikler." onClick={() => void runOpsAction("db_maintenance")} disabled={writeLocked || !!opsActionRunning}>
              {opsActionRunning === "db_maintenance" ? "Running..." : "Run DB Maintenance"}
            </button>
            <button className="guide-toggle self-help" data-help="Preflight raporunu yeniden olusturur." onClick={() => void runOpsAction("preflight_check")} disabled={writeLocked || !!opsActionRunning}>
              {opsActionRunning === "preflight_check" ? "Running..." : "Run Preflight"}
            </button>
            <button className="guide-toggle self-help" data-help="Incident bundle dosyasini export eder." onClick={() => void runOpsAction("incident_bundle")} disabled={writeLocked || !!opsActionRunning}>
              {opsActionRunning === "incident_bundle" ? "Running..." : "Export Incident Bundle"}
            </button>
            <button className="guide-toggle self-help" data-help="API/runtime/log dosya erisim kontrolunu calistirir." onClick={() => void runConnectivityCheck()} disabled={diagRunning}>
              {diagRunning ? "Checking..." : "Run Connectivity Check"}
            </button>
            <button className="guide-toggle self-help" data-help="Debug paylasimi icin bug bundle indirir." onClick={() => exportBugBundle()}>
              Export Bug Bundle
            </button>
          </div>
          {opsActionError && <div style={{ color: "var(--yellow)", fontSize: 12, marginBottom: 8 }}>{opsActionError}</div>}
          {diagError && <div style={{ color: "var(--yellow)", fontSize: 12, marginBottom: 8 }}>{diagError}</div>}
          {opsActionResult && (
            <div style={{ fontSize: 12, color: "var(--muted)" }}>
              result: action={opsActionResult.action} ok={String(opsActionResult.ok)} exit={opsActionResult.exit_code} duration={opsActionResult.duration_sec.toFixed(2)}s
              <pre
                style={{
                  marginTop: 6,
                  maxHeight: 120,
                  overflowY: "auto",
                  fontSize: 10,
                  padding: 6,
                  background: "var(--bg)",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                }}
              >
                {(opsActionResult.output || "").slice(-2000)}
              </pre>
            </div>
          )}
          {diagData && (
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 8 }}>
              <div style={{ marginBottom: 6 }}>
                connectivity status:{" "}
                <span className={`badge ${diagData.status === "ok" ? "badge-green" : "badge-yellow"}`}>
                  {(diagData.status || "unknown").toUpperCase()}
                </span>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>item</th>
                    <th>exists</th>
                    <th>readable</th>
                    <th>age_s</th>
                    <th>size</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(diagData.items || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td>{String(v.exists ?? false)}</td>
                      <td>{String(v.readable ?? false)}</td>
                      <td>{v.age_sec ?? "-"}</td>
                      <td>{v.size_bytes ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {(diagData.hints ?? []).length > 0 && (
                <div style={{ marginTop: 6 }}>
                  <div style={{ color: "var(--yellow)", fontSize: 11, marginBottom: 4 }}>hints</div>
                  <ul style={{ margin: 0, paddingLeft: 16 }}>
                    {(diagData.hints ?? []).map((h, i) => <li key={`hint_${i}`}>{h}</li>)}
                  </ul>
                </div>
              )}
            </div>
          )}
          <div style={{ marginTop: 8, fontSize: 11, color: "var(--muted)" }}>
            api error events in bundle: {apiErrors.length}
          </div>
        </div>

        <div className="card">
          <div className="card-title">Ops Trends (last ~1h)</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
            <div>
              <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>
                disk_free_gb ({trend.diskDir === "down" ? "worsening" : trend.diskDir === "up" ? "improving" : "flat"})
              </div>
              {renderSpark(trend.disk, Number(opsPoll.data?.thresholds?.disk_free_min_gb ?? 10), true)}
            </div>
            <div>
              <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>
                wal_mb ({trend.walDir === "up" ? "worsening" : trend.walDir === "down" ? "improving" : "flat"})
              </div>
              {renderSpark(trend.walMb, Number(opsPoll.data?.thresholds?.wal_warn_mb ?? 2048), false)}
            </div>
            <div>
              <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>
                reconnects_5m ({trend.reconnectDir === "up" ? "worsening" : trend.reconnectDir === "down" ? "improving" : "flat"})
              </div>
              {renderSpark(trend.reconnects, Number(opsPoll.data?.thresholds?.reconnect_warn_5m ?? 10), false)}
            </div>
            <div>
              <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>
                rate_usage_pct ({trend.usageDir === "up" ? "worsening" : trend.usageDir === "down" ? "improving" : "flat"})
              </div>
              {renderSpark(trend.usagePct, Number(opsPoll.data?.thresholds?.rate_warn_pct ?? 80), false)}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-title self-help" data-help="Backend supervisor durumu: restart sayisi, PID ve son event.">Backend Supervisor</div>
          <AsyncState loading={supervisorPoll.isLoading} error={supervisorPoll.error}>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
              <span className={`badge ${(supervisorPoll.data?.supervisor_running ?? false) ? "badge-green" : "badge-yellow"}`}>
                supervisor {(supervisorPoll.data?.supervisor_running ?? false) ? "RUNNING" : "UNKNOWN"}
              </span>
              <span className={`badge ${(supervisorPoll.data?.backend_runtime_present ?? false) ? "badge-green" : "badge-gray"}`}>
                runtime ptr {(supervisorPoll.data?.backend_runtime_present ?? false) ? "PRESENT" : "MISSING"}
              </span>
              <span className="badge badge-gray">
                restarts_1h={Number(supervisorPoll.data?.restarts_last_1h ?? 0)}
              </span>
              <button
                className="guide-toggle"
                onClick={() => {
                  if (!window.confirm("Restart dashboard backend now?")) return;
                  void runOpsAction("restart_dashboard_backend");
                }}
                disabled={writeLocked || !!opsActionRunning}
                title="Restart backend and wait for /api/health"
              >
                {opsActionRunning === "restart_dashboard_backend" ? "Restarting..." : "Restart Backend"}
              </button>
              <button
                className="guide-toggle"
                onClick={() => void runOpsAction("dashboard_logs_bundle")}
                disabled={writeLocked || !!opsActionRunning}
                title="Export backend/supervisor logs bundle"
              >
                {opsActionRunning === "dashboard_logs_bundle" ? "Exporting..." : "Export Logs Bundle"}
              </button>
            </div>
            <table>
              <tbody>
                <tr>
                  <td>backend</td>
                  <td>{supervisorPoll.data?.backend_host ?? "-"}:{supervisorPoll.data?.backend_port ?? "-"}</td>
                </tr>
                <tr>
                  <td>backend_pid</td>
                  <td>{supervisorPoll.data?.backend_pid ?? "-"}</td>
                </tr>
                <tr>
                  <td>runtime_age_sec</td>
                  <td>{supervisorPoll.data?.backend_runtime_age_sec ?? "-"}</td>
                </tr>
                <tr>
                  <td>last_event_age_sec</td>
                  <td>{supervisorPoll.data?.last_event_age_sec ?? "-"}</td>
                </tr>
              </tbody>
            </table>
            <pre
              style={{
                marginTop: 8,
                maxHeight: 120,
                overflowY: "auto",
                fontSize: 10,
                padding: 6,
                background: "var(--bg)",
                borderRadius: 4,
                border: "1px solid var(--border)",
              }}
            >
              {supervisorPoll.data?.last_event || "No supervisor events yet"}
            </pre>
          </AsyncState>
        </div>

        <div className="card">
          <div className="card-title">Incident Auto-Recovery Policy</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center", marginBottom: 8 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <input
                type="checkbox"
                checked={!!policyDraft.enabled}
                onChange={(e) => setPolicyDraft((p) => ({ ...p, enabled: e.target.checked }))}
                disabled={writeLocked}
              />
              enabled
            </label>
            <label style={{ fontSize: 12 }}>
              min_level
              <select
                value={policyDraft.min_level || "WARNING"}
                onChange={(e) => setPolicyDraft((p) => ({ ...p, min_level: e.target.value }))}
                disabled={writeLocked}
                style={{ marginLeft: 6 }}
              >
                <option value="INFO">INFO</option>
                <option value="WARNING">WARNING</option>
                <option value="ERROR">ERROR</option>
                <option value="CRITICAL">CRITICAL</option>
              </select>
            </label>
            <label style={{ fontSize: 12 }}>
              cooldown_sec
              <input
                type="number"
                min={30}
                step={30}
                value={Number(policyDraft.cooldown_sec || 900)}
                onChange={(e) => setPolicyDraft((p) => ({ ...p, cooldown_sec: Number(e.target.value) || 900 }))}
                disabled={writeLocked}
                style={{ marginLeft: 6, width: 100 }}
              />
            </label>
            <button className="guide-toggle" onClick={() => void savePolicy()} disabled={writeLocked || policySaving}>
              {policySaving ? "Saving..." : "Save Policy"}
            </button>
            <button className="guide-toggle" onClick={() => void runAutoOnce()} disabled={writeLocked}>
              Run Auto Once
            </button>
          </div>
          {policyMsg && <div style={{ fontSize: 12, color: policyMsg.startsWith("Save failed") ? "var(--yellow)" : "var(--green)" }}>{policyMsg}</div>}
          {autoRunMsg && <div style={{ fontSize: 12, color: autoRunMsg.startsWith("Auto-run failed") ? "var(--yellow)" : "var(--muted)" }}>{autoRunMsg}</div>}
          {!!policyDraft.last_run_ts_by_type && Object.keys(policyDraft.last_run_ts_by_type).length > 0 && (
            <pre
              style={{
                marginTop: 8,
                maxHeight: 120,
                overflowY: "auto",
                fontSize: 10,
                padding: 6,
                background: "var(--bg)",
                borderRadius: 4,
                border: "1px solid var(--border)",
              }}
            >
              {Object.entries(policyDraft.last_run_ts_by_type)
                .map(([k, v]) => `${k}: ${new Date(v * 1000).toLocaleString()}`)
                .join("\n")}
            </pre>
          )}
        </div>

        <div className="card">
          <div className="card-title">Incident Drill-down</div>
          <AsyncState loading={incidentPoll.isLoading} error={incidentPoll.error} isEmpty={(incidentPoll.data ?? []).length === 0} emptyText="No incidents">
            <table>
              <thead>
                <tr>
                  <th>time</th>
                  <th>type</th>
                  <th>level</th>
                  <th>status</th>
                  <th>session</th>
                  <th>actions</th>
                </tr>
              </thead>
              <tbody>
                {(incidentPoll.data ?? []).slice(0, 12).map((inc) => (
                  <tr key={inc.incident_id}>
                    <td style={{ color: "var(--muted)" }}>{inc.ts ? new Date((inc.ts || 0) * 1000).toLocaleString() : "-"}</td>
                    <td>{inc.type}</td>
                    <td>{inc.level}</td>
                    <td>{inc.status}</td>
                    <td style={{ color: "var(--muted)", fontSize: 11 }}>{inc.session_id || "-"}</td>
                    <td style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      <button
                        className="guide-toggle"
                        onClick={() => {
                          setSelectedIncident(inc);
                          if (inc.session_id) void loadSession(inc.session_id);
                        }}
                      >
                        Open Session
                      </button>
                      <button
                        className="guide-toggle"
                        onClick={() => inc.session_id && setCompareAId(inc.session_id)}
                        disabled={!inc.session_id}
                      >
                        Set A
                      </button>
                      <button
                        className="guide-toggle"
                        onClick={() => inc.session_id && setCompareBId(inc.session_id)}
                        disabled={!inc.session_id}
                      >
                        Set B
                      </button>
                      <button className="guide-toggle" onClick={() => void runIncidentNow(inc.incident_id)} disabled={writeLocked}>
                        Run Now
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </AsyncState>
          {selectedIncident && (
            <div style={{ marginTop: 8, fontSize: 12, color: "var(--muted)" }}>
              Selected incident: {selectedIncident.type} ({selectedIncident.incident_id})
            </div>
          )}
          {sessionError && <div style={{ marginTop: 8, color: "var(--yellow)", fontSize: 12 }}>{sessionError}</div>}
          {sessionLoading && <div style={{ marginTop: 8, color: "var(--muted)", fontSize: 12 }}>Loading session details...</div>}
          {sessionDetail && (
            <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", fontSize: 12 }}>
                <span className={`badge ${sessionDetail.ok ? "badge-green" : "badge-yellow"}`}>session {sessionDetail.session_id}</span>
                <span>ok={String(sessionDetail.ok)}</span>
                <span>duration={sessionDetail.duration_sec.toFixed(2)}s</span>
                <span>failed={sessionDetail.failed_action ?? "-"}</span>
                <button className="guide-toggle" onClick={() => setCompareAId(sessionDetail.session_id)}>Set A</button>
                <button className="guide-toggle" onClick={() => setCompareBId(sessionDetail.session_id)}>Set B</button>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>step</th>
                    <th>ok</th>
                    <th>exit</th>
                    <th>duration</th>
                  </tr>
                </thead>
                <tbody>
                  {sessionDetail.steps.map((s, i) => (
                    <tr key={`${s.action}-${i}`}>
                      <td>{s.action}</td>
                      <td>{String(s.ok)}</td>
                      <td>{s.exit_code}</td>
                      <td>{s.duration_sec.toFixed(2)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <pre
                style={{
                  maxHeight: 160,
                  overflowY: "auto",
                  fontSize: 10,
                  padding: 6,
                  background: "var(--bg)",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                }}
              >
                {(sessionTimeline ?? [])
                  .slice(0, 40)
                  .map((e) => `${e.ts ? new Date((e.ts || 0) * 1000).toLocaleTimeString() : "-"} ${e.kind}: ${e.title}${e.detail ? ` :: ${e.detail}` : ""}`)
                  .join("\n") || "No timeline events"}
              </pre>
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-title">Session Compare (A/B)</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 8 }}>
            <span style={{ fontSize: 12, color: "var(--muted)" }}>A: {compareAId || "-"}</span>
            <span style={{ fontSize: 12, color: "var(--muted)" }}>B: {compareBId || "-"}</span>
            <button className="guide-toggle" onClick={() => void loadCompare()} disabled={compareLoading || !compareAId || !compareBId}>
              {compareLoading ? "Comparing..." : "Compare"}
            </button>
            <button
              className="guide-toggle"
              onClick={() => {
                setCompareAId("");
                setCompareBId("");
                setCompareA(null);
                setCompareB(null);
                setCompareError("");
              }}
            >
              Clear
            </button>
            <label style={{ color: "var(--muted)", fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}>
              <input type="checkbox" checked={compareOnlyFailed} onChange={(e) => setCompareOnlyFailed(e.target.checked)} />
              only failed/changed
            </label>
          </div>
          {compareError && <div style={{ color: "var(--yellow)", fontSize: 12 }}>{compareError}</div>}
          {compareA && compareB && (
            <>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
                <span>A ok={String(compareA.ok)} failed={compareA.failed_action ?? "-"}</span>
                <span>B ok={String(compareB.ok)} failed={compareB.failed_action ?? "-"}</span>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>step</th>
                    <th>A</th>
                    <th>B</th>
                    <th>changed</th>
                  </tr>
                </thead>
                <tbody>
                  {compareRows
                    .filter((r) => (compareOnlyFailed ? (r.changed || r.a === "fail" || r.b === "fail") : true))
                    .map((r) => (
                      <tr key={r.action}>
                        <td>{r.action}</td>
                        <td>{r.a}</td>
                        <td>{r.b}</td>
                        <td>{r.changed ? "yes" : "no"}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </>
          )}
        </div>

        <div className="card">
          <div className="card-title">Incident Session</div>
          <div style={{ marginBottom: 6 }}>
            <span className={`badge ${incidentSessionId ? "badge-yellow" : "badge-gray"}`}>
              {incidentSessionId || "no active incident"}
            </span>
          </div>
          <pre
            style={{
              maxHeight: 180,
              overflowY: "auto",
              fontSize: 10,
              padding: 6,
              background: "var(--bg)",
              borderRadius: 4,
              border: "1px solid var(--border)",
            }}
          >
            {incidentTimeline.length
              ? incidentTimeline.slice(-20).map((e) => `${new Date(e.ts).toLocaleTimeString()} ${e.action}${e.detail ? ` :: ${e.detail}` : ""}`).join("\n")
              : "No incident timeline events"}
          </pre>
        </div>
      </AsyncState>
    </div>
  );
}
