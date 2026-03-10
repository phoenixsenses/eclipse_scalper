import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { useBackendStatus } from "../context/BackendStatusContext";
import PageGuide from "../components/PageGuide";
import AsyncState from "../components/AsyncState";
import type { ControlActionInfo, ControlActionResult } from "../api/types";

type StepState = "idle" | "checking" | "recover_cmd" | "recheck" | "ready" | "down";

interface CheckRow {
  name: string;
  ok: boolean;
  detail: string;
}

interface CommandRow {
  label: string;
  tr: string;
  command: string;
}

interface RecoveryHistoryEntry {
  ts: number;
  kind: "check" | "action";
  mode: string;
  ok: boolean;
  summary: string;
  detail: string;
}

const HISTORY_KEY = "eclipse.recovery.history.v1";

const COMMANDS: CommandRow[] = [
  {
    label: "Start backend supervisor",
    tr: "Backend supervisor başlat",
    command: "powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\run_dashboard_backend_supervisor.ps1",
  },
  {
    label: "Start full dashboard",
    tr: "Tüm dashboard stack'ini başlat",
    command: "powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\run_dashboard.ps1",
  },
  {
    label: "Check backend health",
    tr: "Backend health kontrol",
    command: "Invoke-WebRequest http://127.0.0.1:8765/api/health -UseBasicParsing",
  },
];

export default function Recovery() {
  const navigate = useNavigate();
  const backend = useBackendStatus();
  const [step, setStep] = useState<StepState>("idle");
  const [copyHint, setCopyHint] = useState("");
  const [rows, setRows] = useState<CheckRow[]>([]);
  const [lastTs, setLastTs] = useState<string>("");
  const [error, setError] = useState<Error | null>(null);
  const [checking, setChecking] = useState(false);
  const [autoRecheck, setAutoRecheck] = useState(false);
  const [actions, setActions] = useState<ControlActionInfo[]>([]);
  const [selectedAction, setSelectedAction] = useState<string>("");
  const [runResult, setRunResult] = useState<ControlActionResult | null>(null);
  const [runErr, setRunErr] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState<RecoveryHistoryEntry[]>(() => {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(
        (x): x is RecoveryHistoryEntry =>
          x &&
          typeof x.ts === "number" &&
          (x.kind === "check" || x.kind === "action") &&
          typeof x.mode === "string" &&
          typeof x.ok === "boolean" &&
          typeof x.summary === "string" &&
          typeof x.detail === "string",
      );
    } catch {
      return [];
    }
  });

  const score = useMemo(() => rows.filter((r) => r.ok).length, [rows]);

  function pushHistory(entry: RecoveryHistoryEntry) {
    setHistory((prev) => {
      const next = [...prev, entry].slice(-20);
      try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
      } catch {
        // best effort
      }
      return next;
    });
  }

  useEffect(() => {
    let active = true;
    if (!backend.backendUp) return;
    (async () => {
      try {
        const list = await api.debugActions();
        if (!active) return;
        setActions(list ?? []);
        if (!selectedAction && (list ?? []).length > 0) {
          setSelectedAction(String(list[0].action || ""));
        }
      } catch {
        if (!active) return;
        setActions([]);
      }
    })();
    return () => {
      active = false;
    };
  }, [backend.backendUp, selectedAction]);

  async function copyCmd(v: string) {
    try {
      await navigator.clipboard.writeText(v);
      setCopyHint("Command copied");
      window.setTimeout(() => setCopyHint(""), 1500);
    } catch {
      setCopyHint("Copy failed");
      window.setTimeout(() => setCopyHint(""), 1500);
    }
  }

  async function runChecks(mode: "checking" | "recheck") {
    setChecking(true);
    setError(null);
    setStep(mode);
    const next: CheckRow[] = [];

    try {
      const h = await api.health();
      next.push({
        name: "API /health",
        ok: String(h.status || "").toLowerCase() === "ok",
        detail: `status=${h.status ?? "-"}`,
      });
    } catch (e) {
      next.push({
        name: "API /health",
        ok: false,
        detail: e instanceof Error ? e.message : String(e),
      });
    }

    try {
      const r = await api.runtime();
      const status = String(r?.data_freshness?.status ?? "-");
      next.push({
        name: "API /runtime",
        ok: true,
        detail: `freshness=${status}`,
      });
    } catch (e) {
      next.push({
        name: "API /runtime",
        ok: false,
        detail: e instanceof Error ? e.message : String(e),
      });
    }

    try {
      const ops = await api.opsHealth();
      const data = String(ops?.status?.data_integrity ?? "-");
      const net = String(ops?.status?.network ?? "-");
      next.push({
        name: "API /ops/health",
        ok: data !== "-" && net !== "-",
        detail: `data_integrity=${data} network=${net}`,
      });
    } catch (e) {
      next.push({
        name: "API /ops/health",
        ok: false,
        detail: e instanceof Error ? e.message : String(e),
      });
    }

    try {
      const sup = await api.supervisorStatus();
      const supervisorRunning = Boolean(sup?.supervisor_running);
      const runtimePresent = Boolean(sup?.backend_runtime_present);
      next.push({
        name: "API /ops/supervisor",
        ok: supervisorRunning || runtimePresent,
        detail:
          `supervisor_running=${supervisorRunning ? "1" : "0"} ` +
          `runtime_present=${runtimePresent ? "1" : "0"} ` +
          `restarts_1h=${sup?.restarts_last_1h ?? 0}`,
      });
    } catch (e) {
      next.push({
        name: "API /ops/supervisor",
        ok: false,
        detail: e instanceof Error ? e.message : String(e),
      });
    }

    setRows(next);
    setLastTs(new Date().toLocaleTimeString());
    const okAll = next.length > 0 && next.every((r) => r.ok);
    pushHistory({
      ts: Date.now(),
      kind: "check",
      mode,
      ok: okAll,
      summary: `${next.filter((r) => r.ok).length}/${next.length} checks ok`,
      detail: next.map((r) => `${r.name}=${r.ok ? "OK" : "FAIL"}`).join(" | "),
    });
    setStep(okAll ? "ready" : "recover_cmd");
    setChecking(false);
  }

  function start() {
    void runChecks("checking");
  }

  function recheck() {
    void runChecks("recheck");
  }

  async function runSelectedAction() {
    if (!selectedAction || !backend.backendUp) return;
    setRunning(true);
    setRunErr("");
    try {
      const out = await api.runDebugAction(selectedAction);
      setRunResult(out);
      pushHistory({
        ts: Date.now(),
        kind: "action",
        mode: selectedAction,
        ok: Boolean(out.ok),
        summary: `action=${selectedAction} exit=${out.exit_code}`,
        detail: String(out.output || "").slice(-240),
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setRunErr(msg);
      pushHistory({
        ts: Date.now(),
        kind: "action",
        mode: selectedAction,
        ok: false,
        summary: `action=${selectedAction} failed`,
        detail: msg,
      });
    } finally {
      setRunning(false);
    }
  }

  async function runSelectedActionAndRecheck() {
    if (!selectedAction || !backend.backendUp) return;
    await runSelectedAction();
    await runChecks("recheck");
  }

  useEffect(() => {
    if (!autoRecheck) return;
    if (step === "ready") return;
    const id = window.setInterval(() => {
      if (!checking) {
        void runChecks("recheck");
      }
    }, 5000);
    return () => window.clearInterval(id);
  }, [autoRecheck, checking, step]);

  const stepLabel =
    step === "idle" ? "idle" :
    step === "checking" ? "checking" :
    step === "recover_cmd" ? "recover_cmd" :
    step === "recheck" ? "recheck" :
    step === "ready" ? "ready" : "down";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <PageGuide
        icon="🛟"
        titleTr="Kurtarma Sihirbazi"
        titleEn="Recovery Wizard"
        subtitleTr="Backend sorunu oldugunda adim adim toparlama akisi."
        subtitleEn="Step-by-step backend recovery flow when dashboard degrades."
        items={[
          { icon: "1", titleTr: "Kontrol", titleEn: "Check", descTr: "Health/runtime/ops endpointlerini test et.", descEn: "Test health/runtime/ops endpoints." },
          { icon: "2", titleTr: "Komut", titleEn: "Command", descTr: "Supervisor veya tam dashboard komutunu calistir.", descEn: "Run supervisor or full dashboard command." },
          { icon: "3", titleTr: "Dogrula", titleEn: "Verify", descTr: "Recheck ile READY durumuna gec.", descEn: "Recheck and move to READY state." },
        ]}
      />

      <div className="card">
        <div className="card-title">Wizard State</div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
          <span className={`badge ${step === "ready" ? "badge-green" : step === "recover_cmd" ? "badge-yellow" : "badge-blue"}`}>
            {stepLabel.toUpperCase()}
          </span>
          <span style={{ color: "var(--muted)", fontSize: 12 }}>
            backend={backend.backendUp ? "UP" : "DOWN"} message={backend.backendMessage}
          </span>
          {lastTs ? <span style={{ color: "var(--muted)", fontSize: 12 }}>last_check={lastTs}</span> : null}
          <span style={{ color: "var(--muted)", fontSize: 12 }}>checks_ok={score}/{rows.length}</span>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Step 1 - Connectivity Check</div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
          <button className="guide-toggle self-help" data-help="Run full connectivity checks now." onClick={start} disabled={checking}>
            Start Check
          </button>
          <button className="guide-toggle self-help" data-help="Run checks again now." onClick={recheck} disabled={checking}>
            Recheck
          </button>
          <button
            className="guide-toggle self-help"
            data-help="When ON, Recheck runs every 5 seconds until READY."
            onClick={() => setAutoRecheck((v) => !v)}
            disabled={step === "ready"}
          >
            {autoRecheck ? "Auto Recheck ON (5s)" : "Auto Recheck OFF"}
          </button>
        </div>
        <AsyncState loading={checking} error={error} isEmpty={rows.length === 0} emptyText="No checks run yet">
          <table>
            <thead>
              <tr>
                <th>Check</th>
                <th>Status</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name}>
                  <td>{r.name}</td>
                  <td>
                    <span className={`badge ${r.ok ? "badge-green" : "badge-red"}`}>{r.ok ? "OK" : "FAIL"}</span>
                  </td>
                  <td style={{ color: "var(--muted)", fontSize: 12 }}>{r.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </AsyncState>
      </div>

      <div className="card">
        <div className="card-title">Step 2 - Recovery Commands</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 8 }}>
          {COMMANDS.map((c) => (
            <div key={c.command} className="recovery-cmd-card">
              <div className="recovery-cmd-head">
                <span>{c.label}</span>
                <span style={{ color: "var(--muted)", fontSize: 10 }}>{c.tr}</span>
              </div>
              <code className="recovery-code">{c.command}</code>
              <button className="guide-toggle self-help" data-help="Copy command to clipboard." onClick={() => void copyCmd(c.command)}>
                Copy
              </button>
            </div>
          ))}
        </div>
        {copyHint ? <div className="recovery-copy-hint">{copyHint}</div> : null}
        <div style={{ marginTop: 10, borderTop: "1px dashed var(--border)", paddingTop: 10 }}>
          <div style={{ color: "var(--muted)", fontSize: 11, marginBottom: 6 }}>
            Optional: run whitelisted debug action directly
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <select
              value={selectedAction}
              onChange={(e) => setSelectedAction(e.target.value)}
              style={{
                padding: "4px 8px",
                borderRadius: 4,
                border: "1px solid var(--border)",
                background: "var(--bg)",
                color: "var(--text)",
                minWidth: 220,
              }}
            >
              {actions.length === 0 ? <option value="">No actions</option> : null}
              {actions.map((a) => (
                <option key={a.action} value={a.action}>
                  {a.action}
                </option>
              ))}
            </select>
            <button
              className="guide-toggle self-help"
              data-help="Execute selected backend debug action."
              onClick={() => void runSelectedAction()}
              disabled={!backend.backendUp || !selectedAction || running}
              title="Runs /api/debug/run action"
            >
              {running ? "Running..." : "Run Selected Command"}
            </button>
            <button
              className="guide-toggle self-help"
              data-help="Execute action, then run recheck automatically."
              onClick={() => void runSelectedActionAndRecheck()}
              disabled={!backend.backendUp || !selectedAction || running || checking}
              title="Run selected action, then execute recheck"
            >
              {running || checking ? "Working..." : "Run + Recheck"}
            </button>
          </div>
          {runErr ? <div style={{ color: "var(--red)", fontSize: 12, marginTop: 6 }}>{runErr}</div> : null}
          {runResult ? (
            <div style={{ marginTop: 6 }}>
              <span className={`badge ${runResult.ok ? "badge-green" : "badge-red"}`}>
                {runResult.ok ? "SUCCESS" : "FAILED"}
              </span>
              <span style={{ color: "var(--muted)", fontSize: 11, marginLeft: 8 }}>
                exit={runResult.exit_code} duration={runResult.duration_sec?.toFixed?.(2) ?? runResult.duration_sec}s
              </span>
              <pre
                style={{
                  marginTop: 6,
                  maxHeight: 140,
                  overflowY: "auto",
                  fontSize: 11,
                  padding: 6,
                  background: "var(--bg)",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                }}
              >
                {String(runResult.output || "").slice(-2000) || "(no output)"}
              </pre>
            </div>
          ) : null}
        </div>
      </div>

      <div className="card">
        <div className="card-title">Step 3 - Continue</div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button className="guide-toggle" onClick={() => navigate("/tower")} disabled={step !== "ready"}>
            Open Control Tower
          </button>
          <button className="guide-toggle" onClick={() => navigate("/logs")} disabled={step !== "ready"}>
            Open Logs
          </button>
          <button className="guide-toggle" onClick={() => navigate("/debug?auto=triage")} disabled={step !== "ready"}>
            Run Debug Triage
          </button>
        </div>
        {step !== "ready" ? (
          <div style={{ color: "var(--yellow)", fontSize: 12, marginTop: 8 }}>
            Status not READY yet. Run commands, then click Recheck.
          </div>
        ) : (
          <div style={{ color: "var(--green)", fontSize: 12, marginTop: 8 }}>
            READY. Continue to Tower/Logs/Debug.
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">Recovery History (last 20)</div>
        <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
          <button
            className="guide-toggle"
            onClick={() => {
              setHistory([]);
              try {
                localStorage.removeItem(HISTORY_KEY);
              } catch {
                // best effort
              }
            }}
          >
            Clear History
          </button>
        </div>
        <AsyncState loading={false} error={null} isEmpty={history.length === 0} emptyText="No history yet">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Kind</th>
                <th>Mode</th>
                <th>Status</th>
                <th>Summary</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {[...history].reverse().map((h, idx) => (
                <tr key={`${h.ts}_${idx}`}>
                  <td style={{ color: "var(--muted)", fontSize: 12 }}>{new Date(h.ts).toLocaleTimeString()}</td>
                  <td>{h.kind}</td>
                  <td>{h.mode}</td>
                  <td>
                    <span className={`badge ${h.ok ? "badge-green" : "badge-red"}`}>{h.ok ? "OK" : "FAIL"}</span>
                  </td>
                  <td style={{ fontSize: 12 }}>{h.summary}</td>
                  <td style={{ color: "var(--muted)", fontSize: 11 }}>{h.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </AsyncState>
      </div>
    </div>
  );
}
