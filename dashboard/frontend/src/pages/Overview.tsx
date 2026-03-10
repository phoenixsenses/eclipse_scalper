import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { GateSymbol, IncidentInboxItem, OverviewResponse, RegimeEvent, RuntimeStatus } from "../api/types";
import AsyncState from "../components/AsyncState";
import DegradedBanner, { type DegradedMode } from "../components/DegradedBanner";
import PageGuide from "../components/PageGuide";
import { ResearchEventsSummary } from "../components/ResearchEventsPanel";
import TermTip from "../components/TermTip";
import { usePoll } from "../hooks/usePoll";

function Stat({ label, value, color }: { label: string; value: React.ReactNode; color?: string }) {
  return (
    <div className="card" style={{ minWidth: 120 }}>
      <div className="stat-value" style={{ color: color ?? "var(--text)" }}>{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function StatusBadge({ value }: { value: boolean | null }) {
  if (value === null) return <span className="badge badge-gray">-</span>;
  return value
    ? <span className="badge badge-green">LIVE</span>
    : <span className="badge badge-yellow">PAPER</span>;
}

function RegimeBadge({ regime }: { regime?: string | null }) {
  if (!regime) return <span className="badge badge-gray">-</span>;
  const normalized = regime.toLowerCase();
  const cls =
    normalized === "trending" ? "badge-green" :
    normalized === "volatile" ? "badge-red" :
    normalized === "ranging" ? "badge-blue" : "badge-gray";
  return <span className={`badge ${cls}`}>{regime}</span>;
}

type FreshnessStatus = "LIVE" | "DEGRADED" | "STALE";

const STATUS_COLOR: Record<FreshnessStatus, string> = {
  LIVE: "var(--green)",
  DEGRADED: "var(--yellow)",
  STALE: "var(--red)",
};

function CollectorDot({ status }: { status: FreshnessStatus | null }) {
  const s = status ?? "STALE";
  return (
    <span style={{ color: STATUS_COLOR[s], fontWeight: 700, letterSpacing: "0.04em" }}>
      * {s}
    </span>
  );
}

function MetricCell({
  label,
  value,
  warn,
  alert,
}: {
  label: string;
  value: string;
  warn?: boolean;
  alert?: boolean;
}) {
  const color = alert ? "var(--red)" : warn ? "var(--yellow)" : "var(--text)";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 100 }}>
      <span style={{ fontSize: 18, fontWeight: 700, color }}>{value}</span>
      <span style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {label}
      </span>
    </div>
  );
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function fitnessBadge(status: string, connected: boolean | null) {
  const normalized = status.toLowerCase();
  if (normalized === "ok") return <span className="badge badge-green">PASS</span>;
  if (normalized === "warning") return <span className="badge badge-yellow">WARN</span>;
  if (normalized === "degraded") return <span className="badge badge-red">FAIL</span>;
  if (connected === false) return <span className="badge badge-red">DISCONNECTED</span>;
  return <span className="badge badge-gray">UNKNOWN</span>;
}

function blockerRecommendation(reason: string): { action: string; detail: string } {
  const r = reason.toLowerCase();
  if (r.includes("no_match")) {
    return {
      action: "Logs > query=no_match_detail",
      detail: "Eşiklerden hangisinin başarısız olduğunu kontrol et (imb/int/spr).",
    };
  }
  if (r.includes("regime")) {
    return {
      action: "Overview > Recent Regime + paper_trading.log",
      detail: "Rejim uyumsuzluğu veya UNKNOWN durumunu kontrol et.",
    };
  }
  if (r.includes("risk") || r.includes("kill") || r.includes("circuit")) {
    return {
      action: "Trades > Stability/Quality + risk logs",
      detail: "Risk guard ve cooldown bloklarini incele.",
    };
  }
  if (r.includes("data") || r.includes("stale")) {
    return {
      action: "Overview Runtime + preflight",
      detail: "Veri tazeliği ve collector durumunu doğrula.",
    };
  }
  return {
    action: "Debug > Guided Session",
    detail: "Temel tanilama adimlarini sirayla calistir.",
  };
}

function RuntimePanel({ rt, stale }: { rt: RuntimeStatus; stale: boolean }) {
  const freshness = rt.data_freshness ?? {};
  const collector = rt.collector ?? {};
  const db = rt.database ?? {};

  const statusRaw = freshness.status ?? "STALE";
  const status: FreshnessStatus =
    statusRaw === "LIVE" || statusRaw === "DEGRADED" || statusRaw === "STALE" ? statusRaw : "STALE";
  const ageSec = freshness.seconds_since_last_trade ?? null;

  const tradeSec = collector.trades_per_sec_60s != null ? collector.trades_per_sec_60s.toFixed(1) : "-";
  const markSec = collector.mark_per_sec_60s != null ? collector.mark_per_sec_60s.toFixed(1) : "-";
  const liqSec = collector.liquidations_per_sec_60s != null ? collector.liquidations_per_sec_60s.toFixed(2) : "-";
  const uptimeStr = collector.uptime_sec != null
    ? (collector.uptime_sec >= 3600
      ? (collector.uptime_sec / 3600).toFixed(1) + "h"
      : Math.round(collector.uptime_sec / 60) + "m")
    : "-";
  const dbMB = db.size_bytes != null ? (db.size_bytes / 1_048_576).toFixed(1) + " MB" : "-";
  const growthKB = db.growth_bytes_5min != null ? (db.growth_bytes_5min / 1024).toFixed(0) + " KB" : "-";
  const ageStr = ageSec != null ? ageSec.toFixed(0) + "s" : "-";

  const borderColor =
    status === "LIVE" ? "var(--green)" :
    status === "DEGRADED" ? "var(--yellow)" : "var(--red)";

  return (
    <div className="card" style={{ borderLeft: `3px solid ${borderColor}`, opacity: stale ? 0.6 : 1 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
        <span className="card-title" style={{ margin: 0 }}>Runtime Status</span>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <CollectorDot status={status} />
          {stale && <span style={{ fontSize: 11, color: "var(--muted)" }}>connecting...</span>}
        </div>
      </div>

      <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
        <MetricCell label="Trades / sec" value={tradeSec} warn={status === "DEGRADED"} alert={status === "STALE"} />
        <MetricCell label="Mark / sec" value={markSec} />
        <MetricCell label="Liqs / sec" value={liqSec} />
        <MetricCell label="Last trade age" value={ageStr} warn={status === "DEGRADED"} alert={status === "STALE"} />
        <MetricCell label="DB size" value={dbMB} />
        <MetricCell label="DB growth / 5m" value={growthKB} />
        <MetricCell label="Collector uptime" value={uptimeStr} />
        <MetricCell label="Collector proc" value={collector.alive ? "alive" : "dead"} alert={!collector.alive} />
      </div>

      {freshness.last_trade_ts && (
        <div style={{ marginTop: 10, fontSize: 11, color: "var(--muted)" }}>
          Last trade: {freshness.last_trade_ts}
          {collector.last_log_ts && <span style={{ marginLeft: 16 }}>Log updated: {collector.last_log_ts}</span>}
        </div>
      )}
    </div>
  );
}

export default function Overview() {
  const navigate = useNavigate();
  const fetchOverview = useCallback((signal: AbortSignal) => api.overview(signal), []);
  const fetchRuntime = useCallback((signal: AbortSignal) => api.runtime(signal), []);
  const fetchIncidents = useCallback((signal: AbortSignal) => api.debugIncidents(30, signal), []);

  const overviewPoll = usePoll<OverviewResponse>({
    fetcher: fetchOverview,
    pollKey: "api:/overview",
    intervalMs: 10_000,
    staleAfterMs: 30_000,
  });

  const runtimePoll = usePoll<RuntimeStatus>({
    fetcher: fetchRuntime,
    pollKey: "api:/runtime",
    intervalMs: 2_000,
    staleAfterMs: 8_000,
    retryInitialMs: 1_000,
    retryMaxMs: 8_000,
  });
  const incidentPoll = usePoll<IncidentInboxItem[]>({
    fetcher: fetchIncidents,
    pollKey: "api:/debug/incidents",
    intervalMs: 15_000,
    staleAfterMs: 45_000,
  });

  const [bannerMode, setBannerMode] = useState<DegradedMode>("ok");
  const [bannerMessage, setBannerMessage] = useState<string | undefined>(undefined);
  const prevRuntimeProblem = useRef(false);
  const [lastTriage, setLastTriage] = useState<{ ok: boolean; ts: string; failedAction?: string } | null>(() => {
    try {
      const raw = localStorage.getItem("eclipse.debug.last_triage.v1");
      if (!raw) return null;
      const parsed = JSON.parse(raw) as { ok?: boolean; ts?: string; failedAction?: string };
      if (typeof parsed.ok !== "boolean" || typeof parsed.ts !== "string") return null;
      return { ok: parsed.ok, ts: parsed.ts, failedAction: parsed.failedAction };
    } catch {
      return null;
    }
  });

  useEffect(() => {
    const syncLastTriage = () => {
      try {
        const raw = localStorage.getItem("eclipse.debug.last_triage.v1");
        if (!raw) return;
        const parsed = JSON.parse(raw) as { ok?: boolean; ts?: string; failedAction?: string };
        if (typeof parsed.ok !== "boolean" || typeof parsed.ts !== "string") return;
        setLastTriage({ ok: parsed.ok, ts: parsed.ts, failedAction: parsed.failedAction });
      } catch {
        // ignore
      }
    };
    const timer = window.setInterval(syncLastTriage, 30_000);
    window.addEventListener("focus", syncLastTriage);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", syncLastTriage);
    };
  }, []);

  useEffect(() => {
    const hasProblem = Boolean(runtimePoll.error) || runtimePoll.isStale;
    if (hasProblem) {
      setBannerMode(runtimePoll.error ? "down" : "degraded");
      setBannerMessage(runtimePoll.error ? runtimePoll.error.message : "Runtime poll is stale");
    } else if (prevRuntimeProblem.current && !hasProblem) {
      setBannerMode("recovered");
      setBannerMessage("Runtime polling recovered");
      const id = window.setTimeout(() => {
        setBannerMode("ok");
        setBannerMessage(undefined);
      }, 3000);
      return () => window.clearTimeout(id);
    } else {
      setBannerMode("ok");
      setBannerMessage(undefined);
    }
    prevRuntimeProblem.current = hasProblem;
    return undefined;
  }, [runtimePoll.error, runtimePoll.isStale]);

  const data = overviewPoll.data;
  const sb = data?.scoreboard ?? {};
  const gates = data?.gates ?? {};
  const symbols: GateSymbol[] = gates.symbols ?? [];
  const regimes: RegimeEvent[] = data?.recent_regimes ?? [];
  const preflight = data?.preflight ?? {};
  const reliability = data?.reliability ?? {};
  const healthOverall = asRecord(data?.health_overall);
  const research = asRecord(data?.research_events);
  const fitnessStatus = String(healthOverall.data_research_fitness_status || "");
  const fitnessConnected =
    typeof healthOverall.data_research_fitness_connected === "boolean"
      ? healthOverall.data_research_fitness_connected
      : null;
  const fitnessDetail = String(healthOverall.data_research_fitness_detail || "");
  const topBlockers: Array<{ reason: string; count: number }> = useMemo(() => {
    const raw = sb.blocked_by_reason ?? {};
    return Object.entries(raw)
      .map(([reason, count]) => ({
        reason,
        count: typeof count === "number" ? count : Number(count) || 0,
      }))
      .filter((x) => x.count > 0)
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);
  }, [sb.blocked_by_reason]);
  const topBlocker = topBlockers.length > 0 ? topBlockers[0] : null;
  const topBlockerGuide = topBlocker ? blockerRecommendation(topBlocker.reason) : null;
  const recentIncidentRibbon = useMemo(() => {
    const rows = Array.isArray(incidentPoll.data) ? incidentPoll.data : [];
    const cutoff = Date.now() / 1000 - 15 * 60;
    return rows
      .filter((x) => (x.ts ?? 0) >= cutoff)
      .slice(0, 6);
  }, [incidentPoll.data]);

  function openLogPack(pack: "No Match" | "Regime" | "Shutdown" | "Timeout") {
    navigate(`/logs?pack=${encodeURIComponent(pack)}`);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <PageGuide
        icon="🧭"
        titleTr="Sistem Durumu"
        titleEn="System Overview"
        subtitleTr="Önce bu sayfadan sağlık, veri tazeliği ve engel sebeplerini kontrol et."
        subtitleEn="Start here to verify health, data freshness, and entry blockers."
        items={[
          {
            icon: "📡",
            titleTr: "Canlılık",
            titleEn: "Liveness",
            descTr: "LIVE/DEGRADED/STALE etiketi veri akışının kalitesini gösterir.",
            descEn: "LIVE/DEGRADED/STALE shows feed quality and collector health.",
          },
          {
            icon: "🚧",
            titleTr: "Engeller",
            titleEn: "Blockers",
            descTr: "Top Blockers tablosu işlemlerin neden açılmadığını özetler.",
            descEn: "Top Blockers summarizes why entries are getting blocked.",
          },
          {
            icon: "🌦️",
            titleTr: "Rejim",
            titleEn: "Regime",
            descTr: "Rejim geçişleri sinyal davranışının piyasa modunu açıklar.",
            descEn: "Regime transitions explain market mode behind signal behavior.",
          },
        ]}
      />

      {lastTriage && (
        <div
          className="card"
          style={{
            borderLeft: `3px solid ${lastTriage.ok ? "var(--green)" : "var(--yellow)"}`,
          }}
        >
          <div className="card-title self-help" data-help="Son otomatik tanılama çalışmasının PASS/FAIL özeti.">Last Triage</div>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
            <span className={`badge ${lastTriage.ok ? "badge-green" : "badge-yellow"}`}>
              {lastTriage.ok ? "PASS" : "FAIL"}
            </span>
            <span style={{ color: "var(--muted)" }}>{lastTriage.ts}</span>
            {lastTriage.failedAction && <span>failed={lastTriage.failedAction}</span>}
            <button
              className="self-help"
              data-help="Debug sayfasina gidip adim adim root-cause tanisi calistir."
              onClick={() => navigate("/debug")}
              style={{
                marginLeft: "auto",
                padding: "3px 8px",
                borderRadius: 4,
                border: "1px solid var(--border)",
                background: "transparent",
                color: "var(--muted)",
                cursor: "pointer",
                fontSize: 11,
              }}
            >
              Open Debug
            </button>
          </div>
        </div>
      )}

      <DegradedBanner mode={bannerMode} message={bannerMessage} />

      <AsyncState loading={overviewPoll.isLoading} error={overviewPoll.error} loadingText="Loading overview...">
        {runtimePoll.data && <RuntimePanel rt={runtimePoll.data} stale={runtimePoll.isStale} />}

        <div className="card">
          <div className="card-title self-help" data-help="Son 15 dakikadaki olaylar. Tıklayınca ilgili log paketine yönlendirir.">Recent Incidents (Last 15m)</div>
          {recentIncidentRibbon.length === 0 ? (
            <div style={{ color: "var(--muted)" }}>No recent incidents in last 15m.</div>
          ) : (
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {recentIncidentRibbon.map((inc) => (
                <button
                  key={inc.incident_id}
                  onClick={() => {
                    const t = (inc.type || "").toLowerCase();
                    if (t.includes("shutdown")) return openLogPack("Shutdown");
                    if (t.includes("timeout") || t.includes("exchange")) return openLogPack("Timeout");
                    if (t.includes("regime")) return openLogPack("Regime");
                    if (t.includes("signal") || t.includes("match")) return openLogPack("No Match");
                    return navigate("/debug");
                  }}
                  style={{
                    padding: "3px 8px",
                    borderRadius: 999,
                    border: "1px solid var(--border)",
                    background: "transparent",
                    color: "var(--muted)",
                    cursor: "pointer",
                    fontSize: 11,
                  }}
                  title={`type=${inc.type} level=${inc.level} status=${inc.status}`}
                >
                  {inc.type} [{inc.level}]
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="card" style={{ borderLeft: "3px solid var(--yellow)" }}>
          <div className="card-title self-help" data-help="En sık görülen blokaj nedenini ve sonraki adımı özetler.">Entry Blocker Snapshot</div>
          {!topBlocker ? (
            <div style={{ color: "var(--muted)" }}>
              Son pencerede blocker kaydı yok — giriş kanalı açık görünüyor.
            </div>
          ) : (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
              <div>
                <div style={{ fontSize: 18, fontWeight: 700, color: "var(--yellow)" }}>{topBlocker.reason}</div>
                <div style={{ color: "var(--muted)", fontSize: 11 }}>count={topBlocker.count}</div>
              </div>
              <div>
                <div style={{ fontSize: 12, fontWeight: 700 }}>Next action</div>
                <div style={{ color: "var(--text)" }}>{topBlockerGuide?.action}</div>
                <div style={{ color: "var(--muted)", fontSize: 11 }}>{topBlockerGuide?.detail}</div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 8 }}>
                  <button
                    className="self-help"
                    data-help="No Match log paketini ac ve esik fail nedenlerini incele."
                    onClick={() => openLogPack("No Match")}
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
                    Open Logs: No Match
                  </button>
                  <button
                    className="self-help"
                    data-help="Regime log paketini aç ve rejim uyumsuzluğunu kontrol et."
                    onClick={() => openLogPack("Regime")}
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
                    Open Logs: Regime
                  </button>
                  <button
                    className="self-help"
                    data-help="Guided debug akisini ac ve otomatik triage adimlarini calistir."
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
                    Open Debug Guided
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        <ResearchEventsSummary researchEvents={research} onOpenDetails={() => navigate("/research")} />

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <Stat label="Mode" value={<StatusBadge value={sb.paper_trading == null ? null : !sb.paper_trading} />} />
          <Stat label="Fills" value={sb.fills_total ?? 0} color="var(--green)" />
          <Stat label="Orders" value={sb.orders_total ?? 0} />
          <Stat label="Blocked" value={sb.blocked_total ?? 0} color={(sb.blocked_total ?? 0) > 0 ? "var(--yellow)" : undefined} />
          <Stat
            label="Kill trips"
            value={sb.kill_switch_trips_total ?? 0}
            color={(sb.kill_switch_trips_total ?? 0) > 0 ? "var(--red)" : undefined}
          />
          <Stat
            label="CB trips"
            value={sb.circuit_breaker_trips_total ?? 0}
            color={(sb.circuit_breaker_trips_total ?? 0) > 0 ? "var(--red)" : undefined}
          />
        </div>

        <div
          className="card"
          style={{
            borderLeft: `3px solid ${
              fitnessStatus === "ok"
                ? "var(--green)"
                : fitnessStatus === "warning"
                  ? "var(--yellow)"
                  : fitnessStatus === "degraded"
                    ? "var(--red)"
                    : "var(--border)"
            }`,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <div>
              <div className="card-title self-help" data-help="Runtime health artifact icindeki data research fitness component durumu.">
                Data Research Fitness
              </div>
              <div style={{ color: "var(--muted)" }}>
                overall.json component surfaced for operator health review
              </div>
            </div>
            {fitnessBadge(fitnessStatus, fitnessConnected)}
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
            <span className={`badge ${fitnessConnected ? "badge-green" : "badge-red"}`}>
              connected={fitnessConnected === null ? "-" : String(fitnessConnected)}
            </span>
            {fitnessStatus ? <span className="badge badge-blue">status={fitnessStatus}</span> : null}
            {fitnessDetail ? <span className="badge badge-gray">{fitnessDetail}</span> : null}
            <button
              onClick={() => navigate("/research")}
              style={{ padding: "3px 8px", borderRadius: 4, border: "1px solid var(--accent)", background: "transparent", color: "var(--accent)", cursor: "pointer", fontSize: 11 }}
            >
              Open Fitness Detail
            </button>
          </div>
        </div>

        <div className="card">
          <div className="card-title self-help" data-help="Sembol bazli gate performansi: hit-rate ve baseline farki.">Alpha Gates</div>
          <AsyncState loading={false} error={null} isEmpty={symbols.length === 0} emptyText="No gate data">
            <table>
              <thead>
                <tr>
                  <th>
                    <TermTip term="Symbol" tr="Islem yapilan enstruman kodu." en="Market symbol identifier." />
                  </th>
                  <th>
                    <TermTip term="Rule" tr="Aktif gate/filtre kurali." en="Active gate/filter rule." />
                  </th>
                  <th>
                    <TermTip term="Hit rate" tr="Kuralin tetiklenme orani." en="Rule trigger frequency ratio." />
                  </th>
                  <th>
                    <TermTip term="Delta baseline" tr="Baz modele gore performans farki." en="Performance delta vs baseline." />
                  </th>
                </tr>
              </thead>
              <tbody>
                {symbols.map((s) => (
                  <tr key={s.symbol}>
                    <td>{s.symbol}</td>
                    <td style={{ color: "var(--muted)" }}>{s.rule_name || "-"}</td>
                    <td>{s.hit_rate != null ? (s.hit_rate * 100).toFixed(1) + "%" : "-"}</td>
                    <td
                      style={{
                        color: (s.delta_vs_baseline ?? 0) > 0
                          ? "var(--green)"
                          : (s.delta_vs_baseline ?? 0) < 0
                            ? "var(--red)"
                            : undefined,
                      }}
                    >
                      {s.delta_vs_baseline != null ? (s.delta_vs_baseline * 100).toFixed(2) + "%" : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </AsyncState>
          {gates.generated_utc && (
            <div style={{ marginTop: 8, color: "var(--muted)", fontSize: 11 }}>Generated: {gates.generated_utc}</div>
          )}
        </div>

        <div className="card">
          <div className="card-title self-help" data-help="Son rejim geçişleri: market modundaki değişimleri gösterir.">Recent Regime Transitions</div>
          <AsyncState loading={false} error={null} isEmpty={regimes.length === 0} emptyText="No recent transitions">
            <table>
              <thead>
                <tr>
                  <th><TermTip term="Time" tr="Olay zamani." en="Event timestamp." /></th>
                  <th><TermTip term="Symbol" tr="Enstruman." en="Instrument symbol." /></th>
                  <th><TermTip term="Timeframe" tr="Rejim penceresi." en="Regime window timeframe." /></th>
                  <th><TermTip term="Regime" tr="Piyasa modu." en="Market state classification." /></th>
                  <th><TermTip term="Confidence" tr="Rejim guven skoru." en="Regime confidence score." /></th>
                </tr>
              </thead>
              <tbody>
                {regimes.slice(0, 10).map((r, i) => (
                  <tr key={i}>
                    <td style={{ color: "var(--muted)" }}>{r.ts ?? "-"}</td>
                    <td>{r.symbol ?? "-"}</td>
                    <td>{r.timeframe ?? "-"}</td>
                    <td><RegimeBadge regime={r.effective_regime} /></td>
                    <td>{r.confidence != null ? (r.confidence * 100).toFixed(0) + "%" : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </AsyncState>
        </div>

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <div className="card" style={{ flex: 1, minWidth: 260 }}>
            <div className="card-title self-help" data-help="Preflight kontrollerinin son degerleri (env/db/runtime).">Preflight Check</div>
            <AsyncState
              loading={false}
              error={null}
              isEmpty={Object.keys(preflight).length === 0}
              emptyText="No preflight data"
            >
              <table>
                <tbody>
                  {Object.entries(preflight).slice(0, 10).map(([k, v]) => (
                    <tr key={k}>
                      <td style={{ color: "var(--muted)" }}>{k}</td>
                      <td>{String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </AsyncState>
          </div>
          <div className="card" style={{ flex: 1, minWidth: 260 }}>
            <div className="card-title self-help" data-help="Sistem guvenilirlik gate metrikleri ve durum alanlari.">Reliability Gate</div>
            <AsyncState
              loading={false}
              error={null}
              isEmpty={Object.keys(reliability).length === 0}
              emptyText="No reliability data"
            >
              <table>
                <tbody>
                  {Object.entries(reliability).slice(0, 10).map(([k, v]) => (
                    <tr key={k}>
                      <td style={{ color: "var(--muted)" }}>{k}</td>
                      <td>{String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </AsyncState>
          </div>
          <div className="card" style={{ flex: 1, minWidth: 260 }}>
            <div className="card-title self-help" data-help="Entry kararlarinda en cok gorulen blocker reason listesi.">Top Blockers (reason)</div>
            <AsyncState
              loading={false}
              error={null}
              isEmpty={topBlockers.length === 0}
              emptyText="No blocked reasons recorded"
            >
              <table>
                <thead>
                  <tr>
                    <th>Reason</th>
                    <th>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {topBlockers.map((row) => (
                    <tr key={row.reason}>
                      <td style={{ color: "var(--muted)" }}>{row.reason}</td>
                      <td>{row.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </AsyncState>
          </div>
        </div>
      </AsyncState>
    </div>
  );
}
