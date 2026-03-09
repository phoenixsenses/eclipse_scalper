import React from "react";
import AsyncState from "./AsyncState";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function asArray<T = Record<string, unknown>>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function metricString(value: unknown, suffix = ""): string {
  if (typeof value === "number" && Number.isFinite(value)) return `${value}${suffix}`;
  if (typeof value === "string" && value.trim()) return value;
  return "-";
}

function boolBadge(value: unknown) {
  if (value === true) return <span className="badge badge-green">true</span>;
  if (value === false) return <span className="badge badge-red">false</span>;
  return <span className="badge badge-gray">-</span>;
}

function titleizeLane(name: string): string {
  return name
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function levelBadge(level?: string | null, stale?: boolean) {
  const normalized = String(level || "quiet").toLowerCase();
  if (stale) return <span className="badge badge-gray">{level || "stale"}</span>;
  if (normalized === "severe") return <span className="badge badge-red">{level}</span>;
  if (normalized === "elevated" || normalized === "medium") return <span className="badge badge-yellow">{level}</span>;
  if (normalized === "high") return <span className="badge badge-red">{level}</span>;
  if (normalized === "quiet" || normalized === "none") return <span className="badge badge-green">{level || "quiet"}</span>;
  return <span className="badge badge-blue">{level || "-"}</span>;
}

function freshnessBadge(status?: string | null) {
  const normalized = String(status || "unknown").toLowerCase();
  if (normalized === "fresh" || normalized === "live") return <span className="badge badge-green">{status}</span>;
  if (normalized === "stale") return <span className="badge badge-gray">{status}</span>;
  if (normalized === "degraded" || normalized === "aging") return <span className="badge badge-yellow">{status}</span>;
  return <span className="badge badge-blue">{status || "unknown"}</span>;
}

function levelRank(level?: string | null): number {
  const normalized = String(level || "quiet").toLowerCase();
  if (normalized === "severe" || normalized === "high") return 0;
  if (normalized === "elevated" || normalized === "medium") return 1;
  if (normalized === "quiet" || normalized === "none") return 2;
  return 3;
}

function sectionTitle(title: string, subtitle: string) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
      <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: "0.03em", textTransform: "uppercase" }}>{title}</div>
      <div style={{ color: "var(--muted)", fontSize: 12 }}>{subtitle}</div>
    </div>
  );
}

function SummaryChip({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div
      style={{
        minWidth: 140,
        padding: "10px 12px",
        border: "1px solid var(--border)",
        borderRadius: 8,
        background: "var(--surface-2)",
      }}
    >
      <div style={{ color: "var(--muted)", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700, marginTop: 4 }}>{value}</div>
    </div>
  );
}

function ResearchStateCard({
  lane,
  payload,
  metricLabels,
}: {
  lane: string;
  payload: Record<string, unknown>;
  metricLabels: Array<{ key: string; label: string }>;
}) {
  const meta = asRecord(payload._meta);
  const state = asRecord(payload.state);
  const card = asRecord(payload.card);
  const stale = Boolean(meta.stale);
  const level = String(state.level || "missing");
  const freshness = asRecord(state.freshness);
  const freshnessStatus = String(freshness.status || (stale ? "stale" : "unknown"));
  const reasons = asArray<string>(state.reasons);
  const title = titleizeLane(lane);
  const summary = String(payload.dashboard_summary || payload.notification_text || card.headline || "No state payload available.");

  return (
    <div className="card" style={{ minWidth: 280, borderLeft: `3px solid ${stale ? "var(--muted)" : level === "severe" ? "var(--red)" : level === "elevated" ? "var(--yellow)" : "var(--green)"}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>{title}</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {levelBadge(level, stale)}
          {freshnessBadge(freshnessStatus)}
        </div>
      </div>
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>{metricString(card.headline || state.headline || title)}</div>
      <div style={{ color: "var(--muted)", marginBottom: 8 }}>{summary}</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 8 }}>
        <span className="badge badge-blue">action={metricString(payload.recommended_action)}</span>
        {card.top_side ? <span className="badge badge-gray">top_side={metricString(card.top_side)}</span> : null}
        {state.primary_side_bias ? <span className="badge badge-gray">bias={metricString(state.primary_side_bias)}</span> : null}
        {state.dominant_direction ? <span className="badge badge-gray">direction={metricString(state.dominant_direction)}</span> : null}
        {card.operator_note ? <span className="badge badge-gray">note={metricString(card.operator_note)}</span> : null}
      </div>
      {reasons.length > 0 ? (
        <div style={{ marginBottom: 8, color: "var(--muted)" }}>
          reasons: {reasons.join(", ")}
        </div>
      ) : null}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 8 }}>
        {metricLabels.map((metric) => (
          <div key={metric.key} style={{ border: "1px solid var(--border)", borderRadius: 6, padding: 8, background: "var(--surface-2)" }}>
            <div style={{ fontSize: 16, fontWeight: 700 }}>{metricString(card[metric.key])}</div>
            <div style={{ color: "var(--muted)", fontSize: 10, textTransform: "uppercase" }}>{metric.label}</div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 8, color: "var(--muted)", fontSize: 11 }}>
        payload_age_sec={metricString(meta.age_sec)} path={metricString(meta.path)}
      </div>
    </div>
  );
}

function ResearchWatchlist({
  lane,
  payload,
}: {
  lane: string;
  payload: Record<string, unknown>;
}) {
  const rows = asArray<Record<string, unknown>>(payload.rows);
  const banner = asRecord(payload.banner);
  const topSummary = asRecord(payload.top_summary);
  const meta = asRecord(payload._meta);
  const title = `${titleizeLane(lane)} Watchlist`;
  return (
    <div className="card" style={{ minWidth: 320 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>{title}</div>
        {meta.stale ? <span className="badge badge-gray">stale</span> : null}
      </div>
      {banner.headline ? (
        <div style={{ marginBottom: 8, padding: 8, borderRadius: 6, background: "var(--surface-2)" }}>
          <div style={{ fontWeight: 700 }}>{metricString(banner.headline)}</div>
          <div style={{ color: "var(--muted)", fontSize: 11 }}>{metricString(banner.detail || banner.summary)}</div>
        </div>
      ) : null}
      {Object.keys(topSummary).length > 0 ? (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
          {topSummary.symbol ? <span className="badge badge-blue">{metricString(topSummary.symbol)}</span> : null}
          {topSummary.top_symbol ? <span className="badge badge-blue">{metricString(topSummary.top_symbol)}</span> : null}
          {topSummary.state_level ? levelBadge(String(topSummary.state_level), String(topSummary.freshness_status || "").toLowerCase() === "stale") : null}
          {topSummary.freshness_status ? freshnessBadge(String(topSummary.freshness_status)) : null}
          {topSummary.recommended_action ? <span className="badge badge-gray">action={metricString(topSummary.recommended_action)}</span> : null}
        </div>
      ) : null}
      <AsyncState loading={false} error={null} isEmpty={rows.length === 0} emptyText="No watchlist rows">
        <table>
          <thead>
            <tr>
              {rows[0] ? Object.keys(rows[0]).slice(0, 5).map((key) => <th key={key}>{key}</th>) : null}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 6).map((row, idx) => (
              <tr key={`${lane}-${idx}`}>
                {Object.keys(row).slice(0, 5).map((key) => (
                  <td key={key}>{metricString(row[key])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </AsyncState>
    </div>
  );
}

export function ResearchEventsSummary({
  researchEvents,
  onOpenDetails,
}: {
  researchEvents: Record<string, unknown>;
  onOpenDetails?: () => void;
}) {
  const dailyReport = asRecord(researchEvents.daily_report);
  const headline = asRecord(dailyReport.headline);
  const watchboard = asRecord(researchEvents.watchboard);
  const banner = asRecord(watchboard.banner);
  const summary = asRecord(watchboard.summary);
  const meta = asRecord(watchboard._meta);
  const lanes = asArray<Record<string, unknown>>(watchboard.lanes);
  const activeCount = lanes.filter((lane) => levelRank(String(asRecord(lane.state).level || lane.level || "quiet")) < 2).length;

  return (
    <div className="card" style={{ borderLeft: "3px solid var(--accent)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>Research Event Watchboard</div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {summary.top_lane ? <span className="badge badge-blue">top_lane={metricString(summary.top_lane)}</span> : null}
          {meta.stale ? <span className="badge badge-gray">stale</span> : null}
          {onOpenDetails ? (
            <button
              onClick={onOpenDetails}
              style={{ padding: "3px 8px", borderRadius: 4, border: "1px solid var(--accent)", background: "transparent", color: "var(--accent)", cursor: "pointer", fontSize: 11 }}
            >
              Open Research Events
            </button>
          ) : null}
        </div>
      </div>
      {banner.headline ? (
        <>
          <div style={{ fontSize: 18, fontWeight: 700 }}>{metricString(banner.headline)}</div>
          <div style={{ color: "var(--muted)", marginTop: 4 }}>{metricString(banner.detail || banner.summary)}</div>
        </>
      ) : (
        <div style={{ color: "var(--muted)" }}>No research event watchboard payload available.</div>
      )}
      {Object.keys(headline).length > 0 ? (
        <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
          <span className="badge badge-gray">daily_event={metricString(headline.event_lanes)}</span>
          <span className="badge badge-gray">daily_recovery={metricString(headline.regime_recovery_prep)}</span>
          <span className="badge badge-gray">daily_promotion={metricString(headline.pocket_promotion_checklist)}</span>
        </div>
      ) : null}
      {lanes.length > 0 ? (
        <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
          <span className="badge badge-blue">active_lanes={activeCount}</span>
          {lanes.slice(0, 4).map((lane, idx) => {
            const state = asRecord(lane.state);
            const freshness = asRecord(state.freshness);
            return (
              <span key={idx} className="badge badge-gray">
                {metricString(lane.lane || lane.name)} {metricString(state.level)} {metricString(freshness.status)}
              </span>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

export default function ResearchEventsPanel({
  researchEvents,
}: {
  researchEvents: Record<string, unknown>;
}) {
  const dailyReport = asRecord(researchEvents.daily_report);
  const fitnessReport = asRecord(researchEvents.data_research_fitness);
  const dailyHeadline = asRecord(dailyReport.headline);
  const dailyMeta = asRecord(dailyReport._meta);
  const fitnessMeta = asRecord(fitnessReport._meta);
  const fitnessContract = asRecord(fitnessReport.contract);
  const fitnessFeatureStats = asRecord(fitnessReport.feature_stats);
  const fitnessSampleStats = asRecord(fitnessReport.sample_stats);
  const fitnessSymbols = asArray<string>(fitnessReport.symbols);
  const fitnessWarnings = asArray<string>(fitnessReport.warnings);
  const fitnessFailures = asArray<string>(fitnessReport.failures);
  const fitnessCsvStatus = asRecord(fitnessReport.csv_status);
  const fitnessDbReadyDetails = asRecord(fitnessReport.db_ready_details);
  const researchWatchboard = asRecord(researchEvents.watchboard);
  const researchStates = asRecord(researchEvents.states);
  const researchWatchlists = asRecord(researchEvents.watchlists);
  const researchLanes = asArray<Record<string, unknown>>(researchWatchboard.lanes).sort((left, right) => {
    const leftState = asRecord(left.state);
    const rightState = asRecord(right.state);
    const leftFreshness = String(asRecord(leftState.freshness).status || "");
    const rightFreshness = String(asRecord(rightState.freshness).status || "");
    const leftStale = leftFreshness.toLowerCase() === "stale";
    const rightStale = rightFreshness.toLowerCase() === "stale";
    if (leftStale !== rightStale) return leftStale ? -1 : 1;
    return levelRank(String(leftState.level || left.level || "quiet")) - levelRank(String(rightState.level || right.level || "quiet"));
  });
  const researchBanner = asRecord(researchWatchboard.banner);
  const researchSummary = asRecord(researchWatchboard.summary);
  const watchboardMeta = asRecord(researchWatchboard._meta);
  const stateCards = [
    { lane: "liquidation", metrics: [{ key: "recent_alert_count", label: "Recent Alerts" }, { key: "tagged_rate", label: "Tagged Rate" }, { key: "max_consecutive_tagged", label: "Max Consecutive" }, { key: "max_liq_rate_recent", label: "Max Liq Rate" }] },
    { lane: "spread_stress", metrics: [{ key: "recent_alert_count", label: "Recent Alerts" }, { key: "high_count", label: "High Count" }, { key: "medium_count", label: "Medium Count" }, { key: "avg_spread_tagged", label: "Avg Spread Tagged" }] },
    { lane: "fill_toxicity", metrics: [{ key: "rows", label: "Rows" }, { key: "toxicity_score", label: "Toxicity Score" }, { key: "adverse_bps_mean", label: "Adverse Bps Mean" }, { key: "pnl_bps_mean", label: "Pnl Bps Mean" }] },
    { lane: "latency_stress", metrics: [{ key: "rows", label: "Rows" }, { key: "fill_rate", label: "Fill Rate" }, { key: "latency_fill_delay_sec_p50", label: "P50 Delay Sec" }, { key: "latency_fill_delay_sec_p95", label: "P95 Delay Sec" }] },
    { lane: "return_shock", metrics: [{ key: "recent_alert_count", label: "Recent Alerts" }, { key: "high_count", label: "High Count" }, { key: "medium_count", label: "Medium Count" }, { key: "avg_abs_ret_1_tagged", label: "Avg Abs Ret1" }] },
    { lane: "volume_vacuum", metrics: [{ key: "recent_alert_count", label: "Recent Alerts" }, { key: "high_count", label: "High Count" }, { key: "medium_count", label: "Medium Count" }, { key: "avg_trade_intensity_tagged", label: "Avg Trade Intensity" }] },
    { lane: "volatility_burst", metrics: [{ key: "recent_alert_count", label: "Recent Alerts" }, { key: "high_count", label: "High Count" }, { key: "medium_count", label: "Medium Count" }, { key: "avg_abs_ret_1_tagged", label: "Avg Abs Ret1" }] },
    { lane: "book_proxy_pressure", metrics: [{ key: "recent_alert_count", label: "Recent Alerts" }, { key: "high_count", label: "High Count" }, { key: "medium_count", label: "Medium Count" }, { key: "avg_abs_imbalance_tagged", label: "Avg Abs Imbalance" }] },
  ];
  const staleLaneCount = researchLanes.filter((lane) => String(asRecord(asRecord(lane.state).freshness).status || "").toLowerCase() === "stale").length;
  const activeLaneCount = researchLanes.filter((lane) => levelRank(String(asRecord(lane.state).level || lane.level || "quiet")) < 2).length;
  const populatedStateCount = stateCards.filter(({ lane }) => Object.keys(asRecord(researchStates[lane])).length > 0).length;
  const populatedWatchlistCount = ["liquidation", "spread_stress", "return_shock", "volume_vacuum", "volatility_burst", "book_proxy_pressure"]
    .filter((lane) => asArray(asRecord(researchWatchlists[lane]).rows).length > 0).length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div
        className="card"
        style={{
          borderLeft: "3px solid var(--yellow)",
          background: "linear-gradient(180deg, rgba(255, 196, 61, 0.12), rgba(0, 0, 0, 0))",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <div>
            <div className="card-title" style={{ marginBottom: 6 }}>Data Research Fitness</div>
            <div style={{ color: "var(--muted)" }}>
              path={metricString(fitnessMeta.path)}
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {fitnessMeta.stale ? <span className="badge badge-gray">stale</span> : <span className="badge badge-green">current</span>}
            {fitnessReport.status ? levelBadge(String(fitnessReport.status).toLowerCase() === "fail" ? "severe" : String(fitnessReport.status).toLowerCase() === "warn" ? "elevated" : "quiet", false) : null}
            {fitnessContract.tier ? <span className="badge badge-blue">tier={metricString(fitnessContract.tier)}</span> : null}
          </div>
        </div>
        {Object.keys(fitnessReport).length > 0 && fitnessReport.status ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10, marginTop: 12 }}>
            <div style={{ padding: "10px 12px", borderRadius: 8, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Research Fitness</div>
              <div style={{ fontWeight: 700, marginTop: 4 }}>{metricString(fitnessReport.status)}</div>
              <div style={{ color: "var(--muted)", fontSize: 12 }}>db_ready={metricString(fitnessReport.db_ready)}</div>
            </div>
            <div style={{ padding: "10px 12px", borderRadius: 8, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Contract</div>
              <div style={{ fontWeight: 700, marginTop: 4 }}>{metricString(fitnessContract.status)}</div>
              <div style={{ color: "var(--muted)", fontSize: 12 }}>requires_book={metricString(fitnessContract.requires_book)}</div>
            </div>
            <div style={{ padding: "10px 12px", borderRadius: 8, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Symbol Coverage</div>
              <div style={{ fontWeight: 700, marginTop: 4 }}>{fitnessSymbols.length} symbols</div>
              <div style={{ color: "var(--muted)", fontSize: 12 }}>{fitnessSymbols.join(", ") || "-"}</div>
            </div>
          </div>
        ) : (
          <div style={{ color: "var(--muted)", marginTop: 12 }}>No data research fitness payload available yet.</div>
        )}
        {fitnessSymbols.length > 0 ? (
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
            {fitnessSymbols.slice(0, 4).map((symbol) => {
              const stats = asRecord(fitnessFeatureStats[symbol]);
              return (
                <span key={symbol} className="badge badge-gray">
                  {symbol} rows={metricString(stats.feature_rows)} mid={metricString(stats.has_mid)} spread={metricString(stats.has_spread)}
                </span>
              );
            })}
          </div>
        ) : null}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12, marginTop: 12 }}>
          <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 12, background: "var(--surface-2)" }}>
            <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Inputs / Freshness</div>
            <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 8 }}>
              <div><span style={{ color: "var(--muted)" }}>db</span> {metricString(fitnessReport.db)}</div>
              <div><span style={{ color: "var(--muted)" }}>csv</span> {metricString(fitnessReport.csv)}</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <span className="badge badge-gray">fresh_sec={metricString(fitnessReport.fresh_sec)}</span>
                <span className="badge badge-gray">csv_age_sec={metricString(fitnessCsvStatus.age_sec)}</span>
                <span className="badge badge-gray">csv_detail={metricString(fitnessCsvStatus.detail)}</span>
              </div>
            </div>
          </div>
          <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 12, background: "var(--surface-2)" }}>
            <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Failure Breakdown</div>
            <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
              <span className={`badge ${fitnessFailures.length > 0 ? "badge-red" : "badge-green"}`}>failures={fitnessFailures.length}</span>
              <span className={`badge ${fitnessWarnings.length > 0 ? "badge-yellow" : "badge-green"}`}>warnings={fitnessWarnings.length}</span>
              <span className="badge badge-gray">db_ready={metricString(fitnessReport.db_ready)}</span>
            </div>
            <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
              {fitnessFailures.length > 0 ? fitnessFailures.map((item) => (
                <div key={`failure-${item}`} style={{ color: "var(--red)", fontSize: 12 }}>{item}</div>
              )) : <div style={{ color: "var(--muted)", fontSize: 12 }}>No failures</div>}
              {fitnessWarnings.length > 0 ? fitnessWarnings.map((item) => (
                <div key={`warning-${item}`} style={{ color: "var(--yellow)", fontSize: 12 }}>{item}</div>
              )) : null}
            </div>
          </div>
        </div>
        {Object.keys(fitnessDbReadyDetails).length > 0 ? (
          <div style={{ marginTop: 12 }}>
            {sectionTitle("DB Readiness Detail", "Freshness and table checks from validator input")}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 8 }}>
              {Object.entries(fitnessDbReadyDetails).slice(0, 8).map(([key, value]) => (
                <span key={key} className="badge badge-gray">
                  {key}={typeof value === "object" ? JSON.stringify(value) : metricString(value)}
                </span>
              ))}
            </div>
          </div>
        ) : null}
        {fitnessSymbols.length > 0 ? (
          <div style={{ marginTop: 16 }}>
            {sectionTitle("Symbol Fitness", "Sample coverage and feature computability by active symbol")}
            <table style={{ marginTop: 8 }}>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Trades</th>
                  <th>Marks</th>
                  <th>Liqs</th>
                  <th>Feature Rows</th>
                  <th>Mid</th>
                  <th>Spread</th>
                  <th>Trade Intensity</th>
                </tr>
              </thead>
              <tbody>
                {fitnessSymbols.map((symbol) => {
                  const feature = asRecord(fitnessFeatureStats[symbol]);
                  const sample = asRecord(fitnessSampleStats[symbol]);
                  return (
                    <tr key={`fitness-symbol-${symbol}`}>
                      <td>{symbol}</td>
                      <td>{metricString(sample.agg_trade_rows)}</td>
                      <td>{metricString(sample.mark_price_rows)}</td>
                      <td>{metricString(sample.liquidation_rows)}</td>
                      <td>{metricString(feature.feature_rows)}</td>
                      <td>{boolBadge(feature.has_mid)}</td>
                      <td>{boolBadge(feature.has_spread)}</td>
                      <td>{boolBadge(feature.has_trade_intensity)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>

      <div
        className="card"
        style={{
          borderLeft: "3px solid var(--blue)",
          background: "linear-gradient(180deg, rgba(56, 139, 253, 0.12), rgba(0, 0, 0, 0))",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <div>
            <div className="card-title" style={{ marginBottom: 6 }}>Daily Research Headline</div>
            <div style={{ color: "var(--muted)" }}>
              report_date={metricString(dailyReport.report_date)} path={metricString(dailyMeta.path)}
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {dailyMeta.stale ? <span className="badge badge-gray">stale</span> : <span className="badge badge-green">current</span>}
            {dailyHeadline.event_lanes ? <span className="badge badge-blue">event={metricString(dailyHeadline.event_lanes)}</span> : null}
            {dailyHeadline.regime_recovery_prep ? <span className="badge badge-yellow">recovery={metricString(dailyHeadline.regime_recovery_prep)}</span> : null}
            {dailyHeadline.pocket_promotion_checklist ? <span className="badge badge-gray">promotion={metricString(dailyHeadline.pocket_promotion_checklist)}</span> : null}
          </div>
        </div>
        {dailyReport.event_lane || dailyReport.recovery || dailyReport.promotion ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10, marginTop: 12 }}>
            <div style={{ padding: "10px 12px", borderRadius: 8, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Event Gate</div>
              <div style={{ fontWeight: 700, marginTop: 4 }}>{metricString(asRecord(dailyReport.event_lane).summary)}</div>
            </div>
            <div style={{ padding: "10px 12px", borderRadius: 8, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Recovery Prep</div>
              <div style={{ fontWeight: 700, marginTop: 4 }}>{metricString(asRecord(dailyReport.recovery).summary)}</div>
            </div>
            <div style={{ padding: "10px 12px", borderRadius: 8, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Promotion</div>
              <div style={{ fontWeight: 700, marginTop: 4 }}>{metricString(asRecord(dailyReport.promotion).summary)}</div>
            </div>
          </div>
        ) : (
          <div style={{ color: "var(--muted)", marginTop: 12 }}>No daily research report payload available yet.</div>
        )}
      </div>

      <div className="card" style={{ borderLeft: "3px solid var(--accent)" }}>
        <div className="card-title">Research Event Watchboard</div>
        {researchBanner.headline ? (
          <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", gap: 12 }}>
            <div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{metricString(researchBanner.headline)}</div>
              <div style={{ color: "var(--muted)" }}>{metricString(researchBanner.detail || researchBanner.summary)}</div>
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
              {researchSummary.top_lane ? <span className="badge badge-blue">top_lane={metricString(researchSummary.top_lane)}</span> : null}
              {researchSummary.state_counts ? <span className="badge badge-gray">states={JSON.stringify(researchSummary.state_counts)}</span> : null}
              {watchboardMeta.stale ? <span className="badge badge-gray">stale</span> : null}
            </div>
          </div>
        ) : (
          <div style={{ color: "var(--muted)" }}>No research event watchboard payload available.</div>
        )}
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 12 }}>
          <SummaryChip label="Top Lane" value={metricString(researchSummary.top_lane || "-")} />
          <SummaryChip label="Active Lanes" value={activeLaneCount} />
          <SummaryChip label="Stale Lanes" value={staleLaneCount} />
          <SummaryChip label="State Cards" value={populatedStateCount} />
          <SummaryChip label="Watchlists" value={populatedWatchlistCount} />
        </div>
        {researchLanes.length > 0 ? (
          <div style={{ marginTop: 16 }}>
            {sectionTitle("Lane Health", "Prioritized by stale and elevated/severe states")}
            <table>
              <thead>
                <tr>
                  <th>Lane</th>
                  <th>Level</th>
                  <th>Freshness</th>
                  <th>Action</th>
                  <th>Summary</th>
                </tr>
              </thead>
              <tbody>
                {researchLanes.map((lane, idx) => {
                  const state = asRecord(lane.state);
                  const freshness = asRecord(state.freshness);
                  return (
                    <tr key={`research-lane-${idx}`}>
                      <td>{metricString(lane.lane || lane.name)}</td>
                      <td>{levelBadge(String(state.level || lane.level || "quiet"), String(freshness.status || "").toLowerCase() === "stale")}</td>
                      <td>{freshnessBadge(String(freshness.status || lane.freshness_status || "unknown"))}</td>
                      <td>{metricString(lane.recommended_action)}</td>
                      <td style={{ color: "var(--muted)" }}>{metricString(lane.dashboard_summary || lane.summary)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>

      <div style={{ marginTop: 4 }}>
        {sectionTitle("Single-Lane States", "Detailed cards for lanes with direct operator context")}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 12 }}>
        {stateCards.map(({ lane, metrics }) => (
          <ResearchStateCard key={lane} lane={lane} payload={asRecord(researchStates[lane])} metricLabels={metrics} />
        ))}
      </div>

      <div style={{ marginTop: 4 }}>
        {sectionTitle("Cross-Symbol Watchlists", "Symbol-level priority queues for multi-asset event lanes")}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 12 }}>
        {["liquidation", "spread_stress", "return_shock", "volume_vacuum", "volatility_burst", "book_proxy_pressure"].map((lane) => (
          <ResearchWatchlist key={lane} lane={lane} payload={asRecord(researchWatchlists[lane])} />
        ))}
      </div>
    </div>
  );
}
