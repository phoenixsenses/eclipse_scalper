import React from "react";
import type { LiqAlertState } from "../api/types";

const LEVEL_COLORS: Record<string, { bg: string; border: string; text: string; badge: string }> = {
  quiet: { bg: "#f0fdf4", border: "#86efac", text: "#166534", badge: "#22c55e" },
  elevated: { bg: "#fffbeb", border: "#fcd34d", text: "#92400e", badge: "#f59e0b" },
  severe: { bg: "#fef2f2", border: "#fca5a5", text: "#991b1b", badge: "#ef4444" },
};

function fmtPct(v: number | undefined | null): string {
  if (v == null) return "-";
  return `${(v * 100).toFixed(2)}%`;
}

function fmtTs(ms: number | undefined | null): string {
  if (!ms) return "-";
  return new Date(ms).toLocaleTimeString();
}

function fmtRate(v: number | undefined | null): string {
  if (v == null) return "-";
  return v.toFixed(4);
}

export default function LiqAlertCard({ data }: { data: LiqAlertState | null }) {
  if (!data || !data.available) {
    return (
      <div style={{ padding: "12px 16px", background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 8, opacity: 0.6 }}>
        <strong>Liquidation Regime</strong>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: "#6b7280" }}>No data available</p>
      </div>
    );
  }

  const level = data.state?.level || "quiet";
  const colors = LEVEL_COLORS[level] || LEVEL_COLORS.quiet;
  const card = data.card || {};
  const summary = data.summary_snapshot || {};
  const sideBias = summary.side_bias_counts || {};
  const severityCounts = summary.severity_counts || {};

  return (
    <div style={{ padding: "12px 16px", background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <strong style={{ color: colors.text }}>Liquidation Regime</strong>
        <span style={{
          display: "inline-block",
          padding: "2px 8px",
          borderRadius: 12,
          fontSize: 12,
          fontWeight: 600,
          color: "#fff",
          background: colors.badge,
          textTransform: "uppercase",
        }}>
          {level}
        </span>
        {data.symbol && <span style={{ fontSize: 12, color: "#6b7280" }}>{data.symbol}</span>}
        {data.stale && <span style={{ fontSize: 11, color: "#dc2626", marginLeft: "auto" }}>STALE</span>}
      </div>

      {card.headline && (
        <p style={{ margin: "0 0 6px", fontSize: 13, color: colors.text, fontWeight: 500 }}>{card.headline}</p>
      )}
      {card.operator_note && (
        <p style={{ margin: "0 0 8px", fontSize: 12, color: "#6b7280", fontStyle: "italic" }}>{card.operator_note}</p>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, fontSize: 12 }}>
        <div>
          <div style={{ color: "#9ca3af" }}>Alerts</div>
          <div style={{ fontWeight: 600 }}>{card.recent_alert_count ?? "-"}</div>
        </div>
        <div>
          <div style={{ color: "#9ca3af" }}>Tagged Rate</div>
          <div style={{ fontWeight: 600 }}>{fmtPct(card.tagged_rate)}</div>
        </div>
        <div>
          <div style={{ color: "#9ca3af" }}>Max Consec.</div>
          <div style={{ fontWeight: 600 }}>{card.max_consecutive_tagged ?? "-"}</div>
        </div>
        <div>
          <div style={{ color: "#9ca3af" }}>Max Liq Rate</div>
          <div style={{ fontWeight: 600 }}>{fmtRate(card.max_liq_rate_recent)}</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 12, marginTop: 8 }}>
        <div>
          <div style={{ color: "#9ca3af" }}>Side Bias</div>
          <div style={{ fontWeight: 600 }}>{data.state.primary_side_bias}</div>
          <div style={{ fontSize: 11, color: "#9ca3af" }}>
            {Object.entries(sideBias).map(([k, v]) => `${k}: ${v}`).join(", ") || "-"}
          </div>
        </div>
        <div>
          <div style={{ color: "#9ca3af" }}>Severity</div>
          <div style={{ fontWeight: 600 }}>{data.state.dominant_severity}</div>
          <div style={{ fontSize: 11, color: "#9ca3af" }}>
            {Object.entries(severityCounts).map(([k, v]) => `${k}: ${v}`).join(", ") || "-"}
          </div>
        </div>
      </div>

      {card.latest_alert_ts_ms ? (
        <div style={{ marginTop: 8, fontSize: 11, color: "#9ca3af" }}>
          Latest alert: {fmtTs(card.latest_alert_ts_ms)}
        </div>
      ) : null}

      {data.alerts && data.alerts.length > 0 && (
        <details style={{ marginTop: 10 }}>
          <summary style={{ cursor: "pointer", fontSize: 12, color: colors.text }}>
            Recent Alerts ({data.alerts.length})
          </summary>
          <div style={{ maxHeight: 200, overflowY: "auto", marginTop: 4 }}>
            <table style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
                  <th style={{ textAlign: "left", padding: "2px 4px" }}>Time</th>
                  <th style={{ textAlign: "left", padding: "2px 4px" }}>Side</th>
                  <th style={{ textAlign: "left", padding: "2px 4px" }}>Severity</th>
                  <th style={{ textAlign: "right", padding: "2px 4px" }}>Liq Rate</th>
                  <th style={{ textAlign: "right", padding: "2px 4px" }}>Spread</th>
                </tr>
              </thead>
              <tbody>
                {data.alerts.slice(0, 20).map((a, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #f3f4f6" }}>
                    <td style={{ padding: "2px 4px" }}>{fmtTs(a.ts_ms)}</td>
                    <td style={{ padding: "2px 4px" }}>{a.side_bias || "-"}</td>
                    <td style={{ padding: "2px 4px" }}>{a.severity || "-"}</td>
                    <td style={{ padding: "2px 4px", textAlign: "right" }}>{fmtRate(a.liq_rate_per_sec)}</td>
                    <td style={{ padding: "2px 4px", textAlign: "right" }}>{a.spread != null ? (a.spread * 10000).toFixed(2) + " bps" : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}
