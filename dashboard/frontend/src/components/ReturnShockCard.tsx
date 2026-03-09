import React from "react";
import type { ReturnShockState } from "../api/types";

const LEVEL_COLORS: Record<string, { bg: string; border: string; text: string; badge: string }> = {
  quiet: { bg: "#f0fdf4", border: "#86efac", text: "#166534", badge: "#22c55e" },
  elevated: { bg: "#fffbeb", border: "#fcd34d", text: "#92400e", badge: "#f59e0b" },
  severe: { bg: "#fef2f2", border: "#fca5a5", text: "#991b1b", badge: "#ef4444" },
};

const DIR_COLORS: Record<string, string> = {
  UP: "#16a34a",
  DOWN: "#dc2626",
  FLAT: "#6b7280",
};

function fmtPct(v: number | undefined | null): string {
  if (v == null) return "-";
  return `${(v * 100).toFixed(2)}%`;
}

function fmtTs(ms: number | undefined | null): string {
  if (!ms) return "-";
  return new Date(ms).toLocaleTimeString();
}

function fmtRetBps(v: number | undefined | null): string {
  if (v == null) return "-";
  return `${(v * 10000).toFixed(2)} bps`;
}

function fmtNum(v: number | undefined | null, digits = 0): string {
  if (v == null) return "-";
  return v.toFixed(digits);
}

export default function ReturnShockCard({ data }: { data: ReturnShockState | null }) {
  if (!data || !data.available) {
    return (
      <div style={{ padding: "12px 16px", background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 8, opacity: 0.6 }}>
        <strong>Return Shock</strong>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: "#6b7280" }}>No data available</p>
      </div>
    );
  }

  const level = data.state?.level || "quiet";
  const colors = LEVEL_COLORS[level] || LEVEL_COLORS.quiet;
  const card = data.card || {};
  const freshness = (data.state?.freshness || {}) as Record<string, unknown>;
  const freshnessStatus = String(freshness.status || card.freshness_status || "");
  const direction = card.dominant_direction || data.state?.dominant_direction || "";
  const dirColor = DIR_COLORS[direction] || "#6b7280";

  return (
    <div style={{ padding: "12px 16px", background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <strong style={{ color: colors.text }}>Return Shock</strong>
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
        {freshnessStatus && (
          <span style={{ fontSize: 11, color: freshnessStatus === "fresh" ? "#16a34a" : "#dc2626", marginLeft: "auto" }}>
            {freshnessStatus.toUpperCase()}
          </span>
        )}
        {data.stale && !freshnessStatus && <span style={{ fontSize: 11, color: "#dc2626", marginLeft: "auto" }}>STALE</span>}
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
          <div style={{ color: "#9ca3af" }}>High / Med</div>
          <div style={{ fontWeight: 600 }}>{card.high_count ?? "-"} / {card.medium_count ?? "-"}</div>
        </div>
        <div>
          <div style={{ color: "#9ca3af" }}>Direction</div>
          <div style={{ fontWeight: 700, color: dirColor }}>{direction || "-"}</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 12, marginTop: 8 }}>
        <div>
          <div style={{ color: "#9ca3af" }}>Avg |Ret| (1s)</div>
          <div style={{ fontWeight: 600 }}>{fmtRetBps(card.avg_abs_ret_1_tagged)}</div>
        </div>
        <div>
          <div style={{ color: "#9ca3af" }}>Avg Intensity</div>
          <div style={{ fontWeight: 600 }}>{fmtNum(card.avg_trade_intensity_tagged, 0)}</div>
        </div>
      </div>

      {card.latest_alert_ts_ms ? (
        <div style={{ marginTop: 8, fontSize: 11, color: "#9ca3af" }}>
          Latest alert: {fmtTs(card.latest_alert_ts_ms)}
          {data.recommended_action && (
            <span style={{ marginLeft: 8, fontStyle: "italic" }}>Action: {data.recommended_action}</span>
          )}
        </div>
      ) : null}
    </div>
  );
}
