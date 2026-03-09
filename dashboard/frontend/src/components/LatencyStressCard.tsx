import React from "react";
import type { LatencyStressState } from "../api/types";

const LEVEL_COLORS: Record<string, { bg: string; border: string; text: string; badge: string }> = {
  quiet: { bg: "#f0fdf4", border: "#86efac", text: "#166534", badge: "#22c55e" },
  elevated: { bg: "#fffbeb", border: "#fcd34d", text: "#92400e", badge: "#f59e0b" },
  severe: { bg: "#fef2f2", border: "#fca5a5", text: "#991b1b", badge: "#ef4444" },
};

function fmtMs(sec: number | undefined | null): string {
  if (sec == null) return "-";
  return `${(sec * 1000).toFixed(0)}ms`;
}

function fmtPct(v: number | undefined | null): string {
  if (v == null) return "-";
  return `${(v * 100).toFixed(1)}%`;
}

export default function LatencyStressCard({ data }: { data: LatencyStressState | null }) {
  if (!data || !data.available) {
    return (
      <div style={{ padding: "12px 16px", background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 8, opacity: 0.6 }}>
        <strong>Latency Stress</strong>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: "#6b7280" }}>No data available</p>
      </div>
    );
  }

  const level = data.state?.level || "quiet";
  const colors = LEVEL_COLORS[level] || LEVEL_COLORS.quiet;
  const card = data.card || {};
  const noData = (card.rows ?? 0) === 0;

  if (noData) {
    return (
      <div style={{ padding: "12px 16px", background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 8, opacity: 0.7 }}>
        <strong>Latency Stress</strong>
        <span style={{ marginLeft: 8, fontSize: 12, color: "#9ca3af" }}>quiet (no trade data)</span>
        {data.stale && <span style={{ fontSize: 11, color: "#dc2626", marginLeft: 8 }}>STALE</span>}
      </div>
    );
  }

  return (
    <div style={{ padding: "12px 16px", background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <strong style={{ color: colors.text }}>Latency Stress</strong>
        <span style={{
          display: "inline-block", padding: "2px 8px", borderRadius: 12,
          fontSize: 12, fontWeight: 600, color: "#fff", background: colors.badge, textTransform: "uppercase",
        }}>{level}</span>
        {data.stale && <span style={{ fontSize: 11, color: "#dc2626", marginLeft: "auto" }}>STALE</span>}
      </div>

      {card.headline && <p style={{ margin: "0 0 6px", fontSize: 13, color: colors.text, fontWeight: 500 }}>{card.headline}</p>}
      {card.operator_note && <p style={{ margin: "0 0 8px", fontSize: 12, color: "#6b7280", fontStyle: "italic" }}>{card.operator_note}</p>}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, fontSize: 12 }}>
        <div><div style={{ color: "#9ca3af" }}>Fill Rate</div><div style={{ fontWeight: 600 }}>{fmtPct(card.fill_rate)}</div></div>
        <div><div style={{ color: "#9ca3af" }}>p50 Delay</div><div style={{ fontWeight: 600 }}>{fmtMs(card.latency_fill_delay_sec_p50)}</div></div>
        <div><div style={{ color: "#9ca3af" }}>p95 Delay</div><div style={{ fontWeight: 600 }}>{fmtMs(card.latency_fill_delay_sec_p95)}</div></div>
        <div><div style={{ color: "#9ca3af" }}>Lat vs Net</div><div style={{ fontWeight: 600 }}>{card.latency_impact_vs_net_corr != null ? card.latency_impact_vs_net_corr.toFixed(3) : "-"}</div></div>
      </div>

      {data.recommended_action && (
        <p style={{ margin: "8px 0 0", fontSize: 11, color: "#6b7280" }}>Action: {data.recommended_action}</p>
      )}
    </div>
  );
}
