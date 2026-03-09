import React from "react";
import type { FillToxicityState } from "../api/types";

const LEVEL_COLORS: Record<string, { bg: string; border: string; text: string; badge: string }> = {
  quiet: { bg: "#f0fdf4", border: "#86efac", text: "#166534", badge: "#22c55e" },
  elevated: { bg: "#fffbeb", border: "#fcd34d", text: "#92400e", badge: "#f59e0b" },
  severe: { bg: "#fef2f2", border: "#fca5a5", text: "#991b1b", badge: "#ef4444" },
};

function fmtBps(v: number | undefined | null): string {
  if (v == null) return "-";
  return `${v.toFixed(2)} bps`;
}

export default function FillToxicityCard({ data }: { data: FillToxicityState | null }) {
  if (!data || !data.available) {
    return (
      <div style={{ padding: "12px 16px", background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 8, opacity: 0.6 }}>
        <strong>Fill Toxicity</strong>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: "#6b7280" }}>No data available</p>
      </div>
    );
  }

  const level = data.state?.level || "quiet";
  const colors = LEVEL_COLORS[level] || LEVEL_COLORS.quiet;
  const card = data.card || {};
  const noData = (data.rows ?? 0) === 0;

  if (noData) {
    return (
      <div style={{ padding: "12px 16px", background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 8, opacity: 0.7 }}>
        <strong>Fill Toxicity</strong>
        <span style={{ marginLeft: 8, fontSize: 12, color: "#9ca3af" }}>quiet (no trade data)</span>
        {data.stale && <span style={{ fontSize: 11, color: "#dc2626", marginLeft: 8 }}>STALE</span>}
      </div>
    );
  }

  return (
    <div style={{ padding: "12px 16px", background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <strong style={{ color: colors.text }}>Fill Toxicity</strong>
        <span style={{
          display: "inline-block", padding: "2px 8px", borderRadius: 12,
          fontSize: 12, fontWeight: 600, color: "#fff", background: colors.badge, textTransform: "uppercase",
        }}>{level}</span>
        {card.top_side && <span style={{ fontSize: 12, color: "#6b7280" }}>bias: {card.top_side}</span>}
        {data.stale && <span style={{ fontSize: 11, color: "#dc2626", marginLeft: "auto" }}>STALE</span>}
      </div>

      {card.headline && <p style={{ margin: "0 0 6px", fontSize: 13, color: colors.text, fontWeight: 500 }}>{card.headline}</p>}
      {card.operator_note && <p style={{ margin: "0 0 8px", fontSize: 12, color: "#6b7280", fontStyle: "italic" }}>{card.operator_note}</p>}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, fontSize: 12 }}>
        <div><div style={{ color: "#9ca3af" }}>Rows</div><div style={{ fontWeight: 600 }}>{card.rows ?? "-"}</div></div>
        <div><div style={{ color: "#9ca3af" }}>Toxicity</div><div style={{ fontWeight: 600 }}>{card.toxicity_score != null ? card.toxicity_score.toFixed(3) : "-"}</div></div>
        <div><div style={{ color: "#9ca3af" }}>Adverse</div><div style={{ fontWeight: 600 }}>{fmtBps(card.adverse_bps_mean)}</div></div>
        <div><div style={{ color: "#9ca3af" }}>PnL Mean</div><div style={{ fontWeight: 600 }}>{fmtBps(card.pnl_bps_mean)}</div></div>
      </div>

      {data.recommended_action && (
        <p style={{ margin: "8px 0 0", fontSize: 11, color: "#6b7280" }}>Action: {data.recommended_action}</p>
      )}
    </div>
  );
}
