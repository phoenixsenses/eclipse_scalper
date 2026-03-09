import React from "react";
import type { WatchboardState } from "../api/types";

const LEVEL_COLORS: Record<string, { bg: string; border: string; text: string; badge: string }> = {
  quiet: { bg: "#f0fdf4", border: "#86efac", text: "#166534", badge: "#22c55e" },
  elevated: { bg: "#fffbeb", border: "#fcd34d", text: "#92400e", badge: "#f59e0b" },
  severe: { bg: "#fef2f2", border: "#fca5a5", text: "#991b1b", badge: "#ef4444" },
};

function levelBadge(level: string) {
  const colors = LEVEL_COLORS[level] || LEVEL_COLORS.quiet;
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 12,
      fontSize: 11, fontWeight: 600, color: "#fff", background: colors.badge, textTransform: "uppercase",
    }}>{level}</span>
  );
}

export default function WatchboardCard({ data }: { data: WatchboardState | null }) {
  if (!data || !data.available) {
    return (
      <div style={{ padding: "12px 16px", background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 8, opacity: 0.6 }}>
        <strong>Research Event Watchboard</strong>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: "#6b7280" }}>No data available</p>
      </div>
    );
  }

  const topLevel = data.banner?.top_level || data.top_event?.level || "quiet";
  const colors = LEVEL_COLORS[topLevel] || LEVEL_COLORS.quiet;
  const summary = data.summary;
  const stateCounts = summary.state_counts || {};

  return (
    <div style={{ padding: "12px 16px", background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <strong style={{ color: colors.text }}>Research Event Watchboard</strong>
        {levelBadge(topLevel)}
        <span style={{ fontSize: 12, color: "#6b7280" }}>{summary.lane_count} lanes</span>
        {data.stale && <span style={{ fontSize: 11, color: "#dc2626", marginLeft: "auto" }}>STALE</span>}
      </div>

      {/* Banner */}
      {data.banner?.headline && (
        <div style={{ marginBottom: 8, padding: "6px 10px", background: "rgba(0,0,0,0.04)", borderRadius: 6, fontSize: 12 }}>
          <strong>{data.banner.headline}</strong>
          {data.banner.recommended_action && (
            <span style={{ marginLeft: 8, color: "#6b7280" }}>({data.banner.recommended_action})</span>
          )}
        </div>
      )}

      {/* State counts */}
      <div style={{ display: "flex", gap: 12, marginBottom: 8, fontSize: 12 }}>
        {Object.entries(stateCounts).map(([level, count]) => (
          <span key={level} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            {levelBadge(level)} <span>{count as number}</span>
          </span>
        ))}
      </div>

      {/* Top event detail */}
      {data.top_event && (
        <div style={{ marginBottom: 8, fontSize: 12 }}>
          <span style={{ color: "#6b7280" }}>Top:</span>{" "}
          <strong>{data.top_event.lane}</strong>{" "}
          {data.top_event.headline && <span>- {data.top_event.headline}</span>}
        </div>
      )}

      {/* Lane table */}
      {data.lanes.length > 0 && (
        <table style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(0,0,0,0.1)" }}>
              <th style={{ textAlign: "left", padding: "3px 4px" }}>Lane</th>
              <th style={{ textAlign: "left", padding: "3px 4px" }}>Level</th>
              <th style={{ textAlign: "left", padding: "3px 4px" }}>Fresh</th>
              <th style={{ textAlign: "left", padding: "3px 4px" }}>Action</th>
              <th style={{ textAlign: "left", padding: "3px 4px" }}>Headline</th>
            </tr>
          </thead>
          <tbody>
            {data.lanes.map((l, i) => {
              const lc = LEVEL_COLORS[l.level || "quiet"] || LEVEL_COLORS.quiet;
              return (
                <tr key={i} style={{ borderBottom: "1px solid rgba(0,0,0,0.05)" }}>
                  <td style={{ padding: "3px 4px", fontWeight: 500 }}>{l.lane}</td>
                  <td style={{ padding: "3px 4px" }}>{levelBadge(l.level || "quiet")}</td>
                  <td style={{ padding: "3px 4px", color: l.freshness_status === "stale" ? "#dc2626" : "#16a34a" }}>
                    {l.freshness_status || "-"}
                  </td>
                  <td style={{ padding: "3px 4px", color: "#6b7280" }}>{l.recommended_action || "-"}</td>
                  <td style={{ padding: "3px 4px", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {l.headline || "-"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
