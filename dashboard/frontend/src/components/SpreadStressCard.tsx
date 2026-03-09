import React, { useState } from "react";
import type { SpreadStressState } from "../api/types";

const LEVEL_COLORS: Record<string, { bg: string; border: string; text: string; badge: string }> = {
  quiet: { bg: "#f0fdf4", border: "#86efac", text: "#166534", badge: "#22c55e" },
  elevated: { bg: "#fffbeb", border: "#fcd34d", text: "#92400e", badge: "#f59e0b" },
  severe: { bg: "#fef2f2", border: "#fca5a5", text: "#991b1b", badge: "#ef4444" },
};

export default function SpreadStressCard({ data }: { data: SpreadStressState | null }) {
  const [showWatchlist, setShowWatchlist] = useState(false);

  if (!data || !data.available) {
    return (
      <div style={{ padding: "12px 16px", background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 8, opacity: 0.6 }}>
        <strong>Spread Stress</strong>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: "#6b7280" }}>No data available</p>
      </div>
    );
  }

  const level = data.state?.level || "quiet";
  const colors = LEVEL_COLORS[level] || LEVEL_COLORS.quiet;
  const card = data.card || {};
  const wl = data.watchlist;
  const banner = wl?.banner;

  return (
    <div style={{ padding: "12px 16px", background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <strong style={{ color: colors.text }}>Spread Stress</strong>
        <span style={{
          display: "inline-block", padding: "2px 8px", borderRadius: 12,
          fontSize: 12, fontWeight: 600, color: "#fff", background: colors.badge, textTransform: "uppercase",
        }}>{level}</span>
        {data.symbol && <span style={{ fontSize: 12, color: "#6b7280" }}>{data.symbol}</span>}
        {data.stale && <span style={{ fontSize: 11, color: "#dc2626", marginLeft: "auto" }}>STALE</span>}
      </div>

      {card.headline && <p style={{ margin: "0 0 6px", fontSize: 13, color: colors.text, fontWeight: 500 }}>{card.headline}</p>}
      {card.operator_note && <p style={{ margin: "0 0 8px", fontSize: 12, color: "#6b7280", fontStyle: "italic" }}>{card.operator_note}</p>}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, fontSize: 12 }}>
        <div><div style={{ color: "#9ca3af" }}>Alerts</div><div style={{ fontWeight: 600 }}>{card.recent_alert_count ?? "-"}</div></div>
        <div><div style={{ color: "#9ca3af" }}>High</div><div style={{ fontWeight: 600 }}>{card.high_count ?? "-"}</div></div>
        <div><div style={{ color: "#9ca3af" }}>Medium</div><div style={{ fontWeight: 600 }}>{card.medium_count ?? "-"}</div></div>
        <div><div style={{ color: "#9ca3af" }}>Avg Spread</div><div style={{ fontWeight: 600 }}>{card.avg_spread_tagged != null ? card.avg_spread_tagged.toFixed(6) : "-"}</div></div>
      </div>

      {data.recommended_action && (
        <p style={{ margin: "8px 0 0", fontSize: 11, color: "#6b7280" }}>Action: {data.recommended_action}</p>
      )}

      {banner?.headline && (
        <div style={{ marginTop: 8, padding: "6px 10px", background: "rgba(0,0,0,0.04)", borderRadius: 6, fontSize: 12 }}>
          <strong>Banner:</strong> {banner.headline}
        </div>
      )}

      {wl?.rows && wl.rows.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <button
            onClick={() => setShowWatchlist(!showWatchlist)}
            style={{ fontSize: 12, color: colors.text, background: "none", border: "none", cursor: "pointer", textDecoration: "underline", padding: 0 }}
          >
            {showWatchlist ? "Hide" : "Show"} watchlist ({wl.rows.length} symbols)
          </button>
          {showWatchlist && (
            <table style={{ width: "100%", fontSize: 11, marginTop: 6, borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid rgba(0,0,0,0.1)" }}>
                  <th style={{ textAlign: "left", padding: "2px 4px" }}>Symbol</th>
                  <th style={{ textAlign: "left", padding: "2px 4px" }}>Level</th>
                  <th style={{ textAlign: "right", padding: "2px 4px" }}>Alerts</th>
                  <th style={{ textAlign: "right", padding: "2px 4px" }}>High</th>
                  <th style={{ textAlign: "right", padding: "2px 4px" }}>Spread</th>
                </tr>
              </thead>
              <tbody>
                {wl.rows.map((r, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid rgba(0,0,0,0.05)" }}>
                    <td style={{ padding: "2px 4px" }}>{r.symbol}</td>
                    <td style={{ padding: "2px 4px" }}>{r.state_level}</td>
                    <td style={{ padding: "2px 4px", textAlign: "right" }}>{r.recent_alert_count ?? "-"}</td>
                    <td style={{ padding: "2px 4px", textAlign: "right" }}>{r.high_count ?? "-"}</td>
                    <td style={{ padding: "2px 4px", textAlign: "right" }}>{r.avg_spread_tagged?.toFixed(6) ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
