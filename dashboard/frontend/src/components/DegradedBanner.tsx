import React from "react";

export type DegradedMode = "ok" | "degraded" | "down" | "recovered";

interface DegradedBannerProps {
  mode: DegradedMode;
  message?: string;
}

export default function DegradedBanner({ mode, message }: DegradedBannerProps) {
  if (mode === "ok") {
    return null;
  }

  const color =
    mode === "down" ? "var(--red)" :
    mode === "degraded" ? "var(--yellow)" :
    "var(--green)";
  const label =
    mode === "down" ? "Backend unavailable" :
    mode === "degraded" ? "Data stale or reconnecting" :
    "Connection recovered";

  return (
    <div
      className="card"
      style={{
        borderLeft: `3px solid ${color}`,
        padding: "8px 12px",
        display: "flex",
        gap: 10,
        alignItems: "center",
      }}
    >
      <span style={{ color, fontWeight: 700 }}>{label}</span>
      {message ? <span style={{ color: "var(--muted)", fontSize: 12 }}>{message}</span> : null}
    </div>
  );
}
