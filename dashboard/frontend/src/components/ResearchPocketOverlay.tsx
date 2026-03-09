import React from "react";
import type { RegimeEvent } from "../api/types";
import type { MarketChartSymbol } from "./MarketStructureChart";

function badge(label: string, tone: "green" | "yellow" | "red" | "gray" | "blue" = "gray") {
  const cls =
    tone === "green" ? "badge badge-green" :
    tone === "yellow" ? "badge badge-yellow" :
    tone === "red" ? "badge badge-red" :
    tone === "blue" ? "badge badge-blue" :
    "badge badge-gray";
  return <span className={cls}>{label}</span>;
}

function normalizeRegime(value?: string | null): "UP" | "DOWN" | "UNKNOWN" {
  const raw = String(value || "").toUpperCase();
  if (raw.includes("UP")) return "UP";
  if (raw.includes("DOWN")) return "DOWN";
  return "UNKNOWN";
}

function verdictTone(verdict: string): "green" | "yellow" | "red" | "gray" {
  if (verdict === "GO") return "green";
  if (verdict === "MARGINAL") return "yellow";
  if (verdict === "NO-GO") return "red";
  return "gray";
}

export default function ResearchPocketOverlay({
  symbol,
  recentRegimes,
}: {
  symbol: MarketChartSymbol;
  recentRegimes: RegimeEvent[];
}) {
  const latestRegime = recentRegimes.find((row) => String(row.symbol || "").toUpperCase() === symbol);
  const regime = normalizeRegime(latestRegime?.effective_regime || latestRegime?.raw_regime);

  if (symbol !== "ETHUSDT") {
    return (
      <div className="card" style={{ borderLeft: "3px solid var(--muted)" }}>
        <div className="card-title">Research Pocket Overlay</div>
        <div style={{ color: "var(--muted)", marginBottom: 10 }}>
          No promoted pocket overlay configured for {symbol} yet.
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {badge(`symbol=${symbol}`, "blue")}
          {badge("overlay=none")}
        </div>
      </div>
    );
  }

  const verdictByRegime = {
    UP: [
      { side: "SELL", verdict: "GO", note: "pass=55.6%, NPA=7.09e-05" },
      { side: "BUY", verdict: "GO", note: "pass=50.0%, NPA=5.28e-05" },
    ],
    DOWN: [
      { side: "SELL", verdict: "NO-GO", note: "all negative NPA" },
      { side: "BUY", verdict: "MARGINAL", note: "only near fee=0" },
    ],
    UNKNOWN: [
      { side: "SELL", verdict: "WAIT", note: "regime unresolved" },
      { side: "BUY", verdict: "WAIT", note: "regime unresolved" },
    ],
  } as const;

  const rows = verdictByRegime[regime];

  return (
    <div className="card" style={{ borderLeft: "3px solid var(--accent)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
        <div>
          <div className="card-title" style={{ marginBottom: 6 }}>Research Pocket Overlay</div>
          <div style={{ color: "var(--muted)" }}>
            ETH alpha pocket, h=120s. Research verdict overlay, not direct execution signaling.
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {badge("ETHUSDT", "blue")}
          {badge("h=120s", "gray")}
          {badge(`regime=${regime}`, regime === "UP" ? "green" : regime === "DOWN" ? "red" : "gray")}
        </div>
      </div>

      <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
        <div style={{ padding: "10px 12px", borderRadius: 8, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
          <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Pocket Rule</div>
          <div style={{ marginTop: 8, fontFamily: "monospace", fontSize: 12, lineHeight: 1.55 }}>
            <div>|imbalance| &gt;= 0.5</div>
            <div>trade_intensity &gt;= 3500</div>
            <div>spread &lt;= 0.0003</div>
          </div>
        </div>
        <div style={{ padding: "10px 12px", borderRadius: 8, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
          <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Pocket Stats</div>
          <div style={{ marginTop: 8, color: "var(--text)", lineHeight: 1.7, fontSize: 13 }}>
            <div>SELL: touch 76.6%, fill 39.7%, hit 55.5%</div>
            <div>BUY: touch 78.9%, fill 38.3%, hit 56.4%</div>
            <div>Break-even maker fee: ~0.8 bps/leg</div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: 14 }}>
        <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.04em", textTransform: "uppercase", marginBottom: 8 }}>
          Regime Verdict
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
          {rows.map((row) => (
            <div key={row.side} style={{ padding: "10px 12px", borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface-2)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}>
                <div style={{ fontWeight: 700 }}>{row.side}</div>
                {badge(row.verdict, verdictTone(row.verdict))}
              </div>
              <div style={{ color: "var(--muted)", marginTop: 6, fontSize: 12 }}>{row.note}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
