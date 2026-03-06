import React from "react";

export interface GlossaryTerm {
  term: string;
  meaning: string;
}

const TERMS: GlossaryTerm[] = [
  { term: "conf", meaning: "Signal confidence. Micro binary mode: mostly 0.00 or 1.00." },
  { term: "gates", meaning: "Which checks passed/failed before an entry decision." },
  { term: "reason", meaning: "Primary block/allow explanation, e.g. no_match, regime_mismatch." },
  { term: "regime", meaning: "Market state filter (UP/DOWN/other labels)." },
  { term: "stale", meaning: "Data too old for safe decisioning." },
  { term: "scratch", meaning: "Fast defensive exit when adverse movement exceeds threshold." },
  { term: "fill", meaning: "Order execution event (paper/live simulation fill)." },
  { term: "db lag", meaning: "Delay between market event and DB-read time in feature pipeline." },
];

export default function GlossaryDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.55)",
        zIndex: 1000,
        display: "flex",
        justifyContent: "flex-end",
      }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: "min(480px, 92vw)",
          height: "100%",
          background: "var(--surface)",
          borderLeft: "1px solid var(--border)",
          padding: 16,
          overflowY: "auto",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <div className="card-title" style={{ marginBottom: 0 }}>Glossary</div>
          <button
            onClick={onClose}
            style={{
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              borderRadius: 4,
              padding: "4px 8px",
              cursor: "pointer",
            }}
          >
            Close
          </button>
        </div>
        <div style={{ color: "var(--muted)", fontSize: 12, marginBottom: 10 }}>
          Quick meanings for runtime terms shown across Overview, Logs, Trades, and Debug.
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {TERMS.map((item) => (
            <div key={item.term} className="card" style={{ padding: 12 }}>
              <div style={{ fontWeight: 700, marginBottom: 4 }}>{item.term}</div>
              <div style={{ color: "var(--muted)", fontSize: 12 }}>{item.meaning}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

