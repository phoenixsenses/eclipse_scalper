import React from "react";

interface TermTipProps {
  term: string;
  tr: string;
  en: string;
}

export default function TermTip({ term, tr, en }: TermTipProps) {
  return (
    <span
      title={`${tr}\n${en}`}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        cursor: "help",
      }}
      aria-label={`${term} explanation`}
    >
      <span>{term}</span>
      <span
        style={{
          color: "var(--info)",
          fontSize: 11,
          border: "1px solid var(--border)",
          borderRadius: 999,
          padding: "0 5px",
          lineHeight: "14px",
          background: "#102338",
        }}
      >
        ?
      </span>
    </span>
  );
}
