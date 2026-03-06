import React, { useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

interface TourStep {
  id: string;
  title: string;
  body: string;
  path: string;
}

const TOUR_DONE_KEY = "eclipse.ui.onboarding.done.v1";
const TOUR_INDEX_KEY = "eclipse.ui.onboarding.index.v1";

const STEPS: TourStep[] = [
  {
    id: "overview",
    title: "Overview",
    body: "Start here for runtime health, freshness, and top blockers.",
    path: "/",
  },
  {
    id: "logs",
    title: "Logs",
    body: "Select a file, then use presets/search to isolate root cause quickly.",
    path: "/logs",
  },
  {
    id: "debug",
    title: "Debug",
    body: "Run guided diagnostics, inspect incident summary, and export session artifacts.",
    path: "/debug",
  },
  {
    id: "trades",
    title: "Trades",
    body: "Inspect signal/stability/quality events and filter by symbol/text.",
    path: "/trades",
  },
];

function loadInitialState(): { open: boolean; idx: number } {
  try {
    const done = localStorage.getItem(TOUR_DONE_KEY);
    if (done === "1") return { open: false, idx: 0 };
    const rawIdx = Number(localStorage.getItem(TOUR_INDEX_KEY) ?? 0);
    const idx = Number.isFinite(rawIdx) ? Math.max(0, Math.min(STEPS.length - 1, rawIdx)) : 0;
    return { open: true, idx };
  } catch {
    return { open: true, idx: 0 };
  }
}

export default function OnboardingTour() {
  const location = useLocation();
  const navigate = useNavigate();
  const initial = useMemo(loadInitialState, []);
  const [open, setOpen] = useState(initial.open);
  const [idx, setIdx] = useState(initial.idx);

  if (!open) return null;

  const step = STEPS[idx];
  const isLast = idx >= STEPS.length - 1;
  const isFirst = idx <= 0;
  const onStepPage = location.pathname === step.path;

  function persistIndex(next: number) {
    setIdx(next);
    try {
      localStorage.setItem(TOUR_INDEX_KEY, String(next));
    } catch {
      // no-op
    }
  }

  function finish(disable: boolean) {
    setOpen(false);
    try {
      localStorage.setItem(TOUR_DONE_KEY, disable ? "1" : "0");
      if (!disable) {
        localStorage.setItem(TOUR_INDEX_KEY, "0");
      }
    } catch {
      // no-op
    }
  }

  return (
    <div
      style={{
        position: "fixed",
        right: 16,
        bottom: 16,
        width: "min(420px, calc(100vw - 32px))",
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: 14,
        zIndex: 1100,
        boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div style={{ fontWeight: 700 }}>Quick Start Tour</div>
        <span style={{ color: "var(--muted)", fontSize: 12 }}>
          {idx + 1}/{STEPS.length}
        </span>
      </div>
      <div style={{ fontSize: 13, marginBottom: 6 }}>{step.title}</div>
      <div style={{ color: "var(--muted)", fontSize: 12, marginBottom: 10 }}>{step.body}</div>
      <div style={{ color: onStepPage ? "var(--green)" : "var(--yellow)", fontSize: 11, marginBottom: 10 }}>
        {onStepPage ? "You are on this step page." : `Open ${step.path} to continue.`}
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button
          onClick={() => navigate(step.path)}
          style={{
            padding: "5px 8px",
            borderRadius: 4,
            border: "1px solid var(--accent)",
            background: "transparent",
            color: "var(--accent)",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          Go to step
        </button>
        <button
          disabled={isFirst}
          onClick={() => persistIndex(Math.max(0, idx - 1))}
          style={{
            padding: "5px 8px",
            borderRadius: 4,
            border: "1px solid var(--border)",
            background: "transparent",
            color: isFirst ? "var(--muted)" : "var(--text)",
            cursor: isFirst ? "not-allowed" : "pointer",
            fontSize: 12,
          }}
        >
          Prev
        </button>
        <button
          onClick={() => {
            if (isLast) {
              finish(false);
            } else {
              persistIndex(Math.min(STEPS.length - 1, idx + 1));
            }
          }}
          style={{
            padding: "5px 8px",
            borderRadius: 4,
            border: "1px solid var(--border)",
            background: "transparent",
            color: "var(--text)",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          {isLast ? "Done" : "Next"}
        </button>
        <button
          onClick={() => finish(true)}
          style={{
            marginLeft: "auto",
            padding: "5px 8px",
            borderRadius: 4,
            border: "1px solid var(--border)",
            background: "transparent",
            color: "var(--muted)",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          Don't show again
        </button>
      </div>
    </div>
  );
}

