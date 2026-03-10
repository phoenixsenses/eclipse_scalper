import React from "react";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import LiveMonitor from "../LiveMonitor";

vi.mock("../../hooks/usePoll", () => ({
  usePoll: vi.fn((cfg: { pollKey?: string }) => {
    const key = cfg?.pollKey ?? "";
    if (key.includes("api:/runtime:live-monitor")) {
      return {
        data: {
          collector: { alive: true, trades_per_sec_60s: 12, mark_per_sec_60s: 2, uptime_sec: 1000 },
          database: { size_bytes: 1024, last_write_ts: "2026-03-05T00:00:00+00:00" },
          data_freshness: { status: "LIVE", seconds_since_last_trade: 1.0 },
          system: {},
        },
        error: null, isLoading: false, isFetching: false, isStale: false, refresh: vi.fn(),
      };
    }
    if (key.includes("api:/ops/health:live-monitor")) {
      return {
        data: { status: { network: "ok" }, network: { reconnects_last_5m: 0, usage_pct: 12.5 } },
        error: null, isLoading: false, isFetching: false, isStale: false, refresh: vi.fn(),
      };
    }
    if (key.includes("api:/logs:live-monitor")) {
      return {
        data: [{ name: "microstructure_collector.log" }, { name: "paper_trades.jsonl" }],
        error: null, isLoading: false, isFetching: false, isStale: false, refresh: vi.fn(),
      };
    }
    if (key.includes("api:/live/metrics")) {
      return {
        data: {
          ts_utc: "2026-03-05T00:00:00+00:00",
          runtime: {
            collector: { alive: true, trades_per_sec_60s: 12 },
            database: { size_bytes: 2048 },
            data_freshness: { status: "LIVE", seconds_since_last_trade: 1 },
            system: {},
          },
          scoreboard: { paper_trading: true, orders_total: 10, fills_total: 4, blocked_total: 2, blocked_by_reason: { no_match: 2 } },
          pnl_strip: { today: 1.2, h24: 2.3, d7: 3.4, sample: 5 },
          fill_quality: { avg_delay_ms: 120, avg_adverse_bps: 0.2, with_delay: 5, with_adverse: 5 },
          tail_kpis: { window_lines: 80, order_count: 20, fill_count: 8, blocked_count: 1, fill_per_order_pct: 40 },
          blocked_reasons: [{ reason: "no_match", count: 2 }],
          last_fills: [{ ts: "2026-03-05T00:00:00+00:00", symbol: "ETHUSDT", side: "buy", price: "2100", qty: "0.01", pnl: "0.5" }],
          alerts: { any_alert: false, trade_age_alert: false, fill_flatline_alert: false, trade_age_sec: 1, fill_age_min: 0.1, config: { trade_age_alert_sec: 10, fill_flatline_alert_min: 15 } },
          trends: { trades_per_sec: [1, 2, 3], fills_tail: [0, 1, 1] },
          paper_file: "paper_trades.jsonl",
        },
        error: null, isLoading: false, isFetching: false, isStale: false, refresh: vi.fn(),
      };
    }
    if (key.includes("api:/live/paper-run")) {
      return {
        data: {
          ts_utc: "2026-03-05T00:00:00+00:00",
          session: {
            status: "running",
            started_ts: "2026-03-05T00:00:00+00:00",
            uptime_sec: 300,
            active_symbols: ["ETHUSDT", "BTCUSDT"],
            telemetry_age_sec: 1,
            telemetry_present: true,
          },
          process_chain: {
            launcher_present: true,
            watchdog_present: true,
            bootstrap_present: true,
            launcher_pid: 111,
            watchdog_pids: [222],
            bootstrap_pids: [333],
            summary: "launcher pid=111, watchdog=1, bootstrap=1",
          },
          entry_state: {
            allow_entries: true,
            guard_mode: "GREEN",
            runtime_gate_degraded: false,
            runtime_gate_reason: "missing",
            data_state: "ok",
            risk_state: "ok",
            regime_state: "ok",
          },
          trade_state: {
            trade_count: 0,
            last_trade_ts: null,
            no_trades_yet: true,
            db_present: true,
            db_path: "data/paper_trades.db",
          },
          diagnosis: {
            code: "signal_not_present",
            summary: "paper run is healthy but no signal is present",
            detail: "entries are allowed; recent blockers show signal not present",
            severity: "info",
          },
          reason_breakdown: {
            signal_not_present: 2,
            gate_blocked: 0,
            data_degraded: 0,
            risk_blocked: 0,
            regime_blocked: 0,
            unknown: 0,
          },
          symbols: [
            { symbol: "ETHUSDT", last_blocker_reason: "signal not present", recent_blocked_count: 2, last_signal_ts: null, last_belief_ts: "2026-03-05T00:00:00+00:00" },
            { symbol: "BTCUSDT", last_blocker_reason: null, recent_blocked_count: 0, last_signal_ts: null, last_belief_ts: "2026-03-05T00:00:00+00:00" },
          ],
        },
        error: null, isLoading: false, isFetching: false, isStale: false, refresh: vi.fn(),
      };
    }
    return {
      data: { file: "paper_trades.jsonl", lines: ["fill_price=2100 qty=0.01 pnl_bps=0.5"] },
      error: null, isLoading: false, isFetching: false, isStale: false, refresh: vi.fn(),
    };
  }),
}));

describe("LiveMonitor smoke", () => {
  it("renders live monitor key sections", () => {
    render(
      <MemoryRouter initialEntries={["/live"]}>
        <LiveMonitor />
      </MemoryRouter>,
    );

    expect(screen.getByText("Live Alerts")).toBeInTheDocument();
    expect(screen.getByText("Paper Run Diagnosis")).toBeInTheDocument();
    expect(screen.getByText("Why No Trade?")).toBeInTheDocument();
    expect(screen.getByText("Symbol Diagnostics")).toBeInTheDocument();
    expect(screen.getByText("Paper Trade Live Summary")).toBeInTheDocument();
    expect(screen.getByText("Last 5 Fills")).toBeInTheDocument();
    expect(screen.getByText("Mini Trends")).toBeInTheDocument();
  });
});

