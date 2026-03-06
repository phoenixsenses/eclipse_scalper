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
    expect(screen.getByText("Paper Trade Live Summary")).toBeInTheDocument();
    expect(screen.getByText("Last 5 Fills")).toBeInTheDocument();
    expect(screen.getByText("Mini Trends")).toBeInTheDocument();
  });
});

