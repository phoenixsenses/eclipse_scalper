import React from "react";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import Overview from "../Overview";

vi.mock("../../hooks/usePoll", () => ({
  usePoll: vi
    .fn()
    // overview poll
    .mockImplementationOnce(() => ({
      data: {
        scoreboard: {
          paper_trading: true,
          fills_total: 1,
          orders_total: 2,
          blocked_total: 3,
          kill_switch_trips_total: 0,
          circuit_breaker_trips_total: 0,
          blocked_by_reason: { no_match: 10, regime_mismatch: 2 },
        },
        gates: { symbols: [] },
        recent_regimes: [],
        preflight: {},
        reliability: {},
        health_overall: {
          data_research_fitness_status: "warning",
          data_research_fitness_connected: true,
          data_research_fitness_detail: "fitness_status=warn warnings=1 failures=0",
          paper_execution_mode: "router_blocked",
          startup_contract_safe: true,
          startup_contract_reason: "",
          binance_testnet: true,
          paper_allow_live_private_api: false,
        },
        research_events: {
          watchboard: {
            banner: { headline: "Research watchboard banner", detail: "top event stale" },
            summary: { top_lane: "liquidation", state_counts: { severe: 2, quiet: 2 } },
            lanes: [
              {
                lane: "liquidation",
                recommended_action: "monitor_only",
                dashboard_summary: "historical context",
                state: { level: "severe", freshness: { status: "stale" } },
              },
            ],
          },
          states: {
            liquidation: {
              state: { level: "elevated", freshness: { status: "fresh" }, primary_side_bias: "sell" },
              card: { headline: "Liquidation context", recent_alert_count: 3, tagged_rate: 0.04, max_consecutive_tagged: 2, max_liq_rate_recent: 11.4 },
              dashboard_summary: "liquidation lane",
              recommended_action: "monitor_only",
            },
          },
          watchlists: {
            liquidation: {
              banner: { headline: "Top liq watch", detail: "ETHUSDT elevated" },
              top_summary: { symbol: "ETHUSDT", state_level: "elevated", freshness_status: "fresh", recommended_action: "monitor_only" },
              rows: [{ symbol: "ETHUSDT", state_level: "elevated", priority_score: 1.2 }],
            },
          },
        },
      },
      error: null,
      isLoading: false,
      isFetching: false,
      isStale: false,
      refresh: vi.fn(),
    }))
    // runtime poll
    .mockImplementation(() => ({
      data: {
        collector: { alive: true, trades_per_sec_60s: 1 },
        database: { size_bytes: 123 },
        data_freshness: { status: "LIVE", seconds_since_last_trade: 1 },
        system: {},
      },
      error: null,
      isLoading: false,
      isFetching: false,
      isStale: false,
      refresh: vi.fn(),
    })),
}));

describe("Overview smoke", () => {
  it("renders top blockers card", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Overview />
      </MemoryRouter>
    );

    expect(screen.getByText("Top Blockers (reason)")).toBeInTheDocument();
    expect(screen.getAllByText("no_match").length).toBeGreaterThan(0);
    expect(screen.getByText("Data Research Fitness")).toBeInTheDocument();
    expect(screen.getByText("fitness_status=warn warnings=1 failures=0")).toBeInTheDocument();
    expect(screen.getByText("Paper Execution Contract")).toBeInTheDocument();
    expect(screen.getAllByText("No-fill rehearsal").length).toBeGreaterThan(0);
    expect(screen.getByText(/SAFE CONTRACT/i)).toBeInTheDocument();
    expect(screen.getByText("Research Event Watchboard")).toBeInTheDocument();
    expect(screen.getByText("Open Research Events")).toBeInTheDocument();
  });
});

