import React from "react";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import ResearchEvents from "../ResearchEvents";

vi.mock("../../hooks/usePoll", () => ({
  usePoll: vi.fn((options?: { pollKey?: string }) => {
    if (options?.pollKey === "api:/market/chart:research") {
      return {
        data: {
          source: "binance_spot",
          symbol: "BTCUSDT",
          interval: "5m",
          limit: 240,
          generated_ts: "2026-03-09T00:00:00Z",
          candles: Array.from({ length: 60 }, (_, index) => ({
            time: 1_700_000_000 + index * 300,
            open: 50000 + index,
            high: 50010 + index,
            low: 49990 + index,
            close: 50005 + index,
            volume: 100 + index,
          })),
          overlays: [
            { name: "EMA 20", values: Array.from({ length: 60 }, (_, index) => 50000 + index) },
            { name: "EMA 50", values: Array.from({ length: 60 }, (_, index) => 49980 + index) },
          ],
          oscillator: { name: "RSI 14", values: Array.from({ length: 60 }, (_, index) => (index < 14 ? null : 45 + (index % 10))) },
          pocket_markers: [
            { time: 1700000000, bucket_time: 1700000000, side: "SELL", verdict: "GO", regime: "UP", imbalance: -0.62, trade_intensity: 4200, spread: 0.00012, score: 3.1 },
          ],
        },
        error: null,
        isLoading: false,
        isFetching: false,
        isStale: false,
        refresh: vi.fn(),
      };
    }
    return {
      data: {
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
    };
  }),
}));

describe("ResearchEvents smoke", () => {
  it("renders research watchboard details", () => {
    render(
      <MemoryRouter initialEntries={["/research"]}>
        <ResearchEvents />
      </MemoryRouter>,
    );

    expect(screen.getByText("Research Event Watchboard")).toBeInTheDocument();
    expect(screen.getByText("Market Structure Chart")).toBeInTheDocument();
    expect(screen.getByText("Research Pocket Overlay")).toBeInTheDocument();
    expect(screen.getByText("Liquidation")).toBeInTheDocument();
    expect(screen.getByText("Liquidation Watchlist")).toBeInTheDocument();
  });
});
