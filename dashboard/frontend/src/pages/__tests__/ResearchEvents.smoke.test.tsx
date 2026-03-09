import React from "react";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import ResearchEvents from "../ResearchEvents";

vi.mock("../../hooks/usePoll", () => ({
  usePoll: vi.fn(() => ({
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
  })),
}));

describe("ResearchEvents smoke", () => {
  it("renders research watchboard details", () => {
    render(
      <MemoryRouter initialEntries={["/research"]}>
        <ResearchEvents />
      </MemoryRouter>,
    );

    expect(screen.getByText("Research Event Watchboard")).toBeInTheDocument();
    expect(screen.getByText("Liquidation")).toBeInTheDocument();
    expect(screen.getByText("Liquidation Watchlist")).toBeInTheDocument();
  });
});
