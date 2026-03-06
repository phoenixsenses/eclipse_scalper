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
  });
});

