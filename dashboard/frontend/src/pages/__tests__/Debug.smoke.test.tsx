import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import Debug from "../Debug";

vi.mock("../../hooks/usePoll", () => ({
  usePoll: vi
    .fn()
    // actions poll
    .mockImplementationOnce(() => ({
      data: [
        { action: "validate_env", description: "x", timeout_sec: 10 },
        { action: "preflight_check", description: "x", timeout_sec: 10 },
        { action: "paper_trade_status", description: "x", timeout_sec: 10 },
        { action: "incident_bundle", description: "x", timeout_sec: 10 },
      ],
      error: null,
      isLoading: false,
      isFetching: false,
      isStale: false,
      refresh: vi.fn(),
    }))
    // history poll
    .mockImplementation(() => ({
      data: [],
      error: null,
      isLoading: false,
      isFetching: false,
      isStale: false,
      refresh: vi.fn(),
    })),
}));

vi.mock("../../api/client", () => ({
  ApiError: class extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
  api: {
    runDebugAction: vi.fn().mockResolvedValue({
      action: "validate_env",
      ok: true,
      exit_code: 0,
      duration_sec: 0.1,
      output: "ok",
      started_ts: 1,
      ended_ts: 2,
    }),
    runDebugRunbook: vi.fn().mockResolvedValue({
      session_id: "session_1",
      started_ts: 1,
      ended_ts: 2,
      duration_sec: 1,
      ok: true,
      failed_action: null,
      steps: [],
      incident_hint: null,
    }),
    debugSessionDetail: vi.fn().mockResolvedValue({
      session_id: "session_1",
      started_ts: 1,
      ended_ts: 2,
      duration_sec: 1,
      ok: true,
      failed_action: null,
      steps: [],
      incident_hint: null,
    }),
  },
}));

describe("Debug smoke", () => {
  it("renders guided controls and runs action", async () => {
    render(
      <MemoryRouter initialEntries={["/debug"]}>
        <Debug />
      </MemoryRouter>
    );

    expect(screen.getByText("Guided Debug Session")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Run Guided Session"));

    await waitFor(() => {
      expect(screen.getByText("Last Action Output")).toBeInTheDocument();
    });
  });
});
