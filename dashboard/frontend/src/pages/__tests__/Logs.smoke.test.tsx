import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import Logs from "../Logs";

vi.mock("../../hooks/useSSE", () => ({
  useSSE: () => ({ status: "closed", error: null, isStale: false }),
}));

vi.mock("../../hooks/usePoll", () => {
  return {
    usePoll: vi
      .fn()
      // files poll
      .mockImplementationOnce(() => ({
        data: [{ name: "paper_trading.log", path: "logs/paper_trading.log", size_bytes: 100, mtime: 1 }],
        error: null,
        isLoading: false,
        isFetching: false,
        isStale: false,
        refresh: vi.fn(),
      }))
      // tail poll
      .mockImplementation(() => ({
        data: null,
        error: null,
        isLoading: false,
        isFetching: false,
        isStale: false,
        refresh: vi.fn(),
      })),
  };
});

vi.mock("../../api/client", () => ({
  api: {
    logTail: vi.fn().mockResolvedValue({ file: "paper_trading.log", lines: ["line a", "line b"] }),
  },
  streamLog: vi.fn(),
}));

describe("Logs smoke", () => {
  it("loads selected file and supports preset save/delete", async () => {
    render(
      <MemoryRouter initialEntries={["/logs"]}>
        <Logs />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByText("paper_trading.log"));

    await waitFor(() => {
      expect(screen.getByText(/showing/i)).toBeInTheDocument();
    });

    fireEvent.change(screen.getByPlaceholderText("Preset name"), { target: { value: "MyPreset" } });
    fireEvent.click(screen.getByText("Save Preset"));
    expect(screen.getByText("MyPreset")).toBeInTheDocument();

    fireEvent.click(screen.getByTitle("Delete preset"));
    expect(screen.queryByText("MyPreset")).not.toBeInTheDocument();
  });
});

