import React, { useCallback } from "react";
import { api } from "../api/client";
import type { OverviewResponse } from "../api/types";
import AsyncState from "../components/AsyncState";
import PageGuide from "../components/PageGuide";
import ResearchEventsPanel from "../components/ResearchEventsPanel";
import { usePoll } from "../hooks/usePoll";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

export default function ResearchEvents() {
  const fetchOverview = useCallback((signal: AbortSignal) => api.overview(signal), []);
  const overviewPoll = usePoll<OverviewResponse>({
    fetcher: fetchOverview,
    pollKey: "api:/overview",
    intervalMs: 10_000,
    staleAfterMs: 30_000,
  });

  const researchEvents = asRecord(overviewPoll.data?.research_events);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <PageGuide
        icon="R"
        titleTr="Arastirma Olaylari"
        titleEn="Research Events"
        subtitleTr="Gunluk rapor ve lane monitorlerinden gelen deneysel olay sinyallerini burada takip et."
        subtitleEn="Track experimental event lanes sourced from the daily research report and monitor outputs."
        items={[
          {
            icon: "1",
            titleTr: "Watchboard",
            titleEn: "Watchboard",
            descTr: "Tum lane durumlarini tek tabloda gosterir.",
            descEn: "Shows all lane states in a single watchboard table.",
          },
          {
            icon: "2",
            titleTr: "State Cards",
            titleEn: "State Cards",
            descTr: "Tekil lane sinyallerinin seviye, tazelik ve operator notlarini ozetler.",
            descEn: "Summarizes level, freshness, and operator notes for single-lane signals.",
          },
          {
            icon: "3",
            titleTr: "Watchlists",
            titleEn: "Watchlists",
            descTr: "Coklu sembol lane listelerini ust oncelik ile gosterir.",
            descEn: "Highlights multi-symbol lane watchlists with top priorities first.",
          },
        ]}
      />

      <AsyncState
        loading={overviewPoll.isLoading}
        error={overviewPoll.error}
        isEmpty={Object.keys(researchEvents).length === 0}
        emptyText="No research event payload available"
      >
        <ResearchEventsPanel researchEvents={researchEvents} />
      </AsyncState>
    </div>
  );
}
