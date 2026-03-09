import React, { useCallback, useMemo, useState } from "react";
import { api } from "../api/client";
import type { MarketChartResponse } from "../api/types";
import { usePoll } from "../hooks/usePoll";

const SYMBOLS = ["BTCUSDT", "ETHUSDT"] as const;
const INTERVALS = ["1m", "5m", "15m", "1h"] as const;
export type MarketChartSymbol = (typeof SYMBOLS)[number];
export type MarketChartInterval = (typeof INTERVALS)[number];

type ChartPoint = { x: number; y: number };
type ChartGeometry = {
  width: number;
  height: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
  priceHeight: number;
  gap: number;
  rsiHeight: number;
  rsiTop: number;
  priceMin: number;
  priceMax: number;
  stepX: number;
  candleWidth: number;
  candles: MarketChartResponse["candles"];
  priceY: (value: number) => number;
  rsiY: (value: number) => number;
  overlayPaths: Array<{ name: string; path: string }>;
  rsiPath: string;
};

function pathFromPoints(points: ChartPoint[]): string {
  if (points.length === 0) return "";
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
}

function formatCompact(value: number | null | undefined, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function asArray<T = Record<string, unknown>>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function titleizeLane(name: string): string {
  return name
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function levelRank(level?: string | null): number {
  const normalized = String(level || "quiet").toLowerCase();
  if (normalized === "severe" || normalized === "high") return 0;
  if (normalized === "elevated" || normalized === "medium") return 1;
  if (normalized === "quiet" || normalized === "none") return 2;
  return 3;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function laneStrength(level: string, stale: boolean): number {
  const normalized = level.toLowerCase();
  const base =
    normalized === "severe" || normalized === "high"
      ? 1
      : normalized === "elevated" || normalized === "medium"
        ? 0.65
        : 0.3;
  return stale ? base * 0.55 : base;
}

export default function MarketStructureChart({
  symbol,
  interval,
  onSymbolChange,
  onIntervalChange,
  researchEvents,
}: {
  symbol: MarketChartSymbol;
  interval: MarketChartInterval;
  onSymbolChange: (symbol: MarketChartSymbol) => void;
  onIntervalChange: (interval: MarketChartInterval) => void;
  researchEvents?: Record<string, unknown>;
}) {
  const [viewportWidth, setViewportWidth] = useState<number>(() =>
    typeof window === "undefined" ? 1440 : window.innerWidth,
  );
  const [hitFilter, setHitFilter] = useState<"ALL" | "GO" | "MARGINAL" | "NO-GO">("ALL");

  React.useEffect(() => {
    const onResize = () => setViewportWidth(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const fetchChart = useCallback(
    (signal: AbortSignal) => api.marketChart(symbol, interval, 120, signal),
    [interval, symbol]
  );

  const chartPoll = usePoll<MarketChartResponse>({
    fetcher: fetchChart,
    pollKey: "api:/market/chart:research",
    intervalMs: 30_000,
    staleAfterMs: 90_000,
  });

  const chart = chartPoll.data;

  const geometry = useMemo<ChartGeometry>(() => {
    const width = 1100;
    const height = 560;
    const left = 54;
    const right = 22;
    const top = 26;
    const priceHeight = 310;
    const gap = 42;
    const rsiHeight = 120;
    const bottom = 36;
    const rsiTop = top + priceHeight + gap;
    const emptyGeometry: ChartGeometry = {
      width,
      height,
      left,
      right,
      top,
      bottom,
      priceHeight,
      gap,
      rsiHeight,
      rsiTop,
      priceMin: 0,
      priceMax: 0,
      stepX: 1,
      candleWidth: 4,
      candles: chart?.candles ?? [],
      priceY: () => top,
      rsiY: () => rsiTop,
      overlayPaths: [],
      rsiPath: "",
    };
    const candles = chart?.candles ?? [];
    if (candles.length < 2) {
      return emptyGeometry;
    }

    const highs = candles.map((candle) => candle.high);
    const lows = candles.map((candle) => candle.low);
    const priceMin = Math.min(...lows);
    const priceMax = Math.max(...highs);
    const priceRange = Math.max(priceMax - priceMin, priceMin * 0.002, 1e-6);
    const stepX = (width - left - right) / candles.length;
    const candleWidth = Math.max(3, Math.min(10, stepX * 0.62));

    const priceY = (value: number) =>
      top + ((priceMax - value) / priceRange) * priceHeight;

    const rsiRange = 100;
    const rsiY = (value: number) => rsiTop + ((100 - value) / rsiRange) * rsiHeight;

    const overlayPaths = (chart?.overlays ?? []).map((overlay) => ({
      name: overlay.name,
      path: pathFromPoints(
        overlay.values.flatMap((value, index) =>
          typeof value === "number"
            ? [{ x: left + (index + 0.5) * stepX, y: priceY(value) }]
            : []
        )
      ),
    }));

    const rsiPath = pathFromPoints(
      (chart?.oscillator?.values ?? []).flatMap((value, index) =>
        typeof value === "number"
          ? [{ x: left + (index + 0.5) * stepX, y: rsiY(value) }]
          : []
      )
    );

    return {
      width,
      height,
      left,
      right,
      top,
      bottom,
      priceHeight,
      gap,
      rsiHeight,
      rsiTop,
      priceMin,
      priceMax,
      stepX,
      candleWidth,
      candles,
      priceY,
      rsiY,
      overlayPaths,
      rsiPath,
    };
  }, [chart]);

  const latest = chart?.candles?.[chart.candles.length - 1];
  const latestRsi = chart?.oscillator?.values?.[chart.oscillator.values.length - 1] ?? null;
  const pocketMarkers = chart?.pocket_markers ?? [];
  const visiblePocketMarkers = useMemo(() => {
    if (pocketMarkers.length === 0) {
      return [];
    }
    const prioritized = pocketMarkers.filter((marker) => marker.verdict !== "WAIT");
    const base = prioritized.length > 0 ? prioritized : pocketMarkers;
    const recent = base.slice(-24);
    const scored = [...recent].sort((left, right) => (right.score ?? 0) - (left.score ?? 0));
    return scored.slice(0, 12).sort((left, right) => left.time - right.time);
  }, [pocketMarkers]);
  const [selectedMarkerKey, setSelectedMarkerKey] = useState<string | null>(null);
  const selectedMarker = useMemo(() => {
    const selected =
      visiblePocketMarkers.find(
        (marker) => `${marker.time}-${marker.bucket_time}` === selectedMarkerKey,
      ) ??
      pocketMarkers.find(
        (marker) => `${marker.time}-${marker.bucket_time}` === selectedMarkerKey,
      );
    return selected ?? pocketMarkers[pocketMarkers.length - 1] ?? null;
  }, [pocketMarkers, selectedMarkerKey, visiblePocketMarkers]);
  React.useEffect(() => {
    const preferred =
      [...pocketMarkers].reverse().find((marker) => marker.verdict === "GO") ??
      pocketMarkers[pocketMarkers.length - 1] ??
      null;
    setSelectedMarkerKey(preferred ? `${preferred.time}-${preferred.bucket_time}` : null);
  }, [interval, symbol, pocketMarkers]);
  const selectedNeighborhood = useMemo(() => {
    if (!selectedMarker || pocketMarkers.length === 0) {
      return [];
    }
    const selectedIndex = pocketMarkers.findIndex(
      (marker) => `${marker.time}-${marker.bucket_time}` === `${selectedMarker.time}-${selectedMarker.bucket_time}`,
    );
    if (selectedIndex < 0) {
      return pocketMarkers.slice(-5);
    }
    const start = Math.max(0, selectedIndex - 2);
    const end = Math.min(pocketMarkers.length, selectedIndex + 3);
    return pocketMarkers.slice(start, end);
  }, [pocketMarkers, selectedMarker]);
  const overlayLatest = (chart?.overlays ?? []).map((overlay) => ({
    name: overlay.name,
    value: overlay.values[overlay.values.length - 1] ?? null,
  }));
  const laneContext = useMemo(() => {
    const events = asRecord(researchEvents);
    const watchboard = asRecord(events.watchboard);
    const watchboardLanes = asArray<Record<string, unknown>>(watchboard.lanes);
    const states = asRecord(events.states);
    const watchlists = asRecord(events.watchlists);

    const entries = [
      "liquidation",
      "spread_stress",
      "return_shock",
      "volume_vacuum",
      "volatility_burst",
      "book_proxy_pressure",
      "fill_toxicity",
      "latency_stress",
    ]
      .map((lane) => {
        const laneState = asRecord(states[lane]);
        const state = asRecord(laneState.state);
        const freshness = asRecord(state.freshness);
        const watchlist = asRecord(watchlists[lane]);
        const topSummary = asRecord(watchlist.top_summary);
        const rows = asArray<Record<string, unknown>>(watchlist.rows);
        const matchingRow = rows.find((row) => String(row.symbol || "").toUpperCase() === symbol);
        const topSymbol = String(topSummary.symbol || topSummary.top_symbol || "").toUpperCase();
        const symbolMatch = topSymbol === symbol || Boolean(matchingRow);
        const boardLane =
          watchboardLanes.find((row) => String(row.lane || row.name || "").toLowerCase() === lane) ?? {};
        const boardRecord = asRecord(boardLane);
        const level = String(
          state.level ||
          boardRecord.level ||
          topSummary.state_level ||
          matchingRow?.state_level ||
          "quiet"
        );
        const freshnessStatus = String(
          freshness.status ||
          boardRecord.freshness_status ||
          topSummary.freshness_status ||
          matchingRow?.freshness_status ||
          "unknown"
        );
        const action = String(
          laneState.recommended_action ||
          boardRecord.recommended_action ||
          topSummary.recommended_action ||
          "monitor_only"
        );
        const banner = asRecord(watchlist.banner);
        const summary = String(
          laneState.dashboard_summary ||
          boardRecord.detail ||
          boardRecord.headline ||
          banner.detail ||
          banner.headline ||
          ""
        );
        return {
          lane,
          title: titleizeLane(lane),
          level,
          freshnessStatus,
          action,
          summary: summary || "No lane summary available.",
          symbolMatch,
          stale: freshnessStatus.toLowerCase() === "stale",
        };
      })
      .filter((entry) => entry.symbolMatch || entry.lane === "fill_toxicity" || entry.lane === "latency_stress")
      .sort((left, right) => {
        if (left.symbolMatch !== right.symbolMatch) return left.symbolMatch ? -1 : 1;
        if (left.stale !== right.stale) return left.stale ? 1 : -1;
        return levelRank(left.level) - levelRank(right.level);
      });

    const prioritized = entries.slice(0, 4);
    const top = prioritized[0] ?? null;
    const operatorHint = top
      ? top.action === "escalate_monitoring"
        ? `Escalate monitoring on ${symbol}; ${top.title} is the strongest current lane.`
        : top.action === "show_caution"
          ? `Keep caution on ${symbol}; ${top.title} is active around this setup.`
          : `Monitor ${symbol} before acting; ${top.title} is the current top lane context.`
      : `No active lane context available for ${symbol}.`;

    return { items: prioritized, operatorHint };
  }, [researchEvents, symbol]);
  const thresholdStatus = useMemo(() => {
    if (!selectedMarker) {
      return [];
    }
    const absImbalance = Math.abs(Number(selectedMarker.imbalance ?? 0));
    const tradeIntensity = Number(selectedMarker.trade_intensity ?? 0);
    const spread = Number(selectedMarker.spread ?? 0);
    return [
      {
        label: "|imbalance|",
        targetLabel: ">= 0.500",
        actualLabel: formatCompact(absImbalance, 3),
        pass: absImbalance >= 0.5,
        ratio: clamp(absImbalance / 0.5, 0, 1.4),
      },
      {
        label: "trade_intensity",
        targetLabel: ">= 3500",
        actualLabel: formatCompact(tradeIntensity, 0),
        pass: tradeIntensity >= 3500,
        ratio: clamp(tradeIntensity / 3500, 0, 1.4),
      },
      {
        label: "spread",
        targetLabel: "<= 0.00030",
        actualLabel: formatCompact(spread, 5),
        pass: spread <= 0.0003,
        ratio: clamp((0.0003 - spread) / 0.0003, 0, 1),
      },
    ];
  }, [selectedMarker]);
  const combinedAction = useMemo(() => {
    if (!selectedMarker) {
      return {
        title: `No selected pocket for ${symbol}`,
        detail: "Select a recent pocket hit to evaluate research and lane context together.",
        tone: "gray",
      };
    }
    const topLane = laneContext.items[0] ?? null;
    if (selectedMarker.verdict === "GO" && topLane && topLane.action === "monitor_only") {
      return {
        title: `${symbol}: research pocket is GO, but lane context says monitor`,
        detail: `${topLane.title} is the leading lane. Treat this as a research-quality setup, not an automatic action.`,
        tone: "yellow",
      };
    }
    if (selectedMarker.verdict === "GO" && topLane && topLane.action === "show_caution") {
      return {
        title: `${symbol}: GO pocket with caution overlay`,
        detail: `${topLane.title} is active. Favor caution and tighter operator review before trusting the setup.`,
        tone: "yellow",
      };
    }
    if (selectedMarker.verdict === "GO") {
      return {
        title: `${symbol}: GO pocket with supportive lane context`,
        detail: laneContext.operatorHint,
        tone: "green",
      };
    }
    if (selectedMarker.verdict === "MARGINAL") {
      return {
        title: `${symbol}: marginal pocket, keep as monitor-only`,
        detail: `Regime and lane context do not support aggressive interpretation of this hit.`,
        tone: "yellow",
      };
    }
    if (selectedMarker.verdict === "NO-GO") {
      return {
        title: `${symbol}: no-go pocket, do not treat as promotable`,
        detail: `The research verdict blocks this hit. Lane context can still be useful for monitoring, but not for promotion logic.`,
        tone: "red",
      };
    }
    return {
      title: `${symbol}: waiting on stronger regime confirmation`,
      detail: laneContext.operatorHint,
      tone: "gray",
    };
  }, [laneContext, selectedMarker, symbol]);
  const combinedActionClass =
    combinedAction.tone === "green"
      ? "badge-green"
      : combinedAction.tone === "yellow"
        ? "badge-yellow"
        : combinedAction.tone === "red"
          ? "badge-red"
          : "badge-gray";
  const modeLabel = symbol === "ETHUSDT" ? "ETH research mode" : "BTC lite mode";
  const modeBadgeClass = symbol === "ETHUSDT" ? "badge-blue" : "badge-gray";
  const stackedLayout = viewportWidth < 1180;
  const compactOverlay = viewportWidth < 840;
  const compactStats = viewportWidth < 640;
  const verdictCounts = useMemo(
    () => ({
      ALL: pocketMarkers.length,
      GO: pocketMarkers.filter((marker) => marker.verdict === "GO").length,
      MARGINAL: pocketMarkers.filter((marker) => marker.verdict === "MARGINAL").length,
      "NO-GO": pocketMarkers.filter((marker) => marker.verdict === "NO-GO").length,
    }),
    [pocketMarkers],
  );
  const filteredPocketHits = useMemo(() => {
    const base = pocketMarkers.slice().reverse();
    if (hitFilter === "ALL") {
      return base;
    }
    return base.filter((marker) => marker.verdict === hitFilter);
  }, [hitFilter, pocketMarkers]);
  const topScoreHit = useMemo(() => {
    if (pocketMarkers.length === 0) return null;
    return [...pocketMarkers].sort((left, right) => (right.score ?? 0) - (left.score ?? 0))[0] ?? null;
  }, [pocketMarkers]);
  const latestGoHit = useMemo(
    () => [...pocketMarkers].reverse().find((marker) => marker.verdict === "GO") ?? null,
    [pocketMarkers],
  );

  return (
    <div className="card" style={{ borderLeft: "3px solid var(--blue)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
        <div>
          <div className="card-title" style={{ marginBottom: 6 }}>Market Structure Chart</div>
          <div style={{ color: "var(--muted)" }}>
            External candles from Binance Spot API, overlaid with EMA 20, EMA 50, and RSI 14.
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          {SYMBOLS.map((nextSymbol) => (
            <button
              key={nextSymbol}
              onClick={() => onSymbolChange(nextSymbol)}
              style={{
                padding: "4px 10px",
                borderRadius: 999,
                border: "1px solid var(--border)",
                background: symbol === nextSymbol ? "rgba(56, 139, 253, 0.18)" : "transparent",
                color: symbol === nextSymbol ? "var(--text)" : "var(--muted)",
                cursor: "pointer",
                fontSize: 12,
              }}
            >
              {nextSymbol}
            </button>
          ))}
          {INTERVALS.map((nextInterval) => (
            <button
              key={nextInterval}
              onClick={() => onIntervalChange(nextInterval)}
              style={{
                padding: "4px 8px",
                borderRadius: 4,
                border: "1px solid var(--border)",
                background: interval === nextInterval ? "var(--surface-2)" : "transparent",
                color: interval === nextInterval ? "var(--text)" : "var(--muted)",
                cursor: "pointer",
                fontSize: 12,
              }}
            >
              {nextInterval}
            </button>
          ))}
          <span className={`badge ${modeBadgeClass}`}>{modeLabel}</span>
          <span className="badge badge-blue">{chart?.source ?? "binance_spot"}</span>
        </div>
      </div>

      {chartPoll.error ? (
        <div style={{ marginTop: 14, color: "var(--red)" }}>{chartPoll.error.message}</div>
      ) : null}

      {!chart || !chart.candles || chart.candles.length < 2 ? (
        <div style={{ marginTop: 14, color: "var(--muted)" }}>
          {chartPoll.isLoading ? "Loading market chart..." : "No chart payload available."}
        </div>
      ) : (
        <>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
            <span className="badge badge-green">close={formatCompact(latest?.close, 2)}</span>
            {overlayLatest.map((item) => (
              <span key={item.name} className="badge badge-gray">
                {item.name}={formatCompact(item.value, 2)}
              </span>
            ))}
            <span className="badge badge-yellow">RSI 14={formatCompact(latestRsi, 2)}</span>
            <span className="badge badge-blue">pocket_hits={pocketMarkers.length}</span>
            <span className="badge badge-gray">visible={visiblePocketMarkers.length}</span>
            <span className="badge badge-gray">generated={chart.generated_ts}</span>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10, alignItems: "center" }}>
            <span style={{ color: "var(--muted)", fontSize: 12 }}>Pocket Marker Legend</span>
            <span className="badge badge-green">GO</span>
            <span className="badge badge-yellow">MARGINAL</span>
            <span className="badge badge-red">NO-GO</span>
            <span className="badge badge-gray">WAIT</span>
            <span className="badge badge-gray">hover marker for details</span>
          </div>

          {symbol !== "ETHUSDT" ? (
            <div style={{ marginTop: 10, padding: "10px 12px", borderRadius: 8, border: "1px solid var(--border)", background: "rgba(255,255,255,0.03)", color: "var(--muted)", fontSize: 12 }}>
              BTC is currently rendered in lite mode. Chart indicators stay available, but promoted pocket logic and richer research interpretations remain ETH-first.
            </div>
          ) : null}

          <div
            style={{
              display: "grid",
              gridTemplateColumns: stackedLayout ? "minmax(0, 1fr)" : "minmax(0, 1fr) 280px",
              gap: 12,
              marginTop: 14,
              alignItems: "start",
            }}
          >
            <div style={{ border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden", background: "linear-gradient(180deg, rgba(56, 139, 253, 0.08), rgba(0, 0, 0, 0))" }}>
            {selectedMarker ? (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: compactOverlay ? "1fr" : "repeat(3, minmax(0, 1fr))",
                  gap: 8,
                  padding: "10px 12px",
                  borderBottom: "1px solid var(--border)",
                  background: "rgba(11, 16, 32, 0.55)",
                }}
              >
                {thresholdStatus.map((item) => (
                  <div key={`overlay-${item.label}`} style={{ minWidth: 0 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}>
                      <div style={{ color: "var(--muted)", fontSize: 11, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {item.label}
                      </div>
                      <span className={`badge ${item.pass ? "badge-green" : "badge-red"}`}>
                        {item.actualLabel}
                      </span>
                    </div>
                    <div style={{ height: 7, borderRadius: 999, background: "rgba(255,255,255,0.08)", marginTop: 6, overflow: "hidden" }}>
                      <div
                        style={{
                          width: `${clamp(item.ratio, 0, 1) * 100}%`,
                          height: "100%",
                          background: item.pass ? "linear-gradient(90deg, #22c55e, #86efac)" : "linear-gradient(90deg, #ef4444, #fca5a5)",
                        }}
                      />
                    </div>
                    <div style={{ color: "var(--muted)", fontSize: 10, marginTop: 4 }}>
                      target {item.targetLabel}
                    </div>
                  </div>
                ))}
              </div>
            ) : null}
            {visiblePocketMarkers.length > 0 ? (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: compactOverlay
                    ? "repeat(6, minmax(0, 1fr))"
                    : `repeat(${Math.min(visiblePocketMarkers.length, 12)}, minmax(0, 1fr))`,
                  gap: 4,
                  padding: "8px 12px",
                  borderBottom: "1px solid var(--border)",
                  background: "rgba(255,255,255,0.02)",
                }}
              >
                {visiblePocketMarkers.map((marker) => {
                  const isSelected =
                    selectedMarker &&
                    `${selectedMarker.time}-${selectedMarker.bucket_time}` === `${marker.time}-${marker.bucket_time}`;
                  const bg =
                    marker.verdict === "GO"
                      ? "linear-gradient(180deg, rgba(34,197,94,0.85), rgba(34,197,94,0.25))"
                      : marker.verdict === "MARGINAL"
                        ? "linear-gradient(180deg, rgba(245,158,11,0.85), rgba(245,158,11,0.25))"
                        : marker.verdict === "NO-GO"
                          ? "linear-gradient(180deg, rgba(239,68,68,0.85), rgba(239,68,68,0.25))"
                          : "linear-gradient(180deg, rgba(148,163,184,0.85), rgba(148,163,184,0.25))";
                  return (
                    <button
                      key={`ribbon-${marker.time}-${marker.bucket_time}`}
                      type="button"
                      onClick={() => setSelectedMarkerKey(`${marker.time}-${marker.bucket_time}`)}
                      title={`${new Date(marker.bucket_time * 1000).toLocaleTimeString()} | ${marker.side} ${marker.verdict} | score=${formatCompact(marker.score, 3)}`}
                      style={{
                        border: isSelected ? "1px solid var(--text)" : "1px solid rgba(255,255,255,0.08)",
                        borderRadius: 6,
                        background: bg,
                        minHeight: 28,
                        cursor: "pointer",
                        color: "white",
                        fontSize: 10,
                        fontWeight: 700,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        padding: 0,
                      }}
                    >
                      {marker.side}
                    </button>
                  );
                })}
              </div>
            ) : null}
            {laneContext.items.length > 0 ? (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: compactOverlay ? "1fr" : `repeat(${Math.min(laneContext.items.length, 4)}, minmax(0, 1fr))`,
                  gap: 8,
                  padding: "8px 12px",
                  borderBottom: "1px solid var(--border)",
                  background: "rgba(255,255,255,0.02)",
                }}
              >
                {laneContext.items.map((item) => {
                  const strength = laneStrength(item.level, item.stale);
                  return (
                    <div key={`lane-strip-${item.lane}`} style={{ minWidth: 0 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}>
                        <div
                          style={{
                            fontSize: 11,
                            fontWeight: 700,
                            whiteSpace: "nowrap",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                          }}
                          title={item.title}
                        >
                          {item.title}
                        </div>
                        <span
                          className={`badge ${
                            item.level.toLowerCase() === "severe" || item.level.toLowerCase() === "high"
                              ? "badge-red"
                              : item.level.toLowerCase() === "elevated" || item.level.toLowerCase() === "medium"
                                ? "badge-yellow"
                                : "badge-green"
                          }`}
                        >
                          {item.level}
                        </span>
                      </div>
                      <div style={{ height: 7, borderRadius: 999, background: "rgba(255,255,255,0.08)", marginTop: 6, overflow: "hidden" }}>
                        <div
                          style={{
                            width: `${strength * 100}%`,
                            height: "100%",
                            background:
                              item.level.toLowerCase() === "severe" || item.level.toLowerCase() === "high"
                                ? "linear-gradient(90deg, #ef4444, #fca5a5)"
                                : item.level.toLowerCase() === "elevated" || item.level.toLowerCase() === "medium"
                                  ? "linear-gradient(90deg, #f59e0b, #fcd34d)"
                                  : "linear-gradient(90deg, #22c55e, #86efac)",
                            opacity: item.stale ? 0.65 : 1,
                          }}
                        />
                      </div>
                      <div style={{ color: "var(--muted)", fontSize: 10, marginTop: 4 }}>
                        current lane context {item.stale ? "stale" : item.action}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : null}
            <svg viewBox={`0 0 ${geometry.width} ${geometry.height}`} width="100%" role="img" aria-label={`${symbol} price structure chart`}>
              <rect x="0" y="0" width={geometry.width} height={geometry.height} fill="transparent" />

              {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
                const y = geometry.top + geometry.priceHeight * ratio;
                const value = geometry.priceMax - (geometry.priceMax - geometry.priceMin) * ratio;
                return (
                  <g key={`price-grid-${ratio}`}>
                    <line x1={geometry.left} y1={y} x2={geometry.width - geometry.right} y2={y} stroke="rgba(255,255,255,0.08)" strokeDasharray="4 6" />
                    <text x={12} y={y + 4} fill="var(--muted)" fontSize="11">{formatCompact(value, 2)}</text>
                  </g>
                );
              })}

              {[30, 50, 70].map((level) => {
                const y = geometry.rsiY(level);
                return (
                  <g key={`rsi-grid-${level}`}>
                    <line x1={geometry.left} y1={y} x2={geometry.width - geometry.right} y2={y} stroke={level === 50 ? "rgba(255,255,255,0.12)" : "rgba(255,196,0,0.18)"} strokeDasharray="4 6" />
                    <text x={12} y={y + 4} fill="var(--muted)" fontSize="11">{level}</text>
                  </g>
                );
              })}

              <text x={geometry.left} y={18} fill="var(--muted)" fontSize="12">Price</text>
              <text x={geometry.left} y={geometry.rsiTop - 12} fill="var(--muted)" fontSize="12">RSI 14</text>

              {geometry.candles.map((candle, index) => {
                const x = geometry.left + (index + 0.5) * geometry.stepX;
                const openY = geometry.priceY(candle.open);
                const closeY = geometry.priceY(candle.close);
                const highY = geometry.priceY(candle.high);
                const lowY = geometry.priceY(candle.low);
                const bullish = candle.close >= candle.open;
                const color = bullish ? "#2dd4bf" : "#f87171";
                const bodyY = Math.min(openY, closeY);
                const bodyHeight = Math.max(1.5, Math.abs(closeY - openY));
                return (
                  <g key={`candle-${candle.time}`}>
                    <line x1={x} y1={highY} x2={x} y2={lowY} stroke={color} strokeWidth="1.2" />
                    <rect
                      x={x - geometry.candleWidth / 2}
                      y={bodyY}
                      width={geometry.candleWidth}
                      height={bodyHeight}
                      fill={bullish ? "rgba(45, 212, 191, 0.45)" : "rgba(248, 113, 113, 0.42)"}
                      stroke={color}
                      strokeWidth="1"
                      rx="1.5"
                    />
                  </g>
                );
              })}

              {geometry.overlayPaths.map((overlay, index) => (
                <path
                  key={overlay.name}
                  d={overlay.path}
                  fill="none"
                  stroke={index === 0 ? "#60a5fa" : "#fbbf24"}
                  strokeWidth="2"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
              ))}

              <path d={geometry.rsiPath} fill="none" stroke="#c084fc" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />

              {visiblePocketMarkers.map((marker) => {
                const index = geometry.candles.findIndex((candle) => candle.time === marker.time);
                if (index < 0) return null;
                const x = geometry.left + (index + 0.5) * geometry.stepX;
                const candle = geometry.candles[index];
                const yBase = geometry.priceY(candle.high) - 10;
                const fill =
                  marker.verdict === "GO" ? "#22c55e" :
                  marker.verdict === "MARGINAL" ? "#f59e0b" :
                  marker.verdict === "NO-GO" ? "#ef4444" :
                  "#94a3b8";
                return (
                  <g key={`marker-${marker.time}-${marker.bucket_time}`}>
                    <title>
                      {[
                        `${marker.side} ${marker.verdict}`,
                        `regime=${marker.regime}`,
                        `bucket=${new Date(marker.bucket_time * 1000).toLocaleString()}`,
                        `imbalance=${formatCompact(marker.imbalance, 3)}`,
                        `trade_intensity=${formatCompact(marker.trade_intensity, 0)}`,
                        `spread=${formatCompact(marker.spread, 5)}`,
                        `score=${formatCompact(marker.score, 3)}`,
                      ].join(" | ")}
                    </title>
                    <circle
                      cx={x}
                      cy={yBase}
                      r={selectedMarker && `${selectedMarker.time}-${selectedMarker.bucket_time}` === `${marker.time}-${marker.bucket_time}` ? 7 : 5.5}
                      fill={fill}
                      stroke="#0b1020"
                      strokeWidth="1.5"
                      style={{ cursor: "pointer" }}
                      onClick={() => setSelectedMarkerKey(`${marker.time}-${marker.bucket_time}`)}
                    />
                    <text x={x} y={yBase - 10} textAnchor="middle" fill={fill} fontSize="10" fontWeight="700">
                      {marker.side}
                    </text>
                  </g>
                );
              })}

              {geometry.candles.flatMap((candle, index) => {
                const sampleEvery = Math.max(1, Math.floor(geometry.candles.length / 6));
                if (index % sampleEvery !== 0) return [];
                const x = geometry.left + (index + 0.5) * geometry.stepX;
                const label = new Date(candle.time * 1000).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                });
                return [
                  <text key={`time-${candle.time}`} x={x} y={geometry.height - 10} textAnchor="middle" fill="var(--muted)" fontSize="11">
                    {label}
                  </text>
                ];
              })}
            </svg>
            </div>

            <div style={{ border: "1px solid var(--border)", borderRadius: 10, padding: 12, background: "var(--surface-2)", minWidth: 0 }}>
              <div style={{ fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>
                Selected Pocket
              </div>
              {selectedMarker ? (
                <>
                  <div style={{ padding: "10px 12px", borderRadius: 8, border: "1px solid var(--border)", background: "rgba(56, 139, 253, 0.08)", marginBottom: 10 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <div style={{ fontWeight: 700 }}>{combinedAction.title}</div>
                      <span className={`badge ${combinedActionClass}`}>{selectedMarker.verdict}</span>
                    </div>
                    <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 6 }}>
                      {combinedAction.detail}
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
                    <span className={`badge ${selectedMarker.verdict === "GO" ? "badge-green" : selectedMarker.verdict === "MARGINAL" ? "badge-yellow" : selectedMarker.verdict === "NO-GO" ? "badge-red" : "badge-gray"}`}>
                      {selectedMarker.verdict}
                    </span>
                    <span className="badge badge-gray">{selectedMarker.side}</span>
                    <span className="badge badge-gray">regime={selectedMarker.regime}</span>
                  </div>
                  <div style={{ fontWeight: 700, marginBottom: 6 }}>
                    {new Date(selectedMarker.bucket_time * 1000).toLocaleString()}
                  </div>
                  <div style={{ color: "var(--muted)", fontSize: 12, marginBottom: 10 }}>
                    Strongest retained microstructure pocket marker in the visible chart window.
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: compactStats ? "1fr" : "1fr 1fr", gap: 8 }}>
                    <div style={{ padding: "8px 10px", borderRadius: 8, border: "1px solid var(--border)" }}>
                      <div style={{ color: "var(--muted)", fontSize: 11 }}>Imbalance</div>
                      <div style={{ fontWeight: 700 }}>{formatCompact(selectedMarker.imbalance, 3)}</div>
                    </div>
                    <div style={{ padding: "8px 10px", borderRadius: 8, border: "1px solid var(--border)" }}>
                      <div style={{ color: "var(--muted)", fontSize: 11 }}>Trade Intensity</div>
                      <div style={{ fontWeight: 700 }}>{formatCompact(selectedMarker.trade_intensity, 0)}</div>
                    </div>
                    <div style={{ padding: "8px 10px", borderRadius: 8, border: "1px solid var(--border)" }}>
                      <div style={{ color: "var(--muted)", fontSize: 11 }}>Spread</div>
                      <div style={{ fontWeight: 700 }}>{formatCompact(selectedMarker.spread, 5)}</div>
                    </div>
                    <div style={{ padding: "8px 10px", borderRadius: 8, border: "1px solid var(--border)" }}>
                      <div style={{ color: "var(--muted)", fontSize: 11 }}>Score</div>
                      <div style={{ fontWeight: 700 }}>{formatCompact(selectedMarker.score, 3)}</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", marginTop: 12, marginBottom: 8 }}>
                    Pocket Thresholds
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {thresholdStatus.map((item) => (
                      <div key={item.label} style={{ padding: "8px 10px", borderRadius: 8, border: "1px solid var(--border)" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                          <div style={{ fontWeight: 700 }}>{item.label}</div>
                          <span className={`badge ${item.pass ? "badge-green" : "badge-red"}`}>
                            {item.actualLabel}
                          </span>
                        </div>
                        <div style={{ color: "var(--muted)", fontSize: 11, marginTop: 4 }}>
                          target {item.targetLabel}
                        </div>
                        <div style={{ height: 8, borderRadius: 999, background: "rgba(255,255,255,0.08)", marginTop: 8, overflow: "hidden" }}>
                          <div
                            style={{
                              width: `${clamp(item.ratio, 0, 1) * 100}%`,
                              height: "100%",
                              background: item.pass ? "linear-gradient(90deg, #22c55e, #86efac)" : "linear-gradient(90deg, #ef4444, #fca5a5)",
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <div style={{ fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", marginTop: 12, marginBottom: 8 }}>
                    Current Lane Context
                  </div>
                  <div style={{ color: "var(--muted)", fontSize: 12, marginBottom: 8 }}>
                    {laneContext.operatorHint}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {laneContext.items.map((item) => (
                      <div key={item.lane} style={{ padding: "8px 10px", borderRadius: 8, border: "1px solid var(--border)" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                          <div style={{ fontWeight: 700 }}>{item.title}</div>
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                            <span className={`badge ${item.level.toLowerCase() === "severe" || item.level.toLowerCase() === "high" ? "badge-red" : item.level.toLowerCase() === "elevated" || item.level.toLowerCase() === "medium" ? "badge-yellow" : "badge-green"}`}>
                              {item.level}
                            </span>
                            <span className={`badge ${item.stale ? "badge-gray" : "badge-blue"}`}>
                              {item.freshnessStatus}
                            </span>
                          </div>
                        </div>
                        <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 6 }}>
                          {item.summary}
                        </div>
                        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6 }}>
                          <span className="badge badge-gray">action={item.action}</span>
                          {item.symbolMatch ? <span className="badge badge-blue">{symbol}</span> : <span className="badge badge-gray">system</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div style={{ fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", marginTop: 12, marginBottom: 8 }}>
                    Pocket Neighborhood
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {selectedNeighborhood.map((marker) => {
                      const isSelected =
                        selectedMarker &&
                        `${selectedMarker.time}-${selectedMarker.bucket_time}` === `${marker.time}-${marker.bucket_time}`;
                      return (
                        <button
                          key={`neighbor-${marker.time}-${marker.bucket_time}`}
                          type="button"
                          onClick={() => setSelectedMarkerKey(`${marker.time}-${marker.bucket_time}`)}
                          style={{
                            padding: "8px 10px",
                            borderRadius: 8,
                            border: isSelected ? "1px solid var(--blue)" : "1px solid var(--border)",
                            background: "transparent",
                            textAlign: "left",
                            cursor: "pointer",
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                            <div style={{ fontWeight: 700 }}>
                              {new Date(marker.bucket_time * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                            </div>
                            <span className={`badge ${marker.verdict === "GO" ? "badge-green" : marker.verdict === "MARGINAL" ? "badge-yellow" : marker.verdict === "NO-GO" ? "badge-red" : "badge-gray"}`}>
                              {marker.verdict}
                            </span>
                          </div>
                          <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 4 }}>
                            side={marker.side} regime={marker.regime} score={formatCompact(marker.score, 3)}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </>
              ) : (
                <div style={{ color: "var(--muted)" }}>No pocket marker selected.</div>
              )}
            </div>
          </div>

          {pocketMarkers.length > 0 ? (
            <div style={{ marginTop: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 8 }}>
                <div style={{ fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Pocket Hits
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                  <button
                    type="button"
                    onClick={() => latestGoHit && setSelectedMarkerKey(`${latestGoHit.time}-${latestGoHit.bucket_time}`)}
                    disabled={!latestGoHit}
                    style={{
                      padding: "4px 8px",
                      borderRadius: 999,
                      border: "1px solid var(--border)",
                      background: "transparent",
                      color: latestGoHit ? "var(--text)" : "var(--muted)",
                      cursor: latestGoHit ? "pointer" : "default",
                      fontSize: 11,
                      opacity: latestGoHit ? 1 : 0.55,
                    }}
                  >
                    Latest GO
                  </button>
                  <button
                    type="button"
                    onClick={() => topScoreHit && setSelectedMarkerKey(`${topScoreHit.time}-${topScoreHit.bucket_time}`)}
                    disabled={!topScoreHit}
                    style={{
                      padding: "4px 8px",
                      borderRadius: 999,
                      border: "1px solid var(--border)",
                      background: "transparent",
                      color: topScoreHit ? "var(--text)" : "var(--muted)",
                      cursor: topScoreHit ? "pointer" : "default",
                      fontSize: 11,
                      opacity: topScoreHit ? 1 : 0.55,
                    }}
                  >
                    Top Score
                  </button>
                  {(["ALL", "GO", "MARGINAL", "NO-GO"] as const).map((value) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => setHitFilter(value)}
                      style={{
                        padding: "4px 8px",
                        borderRadius: 999,
                        border: "1px solid var(--border)",
                        background: hitFilter === value ? "var(--surface-2)" : "transparent",
                        color: hitFilter === value ? "var(--text)" : "var(--muted)",
                        cursor: "pointer",
                        fontSize: 11,
                      }}
                    >
                      {value} {verdictCounts[value]}
                    </button>
                  ))}
                </div>
              </div>
              <div style={{ color: "var(--muted)", fontSize: 12, marginBottom: 8 }}>
                Showing the strongest recent pocket hits on-chart to keep the structure readable. List filter affects only the hit cards below.
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
                {filteredPocketHits.slice(0, 6).map((marker) => (
                  <button
                    key={`hit-${marker.time}-${marker.bucket_time}`}
                    type="button"
                    onClick={() => setSelectedMarkerKey(`${marker.time}-${marker.bucket_time}`)}
                    style={{
                      padding: "10px 12px",
                      borderRadius: 8,
                      border: `${selectedMarker && `${selectedMarker.time}-${selectedMarker.bucket_time}` === `${marker.time}-${marker.bucket_time}` ? "1px solid var(--blue)" : "1px solid var(--border)"}`,
                      background: "var(--surface-2)",
                      textAlign: "left",
                      cursor: "pointer",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                      <div style={{ fontWeight: 700 }}>{new Date(marker.bucket_time * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</div>
                      <span className={`badge ${marker.verdict === "GO" ? "badge-green" : marker.verdict === "MARGINAL" ? "badge-yellow" : marker.verdict === "NO-GO" ? "badge-red" : "badge-gray"}`}>
                        {marker.verdict}
                      </span>
                    </div>
                    <div style={{ color: "var(--muted)", marginTop: 6, fontSize: 12 }}>
                      side={marker.side} regime={marker.regime}
                    </div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 8 }}>
                      <span className="badge badge-gray">imb={formatCompact(marker.imbalance, 3)}</span>
                      <span className="badge badge-gray">ti={formatCompact(marker.trade_intensity, 0)}</span>
                      <span className="badge badge-gray">spr={formatCompact(marker.spread, 5)}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div style={{ marginTop: 12, color: "var(--muted)" }}>
              No recent pocket hits detected from live microstructure buckets for this symbol/interval window.
            </div>
          )}
        </>
      )}
    </div>
  );
}
