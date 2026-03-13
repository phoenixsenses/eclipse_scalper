import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { LiveMetricsResponse, LiveMonitorTestsStatusResponse, LogFile, LogTailResponse, OpsHealthResponse, RuntimeStatus, Scoreboard, LiqAlertState, SpreadStressState, FillToxicityState, LatencyStressState, WatchboardState, BookProxyPressureState, ReturnShockState, VolatilityBurstState, VolumeVacuumState, PaperRunStatusResponse } from "../api/types";
import AsyncState from "../components/AsyncState";
import DegradedBanner, { type DegradedMode } from "../components/DegradedBanner";
import LiqAlertCard from "../components/LiqAlertCard";
import SpreadStressCard from "../components/SpreadStressCard";
import FillToxicityCard from "../components/FillToxicityCard";
import LatencyStressCard from "../components/LatencyStressCard";
import WatchboardCard from "../components/WatchboardCard";
import BookProxyPressureCard from "../components/BookProxyPressureCard";
import ReturnShockCard from "../components/ReturnShockCard";
import VolatilityBurstCard from "../components/VolatilityBurstCard";
import VolumeVacuumCard from "../components/VolumeVacuumCard";
import PageGuide from "../components/PageGuide";
import { usePoll } from "../hooks/usePoll";

type FillRow = {
  ts: string;
  symbol: string;
  side: string;
  price: string;
  qty: string;
  pnl: string;
  tsMs?: number | null;
  pnlNum?: number | null;
  delayMs?: number | null;
  adverseBps?: number | null;
};

const DEFAULT_TRADE_AGE_ALERT_SEC = 10;
const DEFAULT_FILL_FLATLINE_ALERT_MIN = 15;
const LIVE_CFG_KEY = "eclipse.live.monitor.cfg.v1";

function num(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return "-";
  return v.toFixed(digits);
}

function dbSizeMb(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "-";
  return `${(v / (1024 * 1024)).toFixed(1)} MB`;
}

function asObj(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

function parseMaybeJson(line: string): Record<string, unknown> | null {
  const s = line.trim();
  if (!s.startsWith("{") || !s.endsWith("}")) return null;
  try {
    return asObj(JSON.parse(s));
  } catch {
    return null;
  }
}

function fmtUnknown(v: unknown): string {
  if (v == null) return "-";
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : "-";
  if (typeof v === "string" && v.trim()) return v;
  return "-";
}

function toMillis(ts: string | null | undefined): number | null {
  if (!ts) return null;
  const ms = Date.parse(ts);
  return Number.isFinite(ms) ? ms : null;
}

function fmtDuration(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) return "-";
  const total = Math.max(0, Math.floor(sec));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

function parseTsAny(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) {
    return v > 10_000_000_000 ? v : v * 1000;
  }
  if (typeof v === "string" && v.trim()) {
    const asNum = Number(v);
    if (Number.isFinite(asNum) && asNum > 0) return asNum > 10_000_000_000 ? asNum : asNum * 1000;
    const p = Date.parse(v);
    return Number.isFinite(p) ? p : null;
  }
  return null;
}

function parseNumAny(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim()) {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function Spark({ values, color = "var(--accent)" }: { values: number[]; color?: string }) {
  const w = 180;
  const h = 42;
  if (!values.length) {
    return <div style={{ height: h, color: "var(--muted)", fontSize: 11 }}>no samples</div>;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1e-6, max - min);
  const pts = values.map((v, i) => {
    const x = (i / Math.max(1, values.length - 1)) * (w - 4) + 2;
    const y = h - 2 - ((v - min) / span) * (h - 8);
    return `${x},${y}`;
  });
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label="sparkline">
      <polyline fill="none" stroke={color} strokeWidth="2" points={pts.join(" ")} />
    </svg>
  );
}

function rowFromJson(obj: Record<string, unknown>): FillRow | null {
  const type = String(obj.type ?? obj.event ?? obj.action ?? obj.status ?? "").toLowerCase();
  const isFill =
    type.includes("fill") ||
    type.includes("filled") ||
    String(obj.fill_price ?? "").length > 0 ||
    String(obj.avg_fill_price ?? "").length > 0;
  if (!isFill) return null;
  const ts = fmtUnknown(obj.ts ?? obj.ts_utc ?? obj.time ?? obj.timestamp);
  const symbol = fmtUnknown(obj.symbol ?? obj.sym);
  const side = fmtUnknown(obj.side ?? obj.direction);
  const price = fmtUnknown(obj.fill_price ?? obj.avg_fill_price ?? obj.price ?? obj.entry_price ?? obj.exit_price);
  const qty = fmtUnknown(obj.qty ?? obj.size ?? obj.filled_qty ?? obj.amount);
  const pnl = fmtUnknown(obj.pnl ?? obj.pnl_bps ?? obj.net_pnl_bps);
  const tsMs = parseTsAny(obj.ts_ms ?? obj.ts ?? obj.ts_utc ?? obj.time ?? obj.timestamp);
  const pnlNum = parseNumAny(obj.pnl ?? obj.pnl_bps ?? obj.net_pnl_bps);
  const delayMs = parseNumAny(obj.fill_delay_ms ?? obj.latency_ms ?? obj.delay_ms ?? obj.time_to_fill_ms);
  const adverseBps = parseNumAny(obj.adverse_bps ?? obj.mae_bps ?? obj.adverse_selection_bps);
  return { ts, symbol, side, price, qty, pnl, tsMs, pnlNum, delayMs, adverseBps };
}

function rowFromText(line: string): FillRow | null {
  const low = line.toLowerCase();
  if (!low.includes("fill")) return null;
  const pick1 = (re: RegExp): string => {
    const m = re.exec(line);
    return m && m[1] ? m[1] : "-";
  };
  const pick2 = (re: RegExp): string => {
    const m = re.exec(line);
    return m && m[2] ? m[2] : "-";
  };
  const pnlText = pick2(/\b(pnl_bps|pnl|net_pnl_bps)=([\-0-9.]+)/);
  const delayText = pick2(/\b(fill_delay_ms|latency_ms|delay_ms|time_to_fill_ms)=([\-0-9.]+)/);
  const adverseText = pick2(/\b(adverse_bps|mae_bps|adverse_selection_bps)=([\-0-9.]+)/);
  return {
    ts: line.slice(0, 19),
    symbol: pick1(/\b([A-Z]{3,}USDT)\b/),
    side: pick2(/\b(side|direction)=([a-zA-Z]+)/),
    price: pick2(/\b(price|fill_price|avg_fill_price)=([0-9.]+)/),
    qty: pick2(/\b(qty|size|amount|filled_qty)=([0-9.]+)/),
    pnl: pnlText,
    tsMs: parseTsAny(line.slice(0, 24)),
    pnlNum: parseNumAny(pnlText),
    delayMs: parseNumAny(delayText),
    adverseBps: parseNumAny(adverseText),
  };
}

export default function LiveMonitor() {
  const navigate = useNavigate();
  const [runTestsBusy, setRunTestsBusy] = useState(false);
  const [runTestsMsg, setRunTestsMsg] = useState("");
  const [fillSymbolFilter, setFillSymbolFilter] = useState("");
  const [fillSearch, setFillSearch] = useState("");
  const [tradeAgeAlertSec, setTradeAgeAlertSec] = useState<number>(() => {
    try {
      const raw = localStorage.getItem(LIVE_CFG_KEY);
      if (!raw) return DEFAULT_TRADE_AGE_ALERT_SEC;
      const j = JSON.parse(raw) as { tradeAgeAlertSec?: number };
      return Number.isFinite(j.tradeAgeAlertSec) ? Number(j.tradeAgeAlertSec) : DEFAULT_TRADE_AGE_ALERT_SEC;
    } catch {
      return DEFAULT_TRADE_AGE_ALERT_SEC;
    }
  });
  const [fillFlatlineAlertMin, setFillFlatlineAlertMin] = useState<number>(() => {
    try {
      const raw = localStorage.getItem(LIVE_CFG_KEY);
      if (!raw) return DEFAULT_FILL_FLATLINE_ALERT_MIN;
      const j = JSON.parse(raw) as { fillFlatlineAlertMin?: number };
      return Number.isFinite(j.fillFlatlineAlertMin) ? Number(j.fillFlatlineAlertMin) : DEFAULT_FILL_FLATLINE_ALERT_MIN;
    } catch {
      return DEFAULT_FILL_FLATLINE_ALERT_MIN;
    }
  });
  const [tradeSeries, setTradeSeries] = useState<number[]>([]);
  const [fillSeries, setFillSeries] = useState<number[]>([]);
  const runtimePoll = usePoll<RuntimeStatus>({
    fetcher: (signal) => api.runtime(signal),
    pollKey: "api:/runtime:live-monitor",
    intervalMs: 2000,
    staleAfterMs: 8000,
    retryInitialMs: 1000,
    retryMaxMs: 12000,
  });
  const opsPoll = usePoll<OpsHealthResponse>({
    fetcher: (signal) => api.opsHealth(signal),
    pollKey: "api:/ops/health:live-monitor",
    intervalMs: 5000,
    staleAfterMs: 15000,
  });
  const liqAlertPoll = usePoll<LiqAlertState>({
    fetcher: (signal) => api.liqAlertState(signal),
    pollKey: "api:/liq-alert-state:live-monitor",
    intervalMs: 10000,
    staleAfterMs: 30000,
  });
  const spreadStressPoll = usePoll<SpreadStressState>({
    fetcher: (signal) => api.spreadStressState(signal),
    pollKey: "api:/spread-stress-state:live-monitor",
    intervalMs: 10000,
    staleAfterMs: 30000,
  });
  const fillToxicityPoll = usePoll<FillToxicityState>({
    fetcher: (signal) => api.fillToxicityState(signal),
    pollKey: "api:/fill-toxicity-state:live-monitor",
    intervalMs: 10000,
    staleAfterMs: 30000,
  });
  const latencyStressPoll = usePoll<LatencyStressState>({
    fetcher: (signal) => api.latencyStressState(signal),
    pollKey: "api:/latency-stress-state:live-monitor",
    intervalMs: 10000,
    staleAfterMs: 30000,
  });
  const watchboardPoll = usePoll<WatchboardState>({
    fetcher: (signal) => api.watchboardState(signal),
    pollKey: "api:/watchboard-state:live-monitor",
    intervalMs: 10000,
    staleAfterMs: 30000,
  });
  const bookProxyPressurePoll = usePoll<BookProxyPressureState>({
    fetcher: (signal) => api.bookProxyPressureState(signal),
    pollKey: "api:/book-proxy-pressure-state:live-monitor",
    intervalMs: 10000,
    staleAfterMs: 30000,
  });
  const returnShockPoll = usePoll<ReturnShockState>({
    fetcher: (signal) => api.returnShockState(signal),
    pollKey: "api:/return-shock-state:live-monitor",
    intervalMs: 10000,
    staleAfterMs: 30000,
  });
  const volatilityBurstPoll = usePoll<VolatilityBurstState>({
    fetcher: (signal) => api.volatilityBurstState(signal),
    pollKey: "api:/volatility-burst-state:live-monitor",
    intervalMs: 10000,
    staleAfterMs: 30000,
  });
  const volumeVacuumPoll = usePoll<VolumeVacuumState>({
    fetcher: (signal) => api.volumeVacuumState(signal),
    pollKey: "api:/volume-vacuum-state:live-monitor",
    intervalMs: 10000,
    staleAfterMs: 30000,
  });
  const filesPoll = usePoll<LogFile[]>({
    fetcher: (signal) => api.logFiles(signal),
    pollKey: "api:/logs:live-monitor",
    intervalMs: 15000,
    staleAfterMs: 45000,
  });

  const collectorFile = useMemo(() => {
    const names = (filesPoll.data ?? []).map((x) => x.name);
    if (names.includes("microstructure_collector.log")) return "microstructure_collector.log";
    if (names.includes("collector_sim_bg.out.log")) return "collector_sim_bg.out.log";
    if (names.includes("watchdog.log")) return "watchdog.log";
    return names[0] ?? "microstructure_collector.log";
  }, [filesPoll.data]);
  const paperFile = useMemo(() => {
    const names = (filesPoll.data ?? []).map((x) => x.name);
    if (names.includes("paper_trades.jsonl")) return "paper_trades.jsonl";
    if (names.includes("paper_start_e2e.log")) return "paper_start_e2e.log";
    if (names.includes("run-bot-output.log")) return "run-bot-output.log";
    const alt = names.find((n) => n.toLowerCase().includes("paper") && (n.endsWith(".log") || n.endsWith(".jsonl")));
    return alt ?? "paper_trades.jsonl";
  }, [filesPoll.data]);

  const fetchTail = useCallback(
    (signal: AbortSignal) => api.logTail(collectorFile, 80, signal),
    [collectorFile],
  );
  const tailPoll = usePoll<LogTailResponse>({
    fetcher: fetchTail,
    pollKey: `api:/logs/tail:${collectorFile}:live-monitor`,
    intervalMs: 3000,
    staleAfterMs: 12000,
    enabled: !!collectorFile,
  });
  const fetchPaperTail = useCallback(
    (signal: AbortSignal) => api.logTail(paperFile, 80, signal),
    [paperFile],
  );
  const paperTailPoll = usePoll<LogTailResponse>({
    fetcher: fetchPaperTail,
    pollKey: `api:/logs/tail:${paperFile}:paper-monitor`,
    intervalMs: 3000,
    staleAfterMs: 12000,
    enabled: !!paperFile,
  });
  const scoreboardPoll = usePoll<Scoreboard>({
    fetcher: (signal) => api.scoreboard(signal),
    pollKey: "api:/scoreboard:live-monitor",
    intervalMs: 5000,
    staleAfterMs: 15000,
  });
  const liveMetricsPoll = usePoll<LiveMetricsResponse>({
    fetcher: (signal) => api.liveMetrics(signal),
    pollKey: "api:/live/metrics",
    intervalMs: 2000,
    staleAfterMs: 8000,
    retryInitialMs: 1000,
    retryMaxMs: 12000,
  });
  const paperRunPoll = usePoll<PaperRunStatusResponse>({
    fetcher: (signal) => api.paperRunStatus(signal),
    pollKey: "api:/live/paper-run",
    intervalMs: 3000,
    staleAfterMs: 12000,
    retryInitialMs: 1000,
    retryMaxMs: 12000,
  });
  const liveTestsPoll = usePoll<LiveMonitorTestsStatusResponse>({
    fetcher: (signal) => api.liveTestsStatus(80, signal),
    pollKey: "api:/live/tests/status",
    intervalMs: 3000,
    staleAfterMs: 12000,
    retryInitialMs: 1000,
    retryMaxMs: 12000,
  });
  const fetchPaperTailLong = useCallback(
    (signal: AbortSignal) => api.logTail(paperFile, 2000, signal),
    [paperFile],
  );
  const paperTailLongPoll = usePoll<LogTailResponse>({
    fetcher: fetchPaperTailLong,
    pollKey: `api:/logs/tail:${paperFile}:paper-monitor-long`,
    intervalMs: 15000,
    staleAfterMs: 45000,
    enabled: !!paperFile,
  });

  const mode: DegradedMode =
    runtimePoll.error || tailPoll.error || paperTailPoll.error || scoreboardPoll.error || paperTailLongPoll.error || liveMetricsPoll.error || paperRunPoll.error || liveTestsPoll.error
      ? "down"
      : runtimePoll.isStale || tailPoll.isStale || paperTailPoll.isStale || scoreboardPoll.isStale || paperTailLongPoll.isStale || liveMetricsPoll.isStale || paperRunPoll.isStale || liveTestsPoll.isStale
        ? "degraded"
        : "ok";

  const live = liveMetricsPoll.data;
  const paperRun = paperRunPoll.data;
  const tests = liveTestsPoll.data;
  const rt = (live?.runtime as RuntimeStatus | undefined) ?? runtimePoll.data;
  const collector = rt?.collector ?? {};
  const freshness = rt?.data_freshness ?? {};
  const db = rt?.database ?? {};
  const net = opsPoll.data?.network ?? {};
  const netStatus = opsPoll.data?.status?.network ?? "unknown";
  const scoreboard = live?.scoreboard ?? scoreboardPoll.data;
  const fillRatio = useMemo(() => {
    const orders = Number(scoreboard?.orders_total ?? 0);
    const fills = Number(scoreboard?.fills_total ?? 0);
    if (orders <= 0) return null;
    return (fills / orders) * 100.0;
  }, [scoreboard?.fills_total, scoreboard?.orders_total]);
  const blockedReasons = useMemo(() => {
    if (live?.blocked_reasons && live.blocked_reasons.length > 0) {
      return live.blocked_reasons.map((x) => ({ reason: x.reason, count: Number(x.count || 0) }));
    }
    const raw = scoreboard?.blocked_by_reason ?? {};
    return Object.entries(raw)
      .map(([reason, count]) => ({
        reason,
        count: typeof count === "number" ? count : Number(count) || 0,
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
  }, [live?.blocked_reasons, scoreboard?.blocked_by_reason]);
  const nowMs = Date.now();
  const lastFillAgeMin = useMemo(() => {
    const ms = toMillis(scoreboard?.last_fill_ts ?? null);
    if (!ms) return null;
    return (nowMs - ms) / 60000.0;
  }, [scoreboard?.last_fill_ts, nowMs]);
  const configuredTradeAge = Number(live?.alerts?.config?.trade_age_alert_sec ?? tradeAgeAlertSec);
  const configuredFillFlatline = Number(live?.alerts?.config?.fill_flatline_alert_min ?? fillFlatlineAlertMin);
  const tradeAgeAlert = Boolean(live?.alerts?.trade_age_alert ?? ((freshness.seconds_since_last_trade ?? 9999) > configuredTradeAge));
  const fillFlatlineAlert = Boolean(live?.alerts?.fill_flatline_alert ?? ((lastFillAgeMin ?? 9999) > configuredFillFlatline));
  const anyAlert = tradeAgeAlert || fillFlatlineAlert;
  const lastFills = useMemo<FillRow[]>(() => {
    if (live?.last_fills && live.last_fills.length > 0) {
      return live.last_fills.map((r) => ({
        ts: String(r.ts ?? "-"),
        symbol: String(r.symbol ?? "-"),
        side: String(r.side ?? "-"),
        price: String(r.price ?? "-"),
        qty: String(r.qty ?? "-"),
        pnl: String(r.pnl ?? "-"),
        tsMs: r.ts_ms ?? null,
        pnlNum: r.pnl_num ?? null,
        delayMs: r.delay_ms ?? null,
        adverseBps: r.adverse_bps ?? null,
      }));
    }
    const rows: FillRow[] = [];
    for (const line of paperTailPoll.data?.lines ?? []) {
      const obj = parseMaybeJson(line);
      const fromJson = obj ? rowFromJson(obj) : null;
      if (fromJson) {
        rows.push(fromJson);
        continue;
      }
      const fromText = rowFromText(line);
      if (fromText) rows.push(fromText);
    }
    return rows.slice(-5).reverse();
  }, [live?.last_fills, paperTailPoll.data?.lines]);
  const paperTailKpis = useMemo(() => {
    if (live?.tail_kpis) {
      return {
        orderCount: Number(live.tail_kpis.order_count ?? 0),
        fillCount: Number(live.tail_kpis.fill_count ?? 0),
        blockedCount: Number(live.tail_kpis.blocked_count ?? 0),
        fillPerOrder: live.tail_kpis.fill_per_order_pct ?? null,
        windowLines: Number(live.tail_kpis.window_lines ?? 0),
      };
    }
    const lines = paperTailPoll.data?.lines ?? [];
    let orderCount = 0;
    let fillCount = 0;
    let blockedCount = 0;
    for (const line of lines) {
      const low = line.toLowerCase();
      if (low.includes("order")) orderCount += 1;
      if (low.includes("fill")) fillCount += 1;
      if (low.includes("blocked") || low.includes("regime_mismatch") || low.includes("no_match")) blockedCount += 1;
    }
    const fillPerOrder = orderCount > 0 ? (fillCount / orderCount) * 100 : null;
    return { orderCount, fillCount, blockedCount, fillPerOrder, windowLines: lines.length };
  }, [live?.tail_kpis, paperTailPoll.data?.lines]);
  const longFillRows = useMemo<FillRow[]>(() => {
    const rows: FillRow[] = [];
    for (const line of paperTailLongPoll.data?.lines ?? []) {
      const obj = parseMaybeJson(line);
      const fromJson = obj ? rowFromJson(obj) : null;
      if (fromJson) {
        rows.push(fromJson);
        continue;
      }
      const fromText = rowFromText(line);
      if (fromText) rows.push(fromText);
    }
    return rows;
  }, [paperTailLongPoll.data?.lines]);
  const pnlStrip = useMemo(() => {
    if (live?.pnl_strip) {
      return {
        today: Number(live.pnl_strip.today ?? 0),
        h24: Number(live.pnl_strip.h24 ?? 0),
        d7: Number(live.pnl_strip.d7 ?? 0),
        sample: Number(live.pnl_strip.sample ?? 0),
      };
    }
    const now = Date.now();
    const sumWindow = (ms: number) =>
      longFillRows
        .filter((r) => (r.tsMs ?? 0) > 0 && now - Number(r.tsMs) <= ms)
        .reduce((a, r) => a + (r.pnlNum ?? 0), 0);
    const startDay = new Date();
    startDay.setHours(0, 0, 0, 0);
    const today = longFillRows
      .filter((r) => (r.tsMs ?? 0) >= startDay.getTime())
      .reduce((a, r) => a + (r.pnlNum ?? 0), 0);
    return {
      today,
      h24: sumWindow(24 * 60 * 60 * 1000),
      d7: sumWindow(7 * 24 * 60 * 60 * 1000),
      sample: longFillRows.length,
    };
  }, [live?.pnl_strip, longFillRows]);
  const fillQuality = useMemo(() => {
    if (live?.fill_quality) {
      return {
        avgDelayMs: live.fill_quality.avg_delay_ms ?? null,
        avgAdverseBps: live.fill_quality.avg_adverse_bps ?? null,
        withDelay: Number(live.fill_quality.with_delay ?? 0),
        withAdverse: Number(live.fill_quality.with_adverse ?? 0),
      };
    }
    const validDelay = longFillRows.map((r) => r.delayMs).filter((v): v is number => v != null && Number.isFinite(v));
    const validAdverse = longFillRows.map((r) => r.adverseBps).filter((v): v is number => v != null && Number.isFinite(v));
    const avg = (arr: number[]) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
    return {
      avgDelayMs: avg(validDelay),
      avgAdverseBps: avg(validAdverse),
      withDelay: validDelay.length,
      withAdverse: validAdverse.length,
    };
  }, [live?.fill_quality, longFillRows]);
  const visibleFills = useMemo(() => {
    const sym = fillSymbolFilter.trim().toUpperCase();
    const q = fillSearch.trim().toLowerCase();
    return lastFills.filter((r) => {
      if (sym && r.symbol.toUpperCase() !== sym) return false;
      if (q && !`${r.ts} ${r.symbol} ${r.side} ${r.price} ${r.qty} ${r.pnl}`.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [lastFills, fillSearch, fillSymbolFilter]);
  const effectivePaperFile = live?.paper_file ?? paperFile;
  const paperSession = paperRun?.session;
  const paperStartup = paperRun?.startup_manifest;
  const paperEntry = paperRun?.entry_state;
  const paperDiagnosis = paperRun?.diagnosis;
  const paperTradeState = paperRun?.trade_state;
  const paperChain = paperRun?.process_chain;
  const paperReasonBreakdown = paperRun?.reason_breakdown;
  const paperSymbols = paperRun?.symbols ?? [];
  const paperExecutionLabel = paperStartup?.paper_execution_label || scoreboardPoll.data?.paper_execution_label || "-";
  const paperExecutionMode = paperStartup?.paper_execution_mode || scoreboardPoll.data?.paper_execution_mode || "-";
  const paperFillModel = paperStartup?.paper_fill_model || scoreboardPoll.data?.paper_fill_model || "-";
  const paperSafeContract =
    paperStartup?.paper_allow_live_private_api === false &&
    paperStartup?.binance_testnet === true &&
    paperStartup?.paper_profile_active !== false;
  useEffect(() => {
    try {
      localStorage.setItem(LIVE_CFG_KEY, JSON.stringify({ tradeAgeAlertSec, fillFlatlineAlertMin }));
    } catch {
      // ignore
    }
  }, [fillFlatlineAlertMin, tradeAgeAlertSec]);
  useEffect(() => {
    if (live?.trends?.trades_per_sec && live.trends.trades_per_sec.length > 0) {
      setTradeSeries(live.trends.trades_per_sec.map((x) => Number(x || 0)));
      return;
    }
    const v = collector.trades_per_sec_60s;
    if (typeof v !== "number" || Number.isNaN(v)) return;
    setTradeSeries((prev) => [...prev.slice(-29), v]);
  }, [collector.trades_per_sec_60s, live?.trends?.trades_per_sec]);
  useEffect(() => {
    if (live?.trends?.fills_tail && live.trends.fills_tail.length > 0) {
      setFillSeries(live.trends.fills_tail.map((x) => Number(x || 0)));
      return;
    }
    setFillSeries((prev) => [...prev.slice(-29), paperTailKpis.fillCount]);
  }, [paperTailKpis.fillCount, live?.trends?.fills_tail]);

  function exportSnapshot(): void {
    const payload = {
      ts_utc: new Date().toISOString(),
      alerts: { tradeAgeAlert, fillFlatlineAlert, anyAlert, tradeAgeAlertSec, fillFlatlineAlertMin },
      runtime: rt ?? {},
      scoreboard: scoreboard ?? {},
      pnlStrip,
      fillQuality,
      blockedReasons,
      lastFills: visibleFills,
      paperTailKpis,
      collectorFile,
      paperFile,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `LIVE_MONITOR_SNAPSHOT_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function copyTestCommand(): Promise<void> {
    const cmd = tests?.run_command ?? "powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\run_live_monitor_tests.ps1";
    try {
      await navigator.clipboard.writeText(cmd);
    } catch {
      // ignore clipboard failures in restricted browsers
    }
  }

  async function runTestsNow(): Promise<void> {
    if (runTestsBusy) return;
    setRunTestsBusy(true);
    setRunTestsMsg("starting...");
    try {
      const out = await api.liveTestsRun();
      setRunTestsMsg(`started pid=${out.pid ?? "-"} log=${out.runner_log ?? "-"}`);
      liveTestsPoll.refresh();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "run failed";
      setRunTestsMsg(msg);
    } finally {
      setRunTestsBusy(false);
    }
  }

  const sectionHeader = (label: string) => (
    <div style={{ borderBottom: "1px solid var(--border)", paddingBottom: 4, color: "var(--muted)", fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase" as const, marginTop: 4 }}>
      {label}
    </div>
  );

  const watchboardTopLevel = (() => {
    const sc = watchboardPoll.data?.summary?.state_counts ?? {};
    if ((sc["severe"] ?? 0) > 0) return "severe";
    if ((sc["elevated"] ?? 0) > 0) return "elevated";
    return watchboardPoll.data?.available ? "quiet" : undefined;
  })();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <PageGuide
        icon="LIVE"
        titleTr="Canlı Veri Monitörü"
        titleEn="Live Data Monitor"
        subtitleTr="Microstructure akışını tek sayfada canlı izle."
        subtitleEn="Watch microstructure flow and collector health on one page."
        items={[
          {
            icon: "1",
            titleTr: "Hiz",
            titleEn: "Flow",
            descTr: "Trades/sec ve Mark/sec veri akışının hızını gösterir.",
            descEn: "Trades/sec and Mark/sec show data feed throughput.",
          },
          {
            icon: "2",
            titleTr: "Tazelik",
            titleEn: "Freshness",
            descTr: "Last trade age dusukse veri canli demektir.",
            descEn: "Low last-trade age means feed is fresh.",
          },
          {
            icon: "3",
            titleTr: "Log Akisi",
            titleEn: "Collector Log",
            descTr: "Collector log son satirlari otomatik yenilenir.",
            descEn: "Collector log tail auto-refreshes continuously.",
          },
        ]}
      />

      <DegradedBanner mode={mode} message={paperRunPoll.error?.message ?? liveMetricsPoll.error?.message ?? runtimePoll.error?.message ?? tailPoll.error?.message ?? paperTailPoll.error?.message ?? scoreboardPoll.error?.message ?? paperTailLongPoll.error?.message} />

      {sectionHeader("Paper Run Diagnosis")}
      <AsyncState loading={paperRunPoll.isLoading} error={paperRunPoll.error} loadingText="Loading paper run diagnosis...">
        <div style={{ display: "grid", gap: 12 }}>
          <div
            className="card"
            style={{
              borderColor:
                paperDiagnosis?.severity === "critical"
                  ? "var(--red)"
                  : paperDiagnosis?.severity === "warning"
                    ? "var(--yellow)"
                    : "var(--border)",
            }}
          >
            <div className="card-title self-help" data-help="Paper run neden trade uretmiyor ya da neden bozuk; en yuksek oncelikli tani karti.">
              Why No Trade?
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 8 }}>
              <span className={`badge ${paperSession?.status === "down" ? "badge-red" : paperSession?.status === "degraded" ? "badge-yellow" : "badge-green"}`}>
                {String(paperSession?.status ?? "unknown").toUpperCase()}
              </span>
              <span className={`badge ${paperDiagnosis?.severity === "critical" ? "badge-red" : paperDiagnosis?.severity === "warning" ? "badge-yellow" : "badge-green"}`}>
                {String(paperDiagnosis?.code ?? "unknown").replace(/_/g, " ")}
              </span>
              <span className="badge badge-gray">{paperSession?.active_symbols?.join(", ") || "-"}</span>
            </div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>{paperDiagnosis?.summary ?? "paper run status unknown"}</div>
            <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 6 }}>{paperDiagnosis?.detail ?? "awaiting telemetry"}</div>
            {paperDiagnosis?.execution_context ? (
              <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 6 }}>
                execution_context={paperDiagnosis.execution_context}
              </div>
            ) : null}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 10 }}>
            <div className="card self-help" data-help="Session health: paper zinciri ayakta mi, ne kadar suredir calisiyor, hangi semboller aktif.">
              <div className="card-title">Session Health</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
                <span className={`badge ${paperSession?.status === "running" ? "badge-green" : paperSession?.status === "degraded" ? "badge-yellow" : "badge-red"}`}>
                  {String(paperSession?.status ?? "unknown").toUpperCase()}
                </span>
                <span className="badge badge-gray">uptime {fmtDuration(paperSession?.uptime_sec)}</span>
                <span className="badge badge-gray">telemetry {paperSession?.telemetry_present ? "ON" : "OFF"}</span>
                <span className={`badge ${paperExecutionMode === "router_blocked" ? "badge-yellow" : "badge-gray"}`}>
                  {paperExecutionLabel}
                </span>
                <span className="badge badge-gray">fill {paperFillModel}</span>
              </div>
              <div style={{ color: "var(--muted)", fontSize: 12 }}>started={paperSession?.started_ts ?? "-"}</div>
              <div style={{ color: "var(--muted)", fontSize: 12 }}>telemetry_age={paperSession?.telemetry_age_sec == null ? "-" : `${num(paperSession.telemetry_age_sec, 1)}s`}</div>
              <div style={{ color: "var(--muted)", fontSize: 12 }}>symbols={paperSession?.active_symbols?.join(", ") || "-"}</div>
              <div style={{ color: "var(--muted)", fontSize: 12 }}>execution_mode={paperExecutionMode}</div>
              <div style={{ color: paperSafeContract ? "var(--muted)" : "var(--yellow)", fontSize: 12 }}>
                {paperExecutionMode === "router_blocked"
                  ? "router_blocked mode rehearses runtime flow without creating fills"
                  : `paper fill model=${paperFillModel}`}
              </div>
              <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 6 }}>{paperChain?.summary ?? "process chain unknown"}</div>
            </div>

            <div className="card self-help" data-help="Entry readiness: gate, guard ve veri katmanlari entry acisindan uygun mu.">
              <div className="card-title">Entry Readiness</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
                <span className={`badge ${paperEntry?.allow_entries ? "badge-green" : "badge-red"}`}>
                  allow_entries {paperEntry?.allow_entries ? "ON" : "OFF"}
                </span>
                <span className={`badge ${paperEntry?.runtime_gate_degraded ? "badge-red" : "badge-green"}`}>
                  gate {paperEntry?.runtime_gate_degraded ? "DEGRADED" : "OK"}
                </span>
                <span className={`badge ${paperEntry?.data_state === "ok" ? "badge-green" : paperEntry?.data_state === "degraded" ? "badge-yellow" : "badge-red"}`}>
                  data {String(paperEntry?.data_state ?? "unknown").toUpperCase()}
                </span>
                <span className={`badge ${paperEntry?.risk_state === "blocked" ? "badge-red" : "badge-green"}`}>
                  risk {String(paperEntry?.risk_state ?? "ok").toUpperCase()}
                </span>
                <span className={`badge ${paperEntry?.regime_state === "blocked" ? "badge-yellow" : "badge-green"}`}>
                  regime {String(paperEntry?.regime_state ?? "ok").toUpperCase()}
                </span>
              </div>
              <div style={{ color: "var(--muted)", fontSize: 12 }}>guard_mode={paperEntry?.guard_mode ?? "-"}</div>
              <div style={{ color: "var(--muted)", fontSize: 12 }}>runtime_gate_reason={paperEntry?.runtime_gate_reason ?? "-"}</div>
            </div>

            <div className="card self-help" data-help="Trade snapshot: hic trade yok mu, en son trade ne zaman, paper trade DB gorunuyor mu.">
              <div className="card-title">Trade Snapshot</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
                <span className={`badge ${(paperTradeState?.trade_count ?? 0) > 0 ? "badge-green" : "badge-yellow"}`}>
                  trades {paperTradeState?.trade_count ?? 0}
                </span>
                <span className={`badge ${paperTradeState?.no_trades_yet ? "badge-yellow" : "badge-green"}`}>
                  {paperTradeState?.no_trades_yet ? "NO TRADES YET" : "HAS TRADES"}
                </span>
                <span className={`badge ${paperTradeState?.db_present ? "badge-green" : "badge-red"}`}>
                  db {paperTradeState?.db_present ? "PRESENT" : "MISSING"}
                </span>
              </div>
              <div style={{ color: "var(--muted)", fontSize: 12 }}>last_trade={paperTradeState?.last_trade_ts ?? "-"}</div>
              {paperTradeState?.execution_note ? (
                <div style={{ color: "var(--muted)", fontSize: 12 }}>{paperTradeState.execution_note}</div>
              ) : null}
              <div style={{ color: "var(--muted)", fontSize: 12, wordBreak: "break-all" }}>{paperTradeState?.db_path ?? "-"}</div>
            </div>

            <div className="card self-help" data-help="Reason breakdown: son telemetry blockerlarinin kategori dagilimi.">
              <div className="card-title">Reason Breakdown</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 8 }}>
                <div className="card"><div className="card-title">Signal</div><div style={{ fontSize: 16, fontWeight: 700 }}>{paperReasonBreakdown?.signal_not_present ?? 0}</div></div>
                <div className="card"><div className="card-title">Gate</div><div style={{ fontSize: 16, fontWeight: 700 }}>{paperReasonBreakdown?.gate_blocked ?? 0}</div></div>
                <div className="card"><div className="card-title">Data</div><div style={{ fontSize: 16, fontWeight: 700 }}>{paperReasonBreakdown?.data_degraded ?? 0}</div></div>
                <div className="card"><div className="card-title">Risk</div><div style={{ fontSize: 16, fontWeight: 700 }}>{paperReasonBreakdown?.risk_blocked ?? 0}</div></div>
                <div className="card"><div className="card-title">Regime</div><div style={{ fontSize: 16, fontWeight: 700 }}>{paperReasonBreakdown?.regime_blocked ?? 0}</div></div>
                <div className="card"><div className="card-title">Unknown</div><div style={{ fontSize: 16, fontWeight: 700 }}>{paperReasonBreakdown?.unknown ?? 0}</div></div>
              </div>
            </div>
          </div>

          <div className="card self-help" data-help="Symbol diagnostics: her sembol icin son blocker, son signal ve belief zamani.">
            <div className="card-title">Symbol Diagnostics</div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ color: "var(--muted)", textAlign: "left" }}>
                    <th style={{ padding: "6px 4px" }}>Symbol</th>
                    <th style={{ padding: "6px 4px" }}>Last blocker</th>
                    <th style={{ padding: "6px 4px" }}>Blocked count</th>
                    <th style={{ padding: "6px 4px" }}>Last signal</th>
                    <th style={{ padding: "6px 4px" }}>Last belief</th>
                  </tr>
                </thead>
                <tbody>
                  {paperSymbols.length > 0 ? paperSymbols.map((row) => (
                    <tr key={row.symbol}>
                      <td style={{ padding: "6px 4px", fontWeight: 700 }}>{row.symbol}</td>
                      <td style={{ padding: "6px 4px" }}>{row.last_blocker_reason ?? "-"}</td>
                      <td style={{ padding: "6px 4px" }}>{row.recent_blocked_count ?? 0}</td>
                      <td style={{ padding: "6px 4px" }}>{row.last_signal_ts ?? "-"}</td>
                      <td style={{ padding: "6px 4px" }}>{row.last_belief_ts ?? "-"}</td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan={5} style={{ padding: "8px 4px", color: "var(--muted)" }}>No symbol diagnostics yet.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </AsyncState>

      {/* ── LIVE STATUS ─────────────────────────────────────────── */}
      {sectionHeader("Live Status")}

      <AsyncState loading={runtimePoll.isLoading} error={runtimePoll.error} loadingText="Loading live monitor...">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 10 }}>
          <div className="card self-help" data-help="Collector process ayakta mı?">
            <div className="card-title">Collector</div>
            <div style={{ fontSize: 20, fontWeight: 700, color: collector.alive ? "var(--green)" : "var(--red)" }}>
              {collector.alive ? "ALIVE" : "DOWN"}
            </div>
            <div style={{ color: "var(--muted)", fontSize: 11 }}>uptime={num(collector.uptime_sec, 0)}s</div>
          </div>
          <div className="card self-help" data-help="Son 60 saniyede saniye başına trade adedi.">
            <div className="card-title">Trades / sec</div>
            <div style={{ fontSize: 20, fontWeight: 700 }}>{num(collector.trades_per_sec_60s, 1)}</div>
          </div>
          <div className="card self-help" data-help="Son 60 saniyede saniye başına mark price update adedi.">
            <div className="card-title">Mark / sec</div>
            <div style={{ fontSize: 20, fontWeight: 700 }}>{num(collector.mark_per_sec_60s, 1)}</div>
          </div>
          <div className="card self-help" data-help="En son trade verisinin su ana gore kac saniye eski oldugu.">
            <div className="card-title">Last Trade Age</div>
            <div style={{ fontSize: 20, fontWeight: 700, color: (freshness.seconds_since_last_trade ?? 999) <= 5 ? "var(--green)" : "var(--yellow)" }}>
              {num(freshness.seconds_since_last_trade, 2)}s
            </div>
            <div style={{ color: "var(--muted)", fontSize: 11 }}>status={freshness.status ?? "-"}</div>
          </div>
          <div className="card self-help" data-help="Microstructure DB boyutu ve son yazim zamani.">
            <div className="card-title">DB</div>
            <div style={{ fontSize: 16, fontWeight: 700 }}>{dbSizeMb(db.size_bytes)}</div>
            <div style={{ color: "var(--muted)", fontSize: 11 }}>{db.last_write_ts ?? "-"}</div>
          </div>
          <div className="card self-help" data-help="Collector network sagligi ve reconnect/rate-limit durumu.">
            <div className="card-title">Network</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: netStatus === "warning" ? "var(--yellow)" : "var(--green)" }}>
              {String(netStatus).toUpperCase()}
            </div>
            <div style={{ color: "var(--muted)", fontSize: 11 }}>
              reconnects_5m={net.reconnects_last_5m ?? "-"} usage={num(net.usage_pct, 1)}%
            </div>
          </div>
        </div>
      </AsyncState>

      <div className="card">
        <div className="card-title self-help" data-help="Canlı alarm göstergeleri: trade yaşı ve fill flatline durumlarını izler.">
          Live Alerts
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <span className={`badge ${anyAlert ? "badge-yellow" : "badge-green"}`}>
            {anyAlert ? "ALERT" : "OK"}
          </span>
          <span className={`badge ${tradeAgeAlert ? "badge-red" : "badge-green"}`}>
            trade_age {tradeAgeAlert ? "HIGH" : "OK"} ({num(freshness.seconds_since_last_trade, 2)}s)
          </span>
          <span className={`badge ${fillFlatlineAlert ? "badge-red" : "badge-green"}`}>
            fill_flow {fillFlatlineAlert ? "FLATLINE" : "OK"} ({lastFillAgeMin == null ? "-" : `${num(lastFillAgeMin, 1)}m`})
          </span>
          <span className="badge badge-gray">
            {`thresholds: trade>${configuredTradeAge}s fill>${configuredFillFlatline}m`}
          </span>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div className="card">
          <div className="card-title self-help" data-help="Paper trade canlı özet: order/fill/block sayıları ve son fill zamanı.">
            Paper Trade Summary
          </div>
          <AsyncState loading={scoreboardPoll.isLoading} error={scoreboardPoll.error} loadingText="Loading...">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
              <div className="card self-help" data-help="Paper mod acik mi?">
                <div className="card-title">Mode</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: scoreboardPoll.data?.paper_trading ? "var(--yellow)" : "var(--green)" }}>
                  {scoreboardPoll.data?.paper_trading ? "PAPER" : "LIVE"}
                </div>
                <div style={{ color: "var(--muted)", fontSize: 10 }}>{paperExecutionLabel}</div>
              </div>
              <div className="card">
                <div className="card-title">Orders</div>
                <div style={{ fontSize: 16, fontWeight: 700 }}>{scoreboardPoll.data?.orders_total ?? 0}</div>
              </div>
              <div className="card">
                <div className="card-title">Fills</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: "var(--green)" }}>{scoreboardPoll.data?.fills_total ?? 0}</div>
              </div>
              <div className="card">
                <div className="card-title">Fill%</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: (fillRatio ?? 0) >= 40 ? "var(--green)" : "var(--yellow)" }}>
                  {fillRatio == null ? "-" : `${fillRatio.toFixed(1)}%`}
                </div>
                <div style={{ color: "var(--muted)", fontSize: 10 }}>{scoreboard?.fills_total ?? 0}/{scoreboard?.orders_total ?? 0}</div>
              </div>
              <div className="card">
                <div className="card-title">Blocked</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: (scoreboardPoll.data?.blocked_total ?? 0) > 0 ? "var(--yellow)" : "var(--text)" }}>
                  {scoreboardPoll.data?.blocked_total ?? 0}
                </div>
              </div>
              <div className="card">
                <div className="card-title">Last Fill</div>
                <div style={{ fontSize: 11, fontWeight: 700 }}>{scoreboardPoll.data?.last_fill_ts ?? "-"}</div>
              </div>
            </div>
          </AsyncState>
        </div>

        <div className="card">
          <div className="card-title self-help" data-help="PnL strip: tail tabanli today/24h/7d toplamlari.">
            PnL Strip
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 8 }}>
            <div className="card">
              <div className="card-title">Today</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: pnlStrip.today >= 0 ? "var(--green)" : "var(--red)" }}>{num(pnlStrip.today, 2)}</div>
            </div>
            <div className="card">
              <div className="card-title">24h</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: pnlStrip.h24 >= 0 ? "var(--green)" : "var(--red)" }}>{num(pnlStrip.h24, 2)}</div>
            </div>
            <div className="card">
              <div className="card-title">7d</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: pnlStrip.d7 >= 0 ? "var(--green)" : "var(--red)" }}>{num(pnlStrip.d7, 2)}</div>
            </div>
            <div className="card">
              <div className="card-title">Samples</div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>{pnlStrip.sample}</div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div className="card">
          <div className="card-title self-help" data-help="Fill quality: ortalama fill delay ve adverse bps (veri varsa).">
            Fill Quality
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <div className="card">
              <div className="card-title">Avg Delay (ms)</div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>{num(fillQuality.avgDelayMs, 1)}</div>
              <div style={{ color: "var(--muted)", fontSize: 11 }}>n={fillQuality.withDelay}</div>
            </div>
            <div className="card">
              <div className="card-title">Avg Adverse (bps)</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: (fillQuality.avgAdverseBps ?? 0) <= 0 ? "var(--green)" : "var(--yellow)" }}>
                {num(fillQuality.avgAdverseBps, 2)}
              </div>
              <div style={{ color: "var(--muted)", fontSize: 11 }}>n={fillQuality.withAdverse}</div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-title self-help" data-help="Paper log tail penceresinde order/fill/block yogunlugu.">
            Tail KPIs
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
            <div className="card">
              <div className="card-title">Orders</div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>{paperTailKpis.orderCount}</div>
            </div>
            <div className="card">
              <div className="card-title">Fills</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: "var(--green)" }}>{paperTailKpis.fillCount}</div>
            </div>
            <div className="card">
              <div className="card-title">Blocked</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: paperTailKpis.blockedCount > 0 ? "var(--yellow)" : "var(--text)" }}>{paperTailKpis.blockedCount}</div>
            </div>
            <div className="card">
              <div className="card-title">Fill Ratio</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: (paperTailKpis.fillPerOrder ?? 0) >= 40 ? "var(--green)" : "var(--yellow)" }}>
                {paperTailKpis.fillPerOrder == null ? "-" : `${paperTailKpis.fillPerOrder.toFixed(1)}%`}
              </div>
            </div>
            <div className="card">
              <div className="card-title">Lines</div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>{paperTailKpis.windowLines}</div>
            </div>
            <div className="card">
              <div className="card-title">Trades/sec</div>
              <Spark values={tradeSeries} color="var(--accent)" />
            </div>
          </div>
        </div>
      </div>

      {/* ── RISK MONITOR ─────────────────────────────────────────── */}
      {sectionHeader("Risk Monitor — 9 Lanes")}

      <div className="card">
        <div className="card-title">Risk Radar</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {([
            { label: "Watchboard", level: watchboardTopLevel },
            { label: "Liq", level: liqAlertPoll.data?.state?.level },
            { label: "SpreadStress", level: spreadStressPoll.data?.state?.level },
            { label: "Toxicity", level: fillToxicityPoll.data?.state?.level },
            { label: "Latency", level: latencyStressPoll.data?.state?.level },
            { label: "BookProxy", level: bookProxyPressurePoll.data?.state?.level },
            { label: "RetShock", level: returnShockPoll.data?.state?.level },
            { label: "VolBurst", level: volatilityBurstPoll.data?.state?.level },
            { label: "VolVacuum", level: volumeVacuumPoll.data?.state?.level },
          ] as Array<{ label: string; level: string | undefined }>).map(({ label, level }) => (
            <span key={label} style={{
              padding: "3px 10px", borderRadius: 12, fontSize: 11, fontWeight: 600,
              background: level === "severe" ? "var(--danger-bg)" : level === "elevated" ? "var(--warn-bg)" : level ? "var(--ok-bg)" : "var(--surface-3)",
              color: level === "severe" ? "var(--red)" : level === "elevated" ? "var(--yellow)" : level ? "var(--green)" : "var(--muted)",
              border: `1px solid ${level === "severe" ? "rgba(255,64,64,0.3)" : level === "elevated" ? "rgba(245,166,35,0.3)" : level ? "rgba(0,201,107,0.3)" : "var(--border)"}`,
            }}>
              {label}: {level ?? "–"}
            </span>
          ))}
        </div>
      </div>

      <WatchboardCard data={watchboardPoll.data ?? null} />

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
        <LiqAlertCard data={liqAlertPoll.data ?? null} />
        <SpreadStressCard data={spreadStressPoll.data ?? null} />
        <FillToxicityCard data={fillToxicityPoll.data ?? null} />
        <LatencyStressCard data={latencyStressPoll.data ?? null} />
        <BookProxyPressureCard data={bookProxyPressurePoll.data ?? null} />
        <ReturnShockCard data={returnShockPoll.data ?? null} />
        <VolatilityBurstCard data={volatilityBurstPoll.data ?? null} />
        <VolumeVacuumCard data={volumeVacuumPoll.data ?? null} />
      </div>

      {/* ── EXECUTION ────────────────────────────────────────────── */}
      {sectionHeader("Execution")}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 12 }}>
        <div className="card">
          <div className="card-title self-help" data-help="Son blocker reason dagilimi (top 10).">
            Blocked Reasons
          </div>
          <AsyncState loading={scoreboardPoll.isLoading} error={scoreboardPoll.error} isEmpty={blockedReasons.length === 0} emptyText="No blocked reasons">
            <table>
              <thead>
                <tr><th>reason</th><th>count</th></tr>
              </thead>
              <tbody>
                {blockedReasons.map((r) => (
                  <tr key={r.reason}>
                    <td style={{ color: "var(--muted)" }}>{r.reason}</td>
                    <td>{r.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </AsyncState>
        </div>

        <div className="card">
          <div className="card-title self-help" data-help="Paper log icinden yakalanan son 5 fill olayi.">
            Last 5 Fills
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
            <input className="self-help" data-help="Sembol filtresi (ornek: ETHUSDT)." placeholder="Symbol filter" value={fillSymbolFilter}
              onChange={(e) => setFillSymbolFilter(e.target.value.toUpperCase())}
              style={{ padding: "4px 8px", borderRadius: 4, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)", width: 140 }} />
            <input className="self-help" data-help="Fill tablosunda genel metin aramasi." placeholder="Search fills" value={fillSearch}
              onChange={(e) => setFillSearch(e.target.value)}
              style={{ padding: "4px 8px", borderRadius: 4, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)", width: 180 }} />
            <span style={{ marginLeft: "auto", color: "var(--muted)", fontSize: 11 }}>rows={visibleFills.length}</span>
          </div>
          <AsyncState loading={paperTailPoll.isLoading} error={paperTailPoll.error} isEmpty={visibleFills.length === 0} emptyText="No fill lines in log tail">
            <table>
              <thead>
                <tr><th>ts</th><th>symbol</th><th>side</th><th>price</th><th>qty</th><th>pnl</th></tr>
              </thead>
              <tbody>
                {visibleFills.map((r, i) => (
                  <tr key={`${r.ts}_${r.symbol}_${i}`}>
                    <td style={{ color: "var(--muted)" }}>{r.ts}</td>
                    <td>{r.symbol}</td><td>{r.side}</td><td>{r.price}</td><td>{r.qty}</td><td>{r.pnl}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </AsyncState>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div className="card">
          <div className="card-title self-help" data-help="Collector log dosyasının son 80 satırı, 3 saniyede bir yenilenir.">
            Collector Log ({collectorFile})
          </div>
          <AsyncState loading={tailPoll.isLoading} error={tailPoll.error} isEmpty={(tailPoll.data?.lines?.length ?? 0) === 0} emptyText="No collector log lines">
            <pre style={{ maxHeight: 280, overflowY: "auto", fontSize: 11, padding: 8, background: "var(--bg)", borderRadius: 4, border: "1px solid var(--border)", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
              {(tailPoll.data?.lines ?? []).join("\n")}
            </pre>
          </AsyncState>
        </div>
        <div className="card">
          <div className="card-title self-help" data-help="Paper trade log son satirlari (3 saniyede bir yenilenir).">
            Paper Log ({effectivePaperFile})
          </div>
          <AsyncState loading={paperTailPoll.isLoading} error={paperTailPoll.error} isEmpty={(paperTailPoll.data?.lines?.length ?? 0) === 0} emptyText="No paper trade log lines">
            <pre style={{ maxHeight: 280, overflowY: "auto", fontSize: 11, padding: 8, background: "var(--bg)", borderRadius: 4, border: "1px solid var(--border)", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
              {(paperTailPoll.data?.lines ?? []).join("\n")}
            </pre>
          </AsyncState>
        </div>
      </div>

      <div className="card">
        <div className="card-title self-help" data-help="Son 30 ornek icin trades/sec ve fill count sparkline grafigi.">
          Mini Trends
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
          <div className="card">
            <div className="card-title">Trades/sec (30)</div>
            <Spark values={tradeSeries} color="var(--accent)" />
          </div>
          <div className="card">
            <div className="card-title">Fills in tail (30)</div>
            <Spark values={fillSeries} color="var(--green)" />
          </div>
        </div>
      </div>

      {/* ── TOOLS ────────────────────────────────────────────────── */}
      {sectionHeader("Tools")}

      <div className="card">
        <div className="card-title self-help" data-help="Hizli gecis: detayli log/tower/recovery ekranlarina tek tik.">
          Quick Actions
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button className="guide-toggle self-help" data-help="Incident ve runtime sağlığı için Control Tower'a git." onClick={() => navigate("/tower")}>Open Tower</button>
          <button className="guide-toggle self-help" data-help="Log analiz ekranına git." onClick={() => navigate("/logs")}>Open Logs</button>
          <button className="guide-toggle self-help" data-help="Backend kurtarma adımlarını açar." onClick={() => navigate("/recovery")}>Open Recovery</button>
          <button className="guide-toggle self-help" data-help="Sinyal ve trade event ekranına geç." onClick={() => navigate("/trades")}>Open Signals</button>
          <button className="guide-toggle self-help" data-help="Mevcut monitor durumunu JSON olarak indir." onClick={exportSnapshot}>Export Snapshot</button>
        </div>
      </div>

      <div className="card">
        <div className="card-title self-help" data-help="run_live_monitor_tests script'inin son çalışma durumunu canlı izler.">
          Diagnostics — Live Monitor Tests
        </div>
        <AsyncState loading={liveTestsPoll.isLoading} error={liveTestsPoll.error} loadingText="Loading diagnostics...">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
            <div className="card">
              <div className="card-title">State</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: tests?.state === "passed" ? "var(--green)" : tests?.state === "failed" ? "var(--red)" : "var(--yellow)" }}>
                {(tests?.state ?? "unknown").toUpperCase()}
              </div>
              <div style={{ color: "var(--muted)", fontSize: 11 }}>stage={tests?.stage ?? "-"}</div>
            </div>
            <div className="card">
              <div className="card-title">Age</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{num(tests?.status_age_sec ?? null, 1)}s</div>
              <div style={{ color: "var(--muted)", fontSize: 11 }}>{tests?.ts_utc ?? "-"}</div>
            </div>
            <div className="card">
              <div className="card-title">Checklist</div>
              <div style={{ fontSize: 12, color: "var(--muted)" }}>
                backend={tests?.backend_ok ? "OK" : "NO"} typecheck={tests?.frontend_typecheck_ok ? "OK" : "NO"} smoke={tests?.frontend_smoke_ok ? "OK" : tests?.frontend_smoke_skipped ? "SKIP" : "NO"}
              </div>
              <div style={{ fontSize: 11, color: "var(--muted)" }}>strict={tests?.strict_mode ? "1" : "0"} pid={tests?.pid ?? "-"}</div>
            </div>
          </div>
          <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="guide-toggle self-help" data-help="Backend'e test paketi calistir komutu yollar." onClick={runTestsNow} disabled={runTestsBusy}>
              {runTestsBusy ? "Starting..." : "Run Tests Now"}
            </button>
            <button className="guide-toggle self-help" data-help="Test komutunu kopyala ve PowerShell'e yapistir." onClick={copyTestCommand}>
              Copy Test Command
            </button>
            <span className="badge badge-gray" style={{ maxWidth: "100%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {tests?.run_command ?? "powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\run_live_monitor_tests.ps1"}
            </span>
          </div>
          {runTestsMsg ? <div style={{ marginTop: 6, color: "var(--muted)", fontSize: 12 }}>{runTestsMsg}</div> : null}
          <div style={{ marginTop: 8, color: "var(--muted)", fontSize: 12 }}>{tests?.message ?? "-"}</div>
          <pre style={{ marginTop: 8, maxHeight: 180, overflowY: "auto", fontSize: 11, padding: 8, background: "var(--bg)", borderRadius: 4, border: "1px solid var(--border)", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {(tests?.log_tail ?? []).join("\n")}
          </pre>
        </AsyncState>
      </div>

      <div className="card">
        <div className="card-title self-help" data-help="Alarm esiklerini buradan degistir; localStorage'a kaydolur.">
          Alert Config
        </div>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
          <label style={{ color: "var(--muted)", fontSize: 12 }}>
            trade_age_sec
            <input type="number" min={1} value={tradeAgeAlertSec}
              onChange={(e) => setTradeAgeAlertSec(Math.max(1, Number(e.target.value) || DEFAULT_TRADE_AGE_ALERT_SEC))}
              style={{ marginLeft: 6, width: 84 }} />
          </label>
          <label style={{ color: "var(--muted)", fontSize: 12 }}>
            fill_flatline_min
            <input type="number" min={1} value={fillFlatlineAlertMin}
              onChange={(e) => setFillFlatlineAlertMin(Math.max(1, Number(e.target.value) || DEFAULT_FILL_FLATLINE_ALERT_MIN))}
              style={{ marginLeft: 6, width: 84 }} />
          </label>
        </div>
      </div>
    </div>
  );
}
