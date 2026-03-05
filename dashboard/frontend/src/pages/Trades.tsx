import React, { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { DataQualityEvent, SignalEvent, StabilityEvent } from "../api/types";
import AsyncState from "../components/AsyncState";
import DegradedBanner, { type DegradedMode } from "../components/DegradedBanner";
import PageGuide from "../components/PageGuide";
import TermTip from "../components/TermTip";
import { usePoll } from "../hooks/usePoll";
import { useSearchParams } from "react-router-dom";

function TabBtn({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "4px 14px",
        borderRadius: 4,
        border: "1px solid " + (active ? "var(--accent)" : "var(--border)"),
        background: active ? "#0d2044" : "transparent",
        color: active ? "var(--accent)" : "var(--muted)",
        cursor: "pointer",
        fontSize: 12,
      }}
    >
      {label}
    </button>
  );
}

type Tab = "signals" | "stability" | "quality";
type TradeRow = SignalEvent | StabilityEvent | DataQualityEvent;

function colTooltip(tab: Tab, col: string): { tr: string; en: string } | null {
  const map: Record<string, { tr: string; en: string }> = {
    ts_utc: { tr: "UTC zaman damgasi.", en: "UTC event timestamp." },
    ts: { tr: "Zaman damgasi.", en: "Event timestamp." },
    symbol: { tr: "Islem sembolu.", en: "Instrument symbol." },
    regime: { tr: "Aktif piyasa rejimi.", en: "Active market regime." },
    direction: { tr: "Sinyal yonu (buy/sell).", en: "Signal side (buy/sell)." },
    allow_entry: { tr: "Giris izni verildi mi.", en: "Whether entry is allowed." },
    allowed: { tr: "Stability kontrolu gecti mi.", en: "Passed stability gate or not." },
    confidence: { tr: "Model guven puani.", en: "Model confidence score." },
    reason: { tr: "Karar aciklama nedeni.", en: "Decision reason / blocker." },
    signal_type: { tr: "Sinyal kategorisi.", en: "Signal category." },
    streak: { tr: "Ardisik benzer sinyal sayisi.", en: "Consecutive similar signal count." },
    timeframe: { tr: "Veri penceresi.", en: "Data timeframe/window." },
    severity: { tr: "Sorun siddeti.", en: "Issue severity." },
    issues: { tr: "Tespit edilen kalite sorunlari.", en: "Detected quality issues." },
  };
  const key = `${tab}:${col}`;
  if (map[col]) return map[col];
  if (key && map[key]) return map[key];
  return null;
}

export default function Trades() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [tab, setTab] = useState<Tab>("signals");
  const [symbol, setSymbol] = useState("");
  const [query, setQuery] = useState("");

  useEffect(() => {
    const t = searchParams.get("tab");
    const sym = searchParams.get("symbol");
    const q = searchParams.get("q");
    if (t === "signals" || t === "stability" || t === "quality") {
      setTab(t);
    }
    if (sym !== null) {
      setSymbol(sym.toUpperCase());
    }
    if (q !== null) {
      setQuery(q);
    }
  }, [searchParams]);

  const fetchRows = useCallback(
    (signal: AbortSignal) => {
      const sym = symbol.trim() || undefined;
      if (tab === "signals") return api.signals(200, sym, signal);
      if (tab === "stability") return api.stability(200, sym, signal);
      return api.quality(200, sym, signal);
    },
    [symbol, tab]
  );

  const poll = usePoll<TradeRow[]>({
    fetcher: fetchRows,
    pollKey: `api:/events/${tab}`,
    intervalMs: 15_000,
    staleAfterMs: 45_000,
  });

  const rows = useMemo(() => {
    const data = poll.data ?? [];
    const rev = [...data].reverse();
    const q = query.trim().toLowerCase();
    if (!q) return rev;
    return rev.filter((row) => JSON.stringify(row).toLowerCase().includes(q));
  }, [poll.data, query]);

  const mode: DegradedMode = poll.error ? "down" : poll.isStale ? "degraded" : "ok";

  const cols: Record<Tab, string[]> = {
    signals: ["ts_utc", "symbol", "regime", "direction", "allow_entry", "confidence", "reason"],
    stability: ["ts", "symbol", "signal_type", "allowed", "streak", "reason"],
    quality: ["ts", "symbol", "timeframe", "severity", "issues"],
  };

  function cellColor(key: string, val: unknown): string {
    if (key === "allow_entry" || key === "allowed") return val ? "var(--green)" : "var(--red)";
    if (key === "severity") {
      if (val === "critical") return "var(--red)";
      if (val === "warning") return "var(--yellow)";
    }
    return "var(--text)";
  }

  function formatVal(val: unknown): string {
    if (val === null || val === undefined) return "-";
    if (typeof val === "boolean") return val ? "yes" : "no";
    if (Array.isArray(val)) return val.join(", ");
    if (typeof val === "number") return val % 1 !== 0 ? val.toFixed(4) : String(val);
    return String(val);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <PageGuide
        icon="💱"
        titleTr="Islem Analizi"
        titleEn="Trade Analytics"
        subtitleTr="Signals/Stability/Quality sekmeleriyle giris kararini adim adim takip et."
        subtitleEn="Use Signals/Stability/Quality tabs to trace entry decisions end-to-end."
        items={[
          {
            icon: "📈",
            titleTr: "Signals",
            titleEn: "Signals",
            descTr: "Sinyal conf ve allow_entry alanlari burada.",
            descEn: "Signal confidence and allow_entry status live here.",
          },
          {
            icon: "🧱",
            titleTr: "Stability",
            titleEn: "Stability",
            descTr: "Cooldown/streak gibi anti-noise kontrolleri gosterilir.",
            descEn: "Cooldown/streak controls are tracked here.",
          },
          {
            icon: "🧼",
            titleTr: "Quality",
            titleEn: "Quality",
            descTr: "Veri sorunlari ve anomali seviyeleri quality tabinda.",
            descEn: "Data anomalies and severity levels are under quality tab.",
          },
        ]}
      />

      <DegradedBanner mode={mode} message={poll.error?.message} />

      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <TabBtn
          label="Signals"
          active={tab === "signals"}
          onClick={() => {
            setTab("signals");
            const next = new URLSearchParams(searchParams);
            next.set("tab", "signals");
            setSearchParams(next, { replace: true });
          }}
        />
        <TabBtn
          label="Stability"
          active={tab === "stability"}
          onClick={() => {
            setTab("stability");
            const next = new URLSearchParams(searchParams);
            next.set("tab", "stability");
            setSearchParams(next, { replace: true });
          }}
        />
        <TabBtn
          label="Quality"
          active={tab === "quality"}
          onClick={() => {
            setTab("quality");
            const next = new URLSearchParams(searchParams);
            next.set("tab", "quality");
            setSearchParams(next, { replace: true });
          }}
        />
        <input
          placeholder="Filter symbol (e.g. ETHUSDT)"
          value={symbol}
          onChange={(e) => {
            const v = e.target.value.toUpperCase();
            setSymbol(v);
            const next = new URLSearchParams(searchParams);
            next.set("symbol", v);
            setSearchParams(next, { replace: true });
          }}
          style={{
            padding: "4px 10px",
            borderRadius: 4,
            border: "1px solid var(--border)",
            background: "var(--surface)",
            color: "var(--text)",
            fontSize: 12,
            width: 200,
          }}
        />
        <input
          placeholder="Find reason/text"
          value={query}
          onChange={(e) => {
            const v = e.target.value;
            setQuery(v);
            const next = new URLSearchParams(searchParams);
            next.set("q", v);
            setSearchParams(next, { replace: true });
          }}
          style={{
            padding: "4px 10px",
            borderRadius: 4,
            border: "1px solid var(--border)",
            background: "var(--surface)",
            color: "var(--text)",
            fontSize: 12,
            width: 180,
          }}
        />
        {poll.isFetching && <span style={{ color: "var(--muted)", fontSize: 11 }}>Loading...</span>}
        <span style={{ color: "var(--muted)", fontSize: 11, marginLeft: "auto" }}>{rows.length} rows</span>
      </div>

      <div className="card" style={{ overflowX: "auto" }}>
        <AsyncState
          loading={poll.isLoading}
          error={poll.error}
          isEmpty={rows.length === 0}
          loadingText="Loading trades..."
          emptyText="No data"
        >
          <table>
            <thead>
              <tr>
                {cols[tab].map((c) => {
                  const tip = colTooltip(tab, c);
                  return (
                    <th key={c}>
                      {tip ? (
                        <TermTip term={c.replace(/_/g, " ")} tr={tip.tr} en={tip.en} />
                      ) : c.replace(/_/g, " ")}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i}>
                  {cols[tab].map((c) => {
                    const value = (row as Record<string, unknown>)[c];
                    return (
                      <td key={c} style={{ color: cellColor(c, value) }}>
                        {formatVal(value)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </AsyncState>
      </div>
    </div>
  );
}
