import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, streamLog } from "../api/client";
import type { LogFile } from "../api/types";
import AsyncState from "../components/AsyncState";
import DegradedBanner, { type DegradedMode } from "../components/DegradedBanner";
import PageGuide from "../components/PageGuide";
import TermTip from "../components/TermTip";
import { usePoll } from "../hooks/usePoll";
import { useSSE } from "../hooks/useSSE";
import { useSearchParams } from "react-router-dom";
import { useBackendStatus } from "../context/BackendStatusContext";

type Level = "ALL" | "INFO" | "WARNING" | "ERROR" | "CRITICAL";

interface SavedPreset {
  id: string;
  label: string;
  q: string;
  level: Level;
}

interface QueryPack {
  label: string;
  q: string;
  level: Level;
  hint: string;
}

interface LogViewProfile {
  id: string;
  name: string;
  selectedFile: string | null;
  query: string;
  level: Level;
  compareMode: boolean;
  multiFiles: string[];
  multiQuery: string;
  multiLevel: Level;
  forceFallback: boolean;
}

interface IncidentSessionEvent {
  ts: number;
  action: string;
  detail?: string;
}

const PRESET_STORAGE_KEY = "eclipse.logs.presets.v1";
const PROFILE_STORAGE_KEY = "eclipse.logs.view_profiles.v1";
const INCIDENT_SESSION_KEY = "eclipse.logs.incident_session.v1";
const INCIDENT_TIMELINE_KEY = "eclipse.logs.incident_timeline.v1";

export default function Logs() {
  const backend = useBackendStatus();
  const [searchParams, setSearchParams] = useSearchParams();
  const [selected, setSelected] = useState<string | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [tailLoading, setTailLoading] = useState(false);
  const [tailError, setTailError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [level, setLevel] = useState<Level>("ALL");
  const [newPresetName, setNewPresetName] = useState("");
  const [linkCopied, setLinkCopied] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [multiFiles, setMultiFiles] = useState<string[]>([]);
  const [multiQuery, setMultiQuery] = useState("");
  const [multiLevel, setMultiLevel] = useState<Level>("ALL");
  const [multiData, setMultiData] = useState<Record<string, { lines: string[]; error?: string }>>({});
  const [perfMeta, setPerfMeta] = useState<{ listMs?: number; tailMs?: number; cacheHit?: boolean; tailSource?: string }>({});
  const [logApiDegraded, setLogApiDegraded] = useState(false);
  const [degradeScore, setDegradeScore] = useState(0);
  const [forceFallback, setForceFallback] = useState(false);
  const [showDiag, setShowDiag] = useState(false);
  const [timeoutCount, setTimeoutCount] = useState(0);
  const [latencySamples, setLatencySamples] = useState<Array<{ ts: number; listMs?: number; tailMs?: number }>>([]);
  const [savedPresets, setSavedPresets] = useState<SavedPreset[]>(() => {
    try {
      const raw = localStorage.getItem(PRESET_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(
        (p): p is SavedPreset =>
          p &&
          typeof p.id === "string" &&
          typeof p.label === "string" &&
          typeof p.q === "string" &&
          (p.level === "ALL" || p.level === "INFO" || p.level === "WARNING" || p.level === "ERROR" || p.level === "CRITICAL")
      );
    } catch {
      return [];
    }
  });
  const [profiles, setProfiles] = useState<LogViewProfile[]>(() => {
    try {
      const raw = localStorage.getItem(PROFILE_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((p): p is LogViewProfile => p && typeof p.id === "string" && typeof p.name === "string");
    } catch {
      return [];
    }
  });
  const [profileName, setProfileName] = useState("");
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [bundleCopied, setBundleCopied] = useState(false);
  const [incidentSessionId, setIncidentSessionId] = useState<string>(() => {
    try {
      return localStorage.getItem(INCIDENT_SESSION_KEY) || "";
    } catch {
      return "";
    }
  });
  const [incidentTimeline, setIncidentTimeline] = useState<IncidentSessionEvent[]>(() => {
    try {
      const raw = localStorage.getItem(INCIDENT_TIMELINE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((x): x is IncidentSessionEvent => x && typeof x.ts === "number" && typeof x.action === "string");
    } catch {
      return [];
    }
  });
  const bottomRef = useRef<HTMLDivElement>(null);
  const reqSeqRef = useRef(0);

  const fetchLogFiles = useCallback(async (signal: AbortSignal) => {
    const res = await api.logFilesMeta(signal);
    setPerfMeta((p) => ({ ...p, listMs: res.listMs, cacheHit: res.cacheHit }));
    return res.files;
  }, []);
  const fetchTail = useCallback(
    async (signal: AbortSignal) => {
      const res = await api.logTailMeta(selected ?? "", 200, signal);
      setPerfMeta((p) => ({ ...p, tailMs: res.tailMs, tailSource: res.source }));
      return res.tail;
    },
    [selected]
  );

  const filesPoll = usePoll<LogFile[]>({
    fetcher: fetchLogFiles,
    pollKey: "api:/logs",
    intervalMs: (logApiDegraded || forceFallback) ? 60_000 : 30_000,
    staleAfterMs: 90_000,
    enabled: backend.backendUp,
  });

  const tailPoll = usePoll<{ file: string; lines: string[] }>({
    fetcher: fetchTail,
    pollKey: "api:/logs/tail",
    intervalMs: (logApiDegraded || forceFallback) ? 30_000 : 15_000,
    staleAfterMs: 45_000,
    enabled: backend.backendUp && Boolean(selected) && !streaming,
  });

  useEffect(() => {
    const timeoutErr =
      (tailPoll.error?.message || "").toLowerCase().includes("request timeout") ||
      (filesPoll.error?.message || "").toLowerCase().includes("request timeout");
    const latencyBad = (perfMeta.tailMs ?? 0) > 2500 || (perfMeta.listMs ?? 0) > 2500;
    const isBad = timeoutErr || latencyBad;
    if (timeoutErr) setTimeoutCount((c) => c + 1);
    setDegradeScore((prev) => {
      const next = Math.max(0, Math.min(8, prev + (isBad ? 2 : -1)));
      if (!forceFallback) {
        if (next >= 3) setLogApiDegraded(true);
        if (next === 0) setLogApiDegraded(false);
      }
      return next;
    });
  }, [filesPoll.error?.message, tailPoll.error?.message, perfMeta.listMs, perfMeta.tailMs, forceFallback]);

  useEffect(() => {
    setLatencySamples((prev) => {
      const next = [...prev, { ts: Date.now(), listMs: perfMeta.listMs, tailMs: perfMeta.tailMs }];
      return next.slice(-12);
    });
  }, [perfMeta.listMs, perfMeta.tailMs]);

  useEffect(() => {
    if (forceFallback) setLogApiDegraded(true);
  }, [forceFallback]);

  useEffect(() => {
    if (tailPoll.data && !streaming) {
      setLines(tailPoll.data.lines ?? []);
      setTailError(null);
    }
  }, [streaming, tailPoll.data]);

  const sse = useSSE({
    enabled: backend.backendUp && Boolean(selected) && streaming,
    connect: () => streamLog(selected ?? "", 50),
    onMessage: (data) => {
      if (data && !data.startsWith(": keepalive")) {
        setLines((prev) => [...prev.slice(-999), data]);
      }
    },
    reconnectInitialMs: 1000,
    reconnectMaxMs: 10000,
    staleAfterMs: 10000,
  });

  useEffect(() => {
    const el = bottomRef.current;
    if (el && typeof el.scrollIntoView === "function") {
      el.scrollIntoView({ behavior: "smooth" });
    }
  }, [lines]);

  const mode: DegradedMode = useMemo(() => {
    if (logApiDegraded || forceFallback) return "degraded";
    if (filesPoll.error || (streaming && sse.error)) return "down";
    if (filesPoll.isStale || (streaming && (sse.status === "reconnecting" || sse.isStale))) return "degraded";
    return "ok";
  }, [filesPoll.error, filesPoll.isStale, logApiDegraded, forceFallback, sse.error, sse.isStale, sse.status, streaming]);

  async function openFile(name: string) {
    if (!backend.backendUp) {
      setSelected(name);
      setTailError("Backend unavailable. Wait for API UP, then reload.");
      return;
    }
    const reqId = ++reqSeqRef.current;
    setStreaming(false);
    setSelected(name);
    setLines([]);
    const next = new URLSearchParams(searchParams);
    next.set("file", name);
    setSearchParams(next, { replace: true });
    setTailError(null);
    setTailLoading(true);
    try {
      const res = await api.logTailMeta(name, 200);
      if (reqSeqRef.current === reqId) {
        setLines(res.tail.lines ?? []);
        setPerfMeta((p) => ({ ...p, tailMs: res.tailMs, tailSource: res.source }));
      }
    } catch (err) {
      if (reqSeqRef.current === reqId) {
        setTailError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (reqSeqRef.current === reqId) {
        setTailLoading(false);
      }
    }
  }

  function startStream(name: string) {
    setSelected(name);
    setLines([]);
    setStreaming(true);
  }

  function stopStream() {
    setStreaming(false);
  }

  function fileHint(name: string): string {
    const n = name.toLowerCase();
    if (n.includes("microstructure_collector")) return "Collector bağlantı ve akış logu.";
    if (n.includes("alpha_gate")) return "Sinyal gate kararları (allow/block) JSONL.";
    if (n.includes("signal_stability")) return "Stability/cooldown kararları.";
    if (n.includes("data_quality")) return "Veri kalite uyarıları ve anomaliler.";
    if (n.includes("regime")) return "Rejim geçişleri ve güven metrikleri.";
    if (n.includes("risk")) return "Risk motoru skip/kill/cap olayları.";
    if (n.includes("paper")) return "Paper-trade çalışma ve özet logları.";
    return "Genel runtime log dosyası.";
  }

  function fmtSize(bytes: number): string {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}K`;
    return `${(bytes / 1024 / 1024).toFixed(1)}M`;
  }

  function fmtMtime(ts: number): string {
    try {
      return new Date(ts * 1000).toLocaleString();
    } catch {
      return "-";
    }
  }

  const files = filesPoll.data ?? [];

  function toggleMultiFile(name: string) {
    setMultiFiles((prev) => {
      if (prev.includes(name)) return prev.filter((x) => x !== name);
      if (prev.length >= 2) return prev;
      return [...prev, name];
    });
  }

  const packs: QueryPack[] = [
    { label: "No Match", q: "no_match_detail", level: "INFO", hint: "Pocket hangi esikten fail oldu?" },
    { label: "Regime", q: "REGIME", level: "INFO", hint: "Rejim gecisi ve gate bloklarini yakala." },
    { label: "Shutdown", q: "SHUTDOWN", level: "CRITICAL", hint: "Offline/shutdown zincirini izle." },
    { label: "Timeout", q: "RequestTimeout", level: "WARNING", hint: "Ag/API timeout kaynaklarini bul." },
  ];

  useEffect(() => {
    const file = searchParams.get("file");
    const q = searchParams.get("q");
    const lvl = searchParams.get("level");
    const pack = searchParams.get("pack");
    if (q !== null) {
      setQuery(q);
    }
    if (lvl === "ALL" || lvl === "INFO" || lvl === "WARNING" || lvl === "ERROR" || lvl === "CRITICAL") {
      setLevel(lvl);
    }
    if (pack && q === null && lvl === null) {
      const hit = packs.find((p) => p.label.toLowerCase() === pack.toLowerCase());
      if (hit) {
        setQuery(hit.q);
        setLevel(hit.level);
      }
    }
    if (file && file !== selected) {
      void openFile(file);
    }
    // only on mount/route-query change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);
  const selectedFileMeta = useMemo(
    () => files.find((f) => f.name === selected) ?? null,
    [files, selected]
  );
  const filteredLines = useMemo(() => {
    const q = query.trim().toLowerCase();
    return lines.filter((line) => {
      if (level !== "ALL" && !line.includes(`| ${level}`)) {
        return false;
      }
      if (!q) return true;
      return line.toLowerCase().includes(q);
    });
  }, [level, lines, query]);

  useEffect(() => {
    if (!backend.backendUp || !compareMode || !selected) return undefined;
    const panelFiles = [selected, ...multiFiles].filter(Boolean);
    let active = true;

    async function loadMulti() {
      const next: Record<string, { lines: string[]; error?: string }> = {};
      await Promise.all(
        panelFiles.map(async (f) => {
          try {
            const res = await api.logTail(f, 120);
            next[f] = { lines: res.lines ?? [] };
          } catch (err) {
            next[f] = { lines: [], error: err instanceof Error ? err.message : String(err) };
          }
        })
      );
      if (active) setMultiData(next);
    }

    void loadMulti();
    if (logApiDegraded || forceFallback) {
      return () => {
        active = false;
      };
    }
    const id = window.setInterval(loadMulti, 5000);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [backend.backendUp, compareMode, selected, multiFiles, logApiDegraded, forceFallback]);

  function applyLineFilter(source: string[]): string[] {
    const q = multiQuery.trim().toLowerCase();
    return source.filter((line) => {
      if (multiLevel !== "ALL" && !line.includes(`| ${multiLevel}`)) return false;
      if (!q) return true;
      return line.toLowerCase().includes(q);
    });
  }

  function applyPreset(preset: { label: string; q: string; level: Level }) {
    setQuery(preset.q);
    setLevel(preset.level);
    const next = new URLSearchParams(searchParams);
    next.set("q", preset.q);
    next.set("level", preset.level);
    next.set("pack", preset.label);
    setSearchParams(next, { replace: true });
    appendIncidentEvent("pack_applied", preset.label);
  }

  async function copyCurrentLink() {
    const params = new URLSearchParams(searchParams);
    if (selected) params.set("file", selected);
    if (query) params.set("q", query);
    if (level) params.set("level", level);
    const url = `${window.location.origin}${window.location.pathname}?${params.toString()}`;
    try {
      await navigator.clipboard.writeText(url);
      setLinkCopied(true);
      window.setTimeout(() => setLinkCopied(false), 1200);
    } catch {
      setLinkCopied(false);
    }
  }

  function persistPresets(next: SavedPreset[]) {
    setSavedPresets(next);
    try {
      localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // best effort only
    }
  }

  function persistProfiles(next: LogViewProfile[]) {
    setProfiles(next);
    try {
      localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // best effort only
    }
  }

  function persistIncidentTimeline(next: IncidentSessionEvent[]) {
    setIncidentTimeline(next);
    try {
      localStorage.setItem(INCIDENT_TIMELINE_KEY, JSON.stringify(next));
    } catch {
      // best effort only
    }
  }

  function appendIncidentEvent(action: string, detail?: string) {
    if (!incidentSessionId) return;
    const evt: IncidentSessionEvent = { ts: Date.now(), action, detail };
    persistIncidentTimeline([...incidentTimeline, evt].slice(-80));
  }

  function startIncidentSession() {
    const d = new Date();
    const pad = (n: number) => String(n).padStart(2, "0");
    const id = `INC-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
    setIncidentSessionId(id);
    try {
      localStorage.setItem(INCIDENT_SESSION_KEY, id);
    } catch {
      // best effort
    }
    persistIncidentTimeline([{ ts: Date.now(), action: "session_started", detail: id }]);
  }

  function stopIncidentSession() {
    appendIncidentEvent("session_stopped", incidentSessionId);
    setIncidentSessionId("");
    try {
      localStorage.removeItem(INCIDENT_SESSION_KEY);
    } catch {
      // best effort
    }
  }

  function saveCurrentPreset() {
    const label = newPresetName.trim();
    if (!label) return;
    const next: SavedPreset[] = [
      ...savedPresets,
      {
        id: `${Date.now()}_${Math.random().toString(16).slice(2, 8)}`,
        label,
        q: query,
        level,
      },
    ];
    persistPresets(next);
    setNewPresetName("");
  }

  function deletePreset(id: string) {
    persistPresets(savedPresets.filter((p) => p.id !== id));
  }

  function saveCurrentProfile() {
    const name = profileName.trim();
    if (!name) return;
    const item: LogViewProfile = {
      id: `${Date.now()}_${Math.random().toString(16).slice(2, 8)}`,
      name,
      selectedFile: selected,
      query,
      level,
      compareMode,
      multiFiles,
      multiQuery,
      multiLevel,
      forceFallback,
    };
    persistProfiles([...profiles, item]);
    setProfileName("");
    setSelectedProfileId(item.id);
    appendIncidentEvent("profile_saved", name);
  }

  async function applyProfile(id: string) {
    const p = profiles.find((x) => x.id === id);
    if (!p) return;
    setSelectedProfileId(id);
    setQuery(p.query);
    setLevel(p.level);
    setCompareMode(p.compareMode);
    setMultiFiles(p.multiFiles ?? []);
    setMultiQuery(p.multiQuery ?? "");
    setMultiLevel(p.multiLevel ?? "ALL");
    setForceFallback(Boolean(p.forceFallback));
    const next = new URLSearchParams(searchParams);
    if (p.selectedFile) next.set("file", p.selectedFile);
    else next.delete("file");
    if (p.query) next.set("q", p.query); else next.delete("q");
    if (p.level) next.set("level", p.level); else next.delete("level");
    setSearchParams(next, { replace: true });
    if (p.selectedFile) {
      await openFile(p.selectedFile);
    }
    appendIncidentEvent("profile_loaded", p.name);
  }

  function deleteProfile(id: string) {
    persistProfiles(profiles.filter((p) => p.id !== id));
    if (selectedProfileId === id) setSelectedProfileId("");
    appendIncidentEvent("profile_deleted", id);
  }

  function forceRecoverNow() {
    setDegradeScore(0);
    setLogApiDegraded(false);
    setTimeoutCount(0);
  }

  function downloadJson(filename: string, payload: unknown) {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  function exportIncidentBundle() {
    const nowIso = new Date().toISOString();
    const selectedProfile = profiles.find((p) => p.id === selectedProfileId) ?? null;
    const panels = compareMode ? [selected, ...multiFiles].filter((x): x is string => Boolean(x)) : [selected].filter((x): x is string => Boolean(x));
    const panelLines: Record<string, string[]> = {};
    for (const p of panels) {
      const raw = p === selected ? lines : (multiData[p]?.lines ?? []);
      panelLines[p] = applyLineFilter(raw).slice(-300);
    }
    const payload = {
      exported_at: nowIso,
      incident_session_id: incidentSessionId || null,
      route: `${window.location.pathname}${window.location.search}`,
      selected_file: selected,
      filters: {
        query,
        level,
        compare_mode: compareMode,
        multi_files: multiFiles,
        multi_query: multiQuery,
        multi_level: multiLevel,
      },
      profile: selectedProfile,
      diagnostics: {
        perf: perfMeta,
        timeout_count: timeoutCount,
        degrade_score: degradeScore,
        fallback_forced: forceFallback,
        log_api_degraded: logApiDegraded,
        latency_samples: latencySamples,
        incident_timeline: incidentTimeline,
      },
      excerpts: panelLines,
    };
    const safeTs = nowIso.replace(/[:.]/g, "-");
    downloadJson(`logs_incident_bundle_${safeTs}.json`, payload);
    appendIncidentEvent("bundle_exported", safeTs);
  }

  async function copyTriageSummary() {
    const selectedProfile = profiles.find((p) => p.id === selectedProfileId);
    const msg = [
      `Logs Triage Summary`,
      `incident=${incidentSessionId || "-"}`,
      `file=${selected ?? "-"}`,
      `q=${query || "-"}`,
      `level=${level}`,
      `compare=${compareMode ? "on" : "off"} panels=${([selected, ...multiFiles].filter(Boolean)).length}`,
      `list_ms=${perfMeta.listMs?.toFixed?.(1) ?? "-"} tail_ms=${perfMeta.tailMs?.toFixed?.(1) ?? "-"} cache=${perfMeta.cacheHit ? "HIT" : "MISS"} src=${perfMeta.tailSource ?? "-"}`,
      `degraded=${logApiDegraded ? "1" : "0"} forced=${forceFallback ? "1" : "0"} score=${degradeScore} timeout_count=${timeoutCount}`,
      `profile=${selectedProfile?.name ?? "-"}`,
      `url=${window.location.origin}${window.location.pathname}${window.location.search}`,
    ].join(" | ");
    try {
      await navigator.clipboard.writeText(msg);
      setBundleCopied(true);
      window.setTimeout(() => setBundleCopied(false), 1200);
      appendIncidentEvent("summary_copied");
    } catch {
      setBundleCopied(false);
    }
  }

  useEffect(() => {
    const op = searchParams.get("ops");
    if (!op) return;
    if (op === "start_incident") startIncidentSession();
    else if (op === "stop_incident") stopIncidentSession();
    else if (op === "export_bundle") exportIncidentBundle();
    else if (op === "copy_summary") void copyTriageSummary();
    else if (op === "force_recover") forceRecoverNow();
    else if (op === "keep_fallback_on") setForceFallback(true);
    else if (op === "keep_fallback_off") setForceFallback(false);
    const next = new URLSearchParams(searchParams);
    next.delete("ops");
    setSearchParams(next, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <PageGuide
        icon="📜"
        titleTr="Log İnceleme"
        titleEn="Log Explorer"
        subtitleTr="Dosya seç, filtre uygula, sonra canlı akışa geç. Sorun kök nedenini bu akışla bul."
        subtitleEn="Pick file, apply filters, then switch to live stream for root-cause."
        items={[
          {
            icon: "1️⃣",
            titleTr: "Dosya Seç",
            titleEn: "Pick File",
            descTr: "Önce soldan ilgili log dosyasını seç (paper, regime, risk vb.).",
            descEn: "Choose the relevant file first (paper, regime, risk, etc.).",
          },
          {
            icon: "2️⃣",
            titleTr: "Filtrele",
            titleEn: "Filter",
            descTr: "Level + arama ile gereksiz satırları temizle.",
            descEn: "Use level + text query to cut noise quickly.",
          },
          {
            icon: "3️⃣",
            titleTr: "Canlı İzle",
            titleEn: "Live Tail",
            descTr: "Live ile anlık değişimi izle, Stop ile dondurup incele.",
            descEn: "Use Live for real-time, Stop to freeze and inspect.",
          },
        ]}
      />

      <DegradedBanner
        mode={mode}
        message={
          (logApiDegraded || forceFallback)
            ? `LOG API DEGRADED (fallback mode) score=${degradeScore}${forceFallback ? " [forced]" : ""}`
            : filesPoll.error?.message ?? sse.error?.message ?? (streaming ? `SSE: ${sse.status}` : undefined)
        }
      />
      {!backend.backendUp && (
        <div className="card" style={{ borderStyle: "dashed", marginBottom: 8 }}>
          <div className="card-title">Backend Connection</div>
          <div style={{ color: "var(--yellow)", fontSize: 12 }}>
            Backend unreachable ({backend.backendMessage}). Polling paused, retry in {Math.max(1, Math.ceil(backend.nextRetryInMs / 1000))}s.
          </div>
        </div>
      )}

      <div className="card" style={{ padding: 8 }}>
        <div className="card-title self-help" data-help="Incident kayit oturumu: triage adimlarini zaman cizelgesiyle kaydeder.">Incident Session Registry</div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <span className={`badge ${incidentSessionId ? "badge-yellow" : "badge-gray"}`}>
            {incidentSessionId || "no active incident"}
          </span>
          {!incidentSessionId ? (
            <button
              className="self-help"
              data-help="Yeni incident oturumu baslatir."
              onClick={startIncidentSession}
              style={{
                padding: "3px 8px",
                borderRadius: 4,
                border: "1px solid var(--border)",
                background: "transparent",
                color: "var(--muted)",
                cursor: "pointer",
              }}
            >
              Start Session
            </button>
          ) : (
            <button
              className="self-help"
              data-help="Aktif incident oturumunu kapatir."
              onClick={stopIncidentSession}
              style={{
                padding: "3px 8px",
                borderRadius: 4,
                border: "1px solid var(--border)",
                background: "transparent",
                color: "var(--muted)",
                cursor: "pointer",
              }}
            >
              Stop Session
            </button>
          )}
          <button
            className="self-help"
            data-help="Timeline listesini sifirlar."
            onClick={() => persistIncidentTimeline([])}
            style={{
              padding: "3px 8px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--muted)",
              cursor: "pointer",
            }}
          >
            Clear Timeline
          </button>
          <span style={{ color: "var(--muted)", fontSize: 11, marginLeft: "auto" }}>
            events={incidentTimeline.length}
          </span>
        </div>
        {incidentTimeline.length > 0 && (
          <pre
            style={{
              marginTop: 6,
              maxHeight: 90,
              overflowY: "auto",
              fontSize: 10,
              padding: 6,
              background: "var(--bg)",
              borderRadius: 4,
              border: "1px solid var(--border)",
            }}
          >
            {incidentTimeline
              .slice(-10)
              .map((e) => `${new Date(e.ts).toLocaleTimeString()} ${e.action}${e.detail ? ` :: ${e.detail}` : ""}`)
              .join("\n")}
          </pre>
        )}
      </div>

      <div style={{ display: "flex", gap: 16, height: "calc(100vh - 140px)" }}>
        <div className="card" style={{ width: 240, overflowY: "auto", flexShrink: 0 }}>
          <div className="card-title self-help" data-help="Analiz etmek istedigin log dosyasini soldan sec.">Log Files</div>
          <AsyncState
            loading={filesPoll.isLoading}
            error={filesPoll.error}
            isEmpty={files.length === 0}
            emptyText="No files found"
          >
            {files.map((f) => (
              <div
                key={f.name}
                style={{
                  padding: "6px 8px",
                  borderRadius: 4,
                  cursor: "pointer",
                  background: selected === f.name ? "#21262d" : "transparent",
                  marginBottom: 2,
                }}
                onClick={() => openFile(f.name)}
              >
                <div style={{ fontWeight: selected === f.name ? 700 : 400 }}>{f.name}</div>
                <div style={{ color: "var(--muted)", fontSize: 11 }}>{fmtSize(f.size_bytes)}</div>
              </div>
            ))}
          </AsyncState>
        </div>

        <div className="card" style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div className="card-title" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>{selected ?? "Select a file"}</span>
            {selected && (
              <div style={{ display: "flex", gap: 8 }}>
                {streaming ? (
                  <button
                    className="self-help"
                    data-help="Canli akis modunu durdurur ve ekrani dondurur."
                    onClick={stopStream}
                    style={{
                      padding: "2px 10px",
                      borderRadius: 4,
                      border: "1px solid var(--red)",
                      background: "transparent",
                      color: "var(--red)",
                      cursor: "pointer",
                    }}
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    className="self-help"
                    data-help="Canli log akisina gecer (tail)."
                    onClick={() => startStream(selected)}
                    style={{
                      padding: "2px 10px",
                      borderRadius: 4,
                      border: "1px solid var(--accent)",
                      background: "transparent",
                      color: "var(--accent)",
                      cursor: "pointer",
                    }}
                  >
                    Live
                  </button>
                )}
                <button
                className="self-help"
                data-help="Tek seferlik yenileme: son satirlari tekrar ceker."
                onClick={() => {
                  if (streaming) {
                    stopStream();
                  } else {
                    tailPoll.refresh();
                    }
                  }}
                  style={{
                    padding: "2px 10px",
                    borderRadius: 4,
                    border: "1px solid var(--border)",
                    background: "transparent",
                    color: "var(--muted)",
                    cursor: "pointer",
                  }}
                >
                  Reload
                </button>
              </div>
            )}
          </div>

          {selected && selectedFileMeta && (
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 12,
                fontSize: 11,
                color: "var(--muted)",
                marginBottom: 8,
                borderBottom: "1px solid var(--border)",
                paddingBottom: 8,
              }}
            >
              <span>Type: {fileHint(selectedFileMeta.name)}</span>
              <span>Size: {fmtSize(selectedFileMeta.size_bytes)}</span>
              <span>Updated: {fmtMtime(selectedFileMeta.mtime)}</span>
            </div>
          )}

          {selected && (
            <div
              style={{
                display: "flex",
                gap: 8,
                flexWrap: "wrap",
                marginBottom: 8,
                paddingBottom: 8,
                borderBottom: "1px dashed var(--border)",
              }}
            >
              <span style={{ fontSize: 11, color: "var(--muted)", alignSelf: "center" }}>
                Quick Packs:
              </span>
              {packs.map((p) => (
                <button
                  key={`quick_${p.label}`}
                  onClick={() => applyPreset(p)}
                  title={`${p.hint} | q=${p.q} | level=${p.level}`}
                  style={{
                    padding: "3px 8px",
                    borderRadius: 999,
                    border: "1px solid var(--border)",
                    background: query === p.q && level === p.level ? "var(--surface-2)" : "transparent",
                    color: query === p.q && level === p.level ? "var(--text)" : "var(--muted)",
                    cursor: "pointer",
                    fontSize: 11,
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          )}

          {selected && (
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
              <span style={{ fontSize: 11, color: "var(--muted)", alignSelf: "center" }}>
                <TermTip
                  term="Level"
                  tr="Log seviyesi: INFO, WARNING, ERROR, CRITICAL."
                  en="Log severity level: INFO, WARNING, ERROR, CRITICAL."
                />
              </span>
              <input
                className="self-help"
                data-help="Satirlarda serbest metin aramasi yapar."
                value={query}
                onChange={(e) => {
                  const v = e.target.value;
                  setQuery(v);
                  const next = new URLSearchParams(searchParams);
                  next.set("q", v);
                  setSearchParams(next, { replace: true });
                }}
                placeholder="Search in lines (symbol, reason, error...)"
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "var(--bg)",
                  color: "var(--text)",
                  minWidth: 280,
                }}
              />
              <select
                className="self-help"
                data-help="Log seviyesi filtresi (INFO/WARNING/ERROR/CRITICAL)."
                value={level}
                onChange={(e) => {
                  const v = e.target.value as Level;
                  setLevel(v);
                  const next = new URLSearchParams(searchParams);
                  next.set("level", v);
                  setSearchParams(next, { replace: true });
                }}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "var(--bg)",
                  color: "var(--text)",
                }}
              >
                <option value="ALL">All levels</option>
                <option value="INFO">INFO</option>
                <option value="WARNING">WARNING</option>
                <option value="ERROR">ERROR</option>
                <option value="CRITICAL">CRITICAL</option>
              </select>
              <button
                className="self-help"
                data-help="Iki veya daha fazla log dosyasini yanyana karsilastirma modunu acar."
                onClick={() => setCompareMode((v) => !v)}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: compareMode ? "var(--surface-2)" : "transparent",
                  color: compareMode ? "var(--text)" : "var(--muted)",
                  cursor: "pointer",
                }}
                title="Open multi-stream compare mode"
              >
                {compareMode ? "Compare ON" : "Compare OFF"}
              </button>
              {compareMode && (logApiDegraded || forceFallback) && (
                <span style={{ fontSize: 11, color: "var(--yellow)", alignSelf: "center" }}>
                  compare auto-refresh paused (degraded mode)
                </span>
              )}
              <button
                className="self-help"
                data-help="Degrade skorunu temizleyip normal polling moduna dondurur."
                onClick={forceRecoverNow}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: "var(--muted)",
                  cursor: "pointer",
                }}
                title="Reset degrade score and return to normal polling"
              >
                Force Recover
              </button>
              <button
                className="self-help"
                data-help="Fallback modunu manuel ac/kapat."
                onClick={() => setForceFallback((v) => !v)}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: forceFallback ? "var(--surface-2)" : "transparent",
                  color: forceFallback ? "var(--text)" : "var(--muted)",
                  cursor: "pointer",
                }}
                title="Lock fallback mode on/off manually"
              >
                {forceFallback ? "Keep Fallback ON" : "Keep Fallback OFF"}
              </button>
              <button
                className="self-help"
                data-help="Log API gecikme/tail istatistiklerini gosterir."
                onClick={() => setShowDiag((v) => !v)}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: showDiag ? "var(--surface-2)" : "transparent",
                  color: showDiag ? "var(--text)" : "var(--muted)",
                  cursor: "pointer",
                }}
              >
                {showDiag ? "Diagnostics Hide" : "Diagnostics"}
              </button>
              <select
                value={selectedProfileId}
                onChange={(e) => {
                  const id = e.target.value;
                  if (!id) {
                    setSelectedProfileId("");
                    return;
                  }
                  void applyProfile(id);
                }}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "var(--bg)",
                  color: "var(--text)",
                }}
                title="Load saved log view profile"
              >
                <option value="">Profiles...</option>
                {profiles.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
              <input
                value={profileName}
                onChange={(e) => setProfileName(e.target.value)}
                placeholder="Profile name"
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "var(--bg)",
                  color: "var(--text)",
                  width: 130,
                }}
              />
              <button
                className="self-help"
                data-help="Mevcut filtre ayarlarini profile kaydeder."
                onClick={saveCurrentProfile}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: "var(--muted)",
                  cursor: "pointer",
                }}
                title="Save current log view controls as profile"
              >
                Save Profile
              </button>
              <button
                className="self-help"
                data-help="Secili profile kaydini siler."
                onClick={() => selectedProfileId && deleteProfile(selectedProfileId)}
                disabled={!selectedProfileId}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: selectedProfileId ? "var(--muted)" : "var(--border)",
                  cursor: selectedProfileId ? "pointer" : "not-allowed",
                }}
                title="Delete selected profile"
              >
                Delete Profile
              </button>
              <button
                className="self-help"
                data-help="Current triage context'i JSON bundle olarak export eder."
                onClick={exportIncidentBundle}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: "var(--muted)",
                  cursor: "pointer",
                }}
                title="Export current log triage context as JSON bundle"
              >
                Export Bundle (JSON)
              </button>
              <button
                className="self-help"
                data-help="Kisa triage ozetini panoya kopyalar."
                onClick={() => void copyTriageSummary()}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: bundleCopied ? "var(--green)" : "var(--muted)",
                  cursor: "pointer",
                }}
                title="Copy compact triage summary text"
              >
                {bundleCopied ? "Summary Copied" : "Copy Triage Summary"}
              </button>
              <button
                className="self-help"
                data-help="Query/level filtrelerini temizler."
                onClick={() => {
                  setQuery("");
                  setLevel("ALL");
                  const next = new URLSearchParams(searchParams);
                  next.delete("q");
                  next.delete("level");
                  setSearchParams(next, { replace: true });
                }}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: "var(--muted)",
                  cursor: "pointer",
                }}
              >
                Clear
              </button>
              {packs.map((p) => (
                <button
                  key={p.label}
                  onClick={() => applyPreset(p)}
                  title={p.hint}
                  style={{
                    padding: "4px 8px",
                    borderRadius: 4,
                    border: "1px solid var(--border)",
                    background: "transparent",
                    color: "var(--muted)",
                    cursor: "pointer",
                  }}
                >
                  {p.label}
                </button>
              ))}
              <button
                className="self-help"
                data-help="Bu gorunume dogrudan donebilmek icin linki kopyalar."
                onClick={() => void copyCurrentLink()}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: linkCopied ? "var(--green)" : "var(--muted)",
                  cursor: "pointer",
                }}
                title="Copy current log view link"
              >
                {linkCopied ? "Link Copied" : "Copy Link"}
              </button>
              {savedPresets.map((p) => (
                <div key={p.id} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  <button
                    onClick={() => applyPreset(p)}
                    style={{
                      padding: "4px 8px",
                      borderRadius: 4,
                      border: "1px solid var(--accent)",
                      background: "transparent",
                      color: "var(--accent)",
                      cursor: "pointer",
                    }}
                    title={`q=${p.q || "(empty)"} | level=${p.level}`}
                  >
                    {p.label}
                  </button>
                  <button
                    onClick={() => deletePreset(p.id)}
                    style={{
                      padding: "2px 6px",
                      borderRadius: 4,
                      border: "1px solid var(--border)",
                      background: "transparent",
                      color: "var(--muted)",
                      cursor: "pointer",
                      fontSize: 11,
                    }}
                    title="Delete preset"
                  >
                    x
                  </button>
                </div>
              ))}
              <input
                value={newPresetName}
                onChange={(e) => setNewPresetName(e.target.value)}
                placeholder="Preset name"
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "var(--bg)",
                  color: "var(--text)",
                  width: 120,
                }}
              />
              <button
                className="self-help"
                data-help="Query+level ayarini yeni preset olarak kaydeder."
                onClick={saveCurrentPreset}
                style={{
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: "var(--muted)",
                  cursor: "pointer",
                }}
                title="Save current query+level as preset"
              >
                Save Preset
              </button>
              <span style={{ fontSize: 11, color: "var(--muted)", marginLeft: "auto" }}>
                list={perfMeta.listMs != null ? `${perfMeta.listMs.toFixed(1)}ms` : "-"} | tail={perfMeta.tailMs != null ? `${perfMeta.tailMs.toFixed(1)}ms` : "-"} | cache={perfMeta.cacheHit ? "HIT" : "MISS"} | src={perfMeta.tailSource ?? "-"} | 
              </span>
              <span style={{ fontSize: 11, color: "var(--muted)" }}>
                showing {filteredLines.length}/{lines.length} lines
              </span>
            </div>
          )}

          {selected && showDiag && (
            <div
              className="card"
              style={{ marginBottom: 8, padding: 8, borderStyle: "dashed" }}
            >
              <div className="card-title">Log Diagnostics</div>
              <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6 }}>
                timeout_count={timeoutCount} | degrade_score={degradeScore} | fallback_forced={forceFallback ? "1" : "0"}
              </div>
              <pre
                style={{
                  maxHeight: 120,
                  overflowY: "auto",
                  fontSize: 10,
                  padding: 6,
                  background: "var(--bg)",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                }}
              >
                {latencySamples
                  .map((s) => `${new Date(s.ts).toLocaleTimeString()} list=${s.listMs?.toFixed?.(1) ?? "-"}ms tail=${s.tailMs?.toFixed?.(1) ?? "-"}ms`)
                  .join("\n")}
              </pre>
            </div>
          )}

          {selected && compareMode && (
            <div
              style={{
                border: "1px dashed var(--border)",
                borderRadius: 6,
                padding: 8,
                marginBottom: 8,
                display: "flex",
                flexDirection: "column",
                gap: 8,
              }}
            >
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>Compare panels (max 2 extra):</span>
                {files
                  .filter((f) => f.name !== selected)
                  .slice(0, 12)
                  .map((f) => (
                    <button
                      key={`mf_${f.name}`}
                      onClick={() => toggleMultiFile(f.name)}
                      style={{
                        padding: "2px 7px",
                        borderRadius: 999,
                        border: "1px solid var(--border)",
                        background: multiFiles.includes(f.name) ? "var(--surface-2)" : "transparent",
                        color: multiFiles.includes(f.name) ? "var(--text)" : "var(--muted)",
                        cursor: "pointer",
                        fontSize: 10,
                      }}
                    >
                      {f.name}
                    </button>
                  ))}
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                <input
                  value={multiQuery}
                  onChange={(e) => setMultiQuery(e.target.value)}
                  placeholder="Compare filter"
                  style={{
                    padding: "3px 8px",
                    borderRadius: 4,
                    border: "1px solid var(--border)",
                    background: "var(--bg)",
                    color: "var(--text)",
                    fontSize: 11,
                    width: 180,
                  }}
                />
                <select
                  value={multiLevel}
                  onChange={(e) => setMultiLevel(e.target.value as Level)}
                  style={{
                    padding: "3px 8px",
                    borderRadius: 4,
                    border: "1px solid var(--border)",
                    background: "var(--bg)",
                    color: "var(--text)",
                    fontSize: 11,
                  }}
                >
                  <option value="ALL">ALL</option>
                  <option value="INFO">INFO</option>
                  <option value="WARNING">WARNING</option>
                  <option value="ERROR">ERROR</option>
                  <option value="CRITICAL">CRITICAL</option>
                </select>
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                  gap: 8,
                }}
              >
                {[selected, ...multiFiles].map((fname) => {
                  const raw = multiData[fname]?.lines ?? [];
                  const err = multiData[fname]?.error;
                  const filtered = applyLineFilter(raw);
                  return (
                    <div key={`panel_${fname}`} className="card" style={{ padding: 8 }}>
                      <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 4 }}>{fname}</div>
                      {err ? (
                        <div style={{ color: "var(--red)", fontSize: 11 }}>{err}</div>
                      ) : (
                        <pre
                          style={{
                            maxHeight: 220,
                            overflowY: "auto",
                            fontSize: 10,
                            padding: 6,
                            background: "var(--bg)",
                            borderRadius: 4,
                            border: "1px solid var(--border)",
                            whiteSpace: "pre-wrap",
                            wordBreak: "break-all",
                          }}
                        >
                          {filtered.length ? filtered.join("\n") : "No lines"}
                        </pre>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <pre
            style={{
              flex: 1,
              overflowY: "auto",
              fontSize: 11,
              color: "var(--text)",
              padding: 8,
              background: "var(--bg)",
              borderRadius: 4,
              whiteSpace: "pre-wrap",
              wordBreak: "break-all",
            }}
          >
            {!selected ? (
              <span style={{ color: "var(--muted)" }}>Select a file on the left</span>
            ) : (
              <AsyncState
                loading={Boolean(selected) && !streaming && tailLoading && lines.length === 0}
                error={!streaming ? (tailError ? new Error(tailError) : tailPoll.error) : null}
                isEmpty={filteredLines.length === 0}
                loadingText="Loading..."
                emptyText={lines.length === 0 ? "No lines" : "No lines match filter"}
              >
                {filteredLines.join("\n")}
              </AsyncState>
            )}
            <div ref={bottomRef} />
          </pre>

          {streaming && (
            <div style={{ padding: "4px 8px", fontSize: 11, color: sse.status === "open" ? "var(--green)" : "var(--yellow)" }}>
              * LIVE ({sse.status})
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-title">Quick Tips</div>
        <div style={{ color: "var(--muted)", fontSize: 12, display: "flex", flexDirection: "column", gap: 4 }}>
          <span>`INFO` normal akış, `WARNING` dikkat, `ERROR` hata, `CRITICAL` acil durum.</span>
          <span>`reason=` alanı kararın nedenini söyler (ör: `no_match`, `regime_mismatch`, `timeout`).</span>
          <span>`conf=` confidence değeridir; mikro path binary ise tipik olarak `0.00` veya `1.00` olur.</span>
          <span>`gates=...` hangi gate'in geçtiğini/engellediğini gösterir.</span>
        </div>
      </div>
    </div>
  );
}
