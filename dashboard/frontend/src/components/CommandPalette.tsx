import React, { useEffect, useRef, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";

interface CmdItem {
  id: string;
  label: string;
  desc?: string;
  icon: string;
  tag: string;
  action: () => void;
}

const NAV_ITEMS: Omit<CmdItem, "action">[] = [
  { id: "nav-status",   label: "Status Overview",    desc: "System health, scoreboard, gates",    icon: "◉", tag: "page" },
  { id: "nav-live",     label: "Live Monitor",        desc: "Real-time trades, risk cards",        icon: "▶", tag: "page" },
  { id: "nav-tower",    label: "Control Tower",       desc: "Incidents, runbooks, runtime",        icon: "!", tag: "page" },
  { id: "nav-debug",    label: "Tools",               desc: "Debug actions & history",             icon: "⚙", tag: "page" },
  { id: "nav-recovery", label: "Recovery",            desc: "Guided backend recovery wizard",      icon: "↺", tag: "page" },
  { id: "nav-logs",     label: "Logs",                desc: "Log browser & stream viewer",         icon: "≡", tag: "page" },
  { id: "nav-trades",   label: "Signals",             desc: "Trade signals & stability events",    icon: "~", tag: "page" },
  { id: "nav-research", label: "Research",            desc: "Pockets, watchboard, fitness",        icon: "◆", tag: "page" },
  { id: "nav-settings", label: "Settings",            desc: "Config & diagnostics",                icon: "⚙", tag: "page" },
];

const LOG_QUICK_PACKS: Omit<CmdItem, "action">[] = [
  { id: "logs-regime",   label: "Logs: Regime filter",    desc: "Open logs with Regime pack",    icon: "≡", tag: "logs" },
  { id: "logs-shutdown", label: "Logs: Shutdown filter",  desc: "Open logs with Shutdown pack",  icon: "≡", tag: "logs" },
  { id: "logs-timeout",  label: "Logs: Timeout filter",   desc: "Open logs with Timeout pack",   icon: "≡", tag: "logs" },
  { id: "logs-nomatch",  label: "Logs: No Match filter",  desc: "Open logs with No Match pack",  icon: "≡", tag: "logs" },
];

const OPS_ITEMS: Omit<CmdItem, "action">[] = [
  { id: "ops-triage",    label: "Run Triage Macro",       desc: "Execute full triage debug sequence",  icon: "▶", tag: "action" },
  { id: "ops-ack",       label: "Ack All Incidents",      desc: "Open tower to bulk-ack incidents",    icon: "✓", tag: "action" },
  { id: "ops-incident",  label: "Start Incident Session", desc: "Open logs to start incident session", icon: "!", tag: "action" },
];

function fuzzyMatch(query: string, target: string): boolean {
  if (!query) return true;
  const q = query.toLowerCase();
  const t = target.toLowerCase();
  if (t.includes(q)) return true;
  // simple fuzzy
  let qi = 0;
  for (let i = 0; i < t.length && qi < q.length; i++) {
    if (t[i] === q[qi]) qi++;
  }
  return qi === q.length;
}

interface Props {
  onClose: () => void;
}

export default function CommandPalette({ onClose }: Props) {
  const navigate = useNavigate();
  const [query, setQuery]       = useState("");
  const [focused, setFocused]   = useState(0);
  const [actions, setActions]   = useState<{ label: string; action: string; desc: string }[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef  = useRef<HTMLDivElement>(null);

  // Load debug actions once
  useEffect(() => {
    api.debugActions().then((rows) => {
      setActions(rows.map((r) => ({
        label: r.action,
        action: r.action,
        desc: r.description || "",
      })));
    }).catch(() => { /* ignore */ });
  }, []);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const allItems: CmdItem[] = useMemo(() => {
    const goTo = (path: string, params?: string) => {
      navigate(params ? `${path}?${params}` : path);
      onClose();
    };

    const nav: CmdItem[] = NAV_ITEMS.map((n) => ({
      ...n,
      action: () => {
        const routeMap: Record<string, string> = {
          "nav-status": "/", "nav-live": "/live", "nav-tower": "/tower",
          "nav-debug": "/debug", "nav-recovery": "/recovery", "nav-logs": "/logs",
          "nav-trades": "/trades", "nav-research": "/research", "nav-settings": "/settings",
        };
        goTo(routeMap[n.id] ?? "/");
      },
    }));

    const logPacks: CmdItem[] = LOG_QUICK_PACKS.map((lp) => ({
      ...lp,
      action: () => {
        const packMap: Record<string, string> = {
          "logs-regime": "Regime", "logs-shutdown": "Shutdown",
          "logs-timeout": "Timeout", "logs-nomatch": "No%20Match",
        };
        goTo("/logs", `pack=${packMap[lp.id]}`);
      },
    }));

    const ops: CmdItem[] = OPS_ITEMS.map((op) => ({
      ...op,
      action: () => {
        if (op.id === "ops-triage")   goTo("/debug", "auto=triage");
        else if (op.id === "ops-ack") goTo("/tower");
        else if (op.id === "ops-incident") goTo("/logs", "ops=start_incident");
      },
    }));

    const dbgActions: CmdItem[] = actions.map((a) => ({
      id: `action-${a.action}`,
      label: `Run: ${a.action}`,
      desc: a.desc,
      icon: "▷",
      tag: "tool",
      action: () => goTo("/debug", `action=${encodeURIComponent(a.action)}`),
    }));

    return [...nav, ...logPacks, ...ops, ...dbgActions];
  }, [navigate, onClose, actions]);

  const filtered = useMemo(() => {
    if (!query.trim()) return allItems;
    return allItems.filter(
      (it) => fuzzyMatch(query, it.label) || fuzzyMatch(query, it.desc ?? "")
    );
  }, [query, allItems]);

  // Group by tag
  const grouped = useMemo(() => {
    const groups: Record<string, CmdItem[]> = {};
    for (const it of filtered) {
      if (!groups[it.tag]) groups[it.tag] = [];
      groups[it.tag].push(it);
    }
    return groups;
  }, [filtered]);

  const flatFiltered = filtered;

  useEffect(() => { setFocused(0); }, [query]);

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") { e.preventDefault(); setFocused((v) => Math.min(v + 1, flatFiltered.length - 1)); }
    if (e.key === "ArrowUp")   { e.preventDefault(); setFocused((v) => Math.max(v - 1, 0)); }
    if (e.key === "Enter")     { e.preventDefault(); flatFiltered[focused]?.action(); }
    if (e.key === "Escape")    { onClose(); }
  }

  const tagLabels: Record<string, string> = {
    page: "Pages", logs: "Log Filters", action: "Quick Actions", tool: "Debug Tools",
  };

  let globalIdx = 0;

  return (
    <div className="cmd-overlay" onClick={onClose}>
      <div className="cmd-modal" onClick={(e) => e.stopPropagation()}>
        <div className="cmd-input-row">
          <span className="cmd-input-icon">⌕</span>
          <input
            ref={inputRef}
            className="cmd-input"
            placeholder="Search pages, tools, log filters..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKey}
            spellCheck={false}
          />
          {query && (
            <button
              style={{ background: "none", border: "none", color: "var(--muted)", cursor: "pointer", fontSize: 16 }}
              onClick={() => setQuery("")}
            >×</button>
          )}
        </div>

        <div className="cmd-results" ref={listRef}>
          {flatFiltered.length === 0 && (
            <div className="cmd-empty">No results for "{query}"</div>
          )}
          {Object.entries(grouped).map(([tag, items]) => (
            <div key={tag}>
              <div className="cmd-section-label">{tagLabels[tag] ?? tag}</div>
              {items.map((it) => {
                const myIdx = globalIdx++;
                const isFocused = myIdx === focused;
                return (
                  <div
                    key={it.id}
                    className={`cmd-item${isFocused ? " focused" : ""}`}
                    onClick={it.action}
                    onMouseEnter={() => setFocused(myIdx)}
                  >
                    <div className="cmd-item-icon">{it.icon}</div>
                    <div className="cmd-item-text">
                      <div className="cmd-item-label">{it.label}</div>
                      {it.desc && <div className="cmd-item-desc">{it.desc}</div>}
                    </div>
                    <span className="cmd-item-tag">{it.tag}</span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        <div className="cmd-footer">
          <span>↑↓ navigate</span>
          <span>↵ open</span>
          <span>esc close</span>
        </div>
      </div>
    </div>
  );
}
