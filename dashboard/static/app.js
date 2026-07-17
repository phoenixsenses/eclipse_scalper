/* Canonical S34 operator dashboard — read-only client.
 * Fetches /api/overview (GET), renders typed panels with trust overlays.
 * Never issues a mutating request; there are no control actions in this UI. */
"use strict";

const REFRESH_MS = 7000;
const NON_CURRENT = new Set(["stale", "missing", "malformed", "inferred"]);

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}

function fmtAge(sec) {
  if (sec === null || sec === undefined) return "—";
  sec = Number(sec);
  if (sec < 60) return sec.toFixed(0) + "s";
  if (sec < 3600) return (sec / 60).toFixed(1) + "m";
  if (sec < 86400) return (sec / 3600).toFixed(1) + "h";
  return (sec / 86400).toFixed(1) + "d";
}

function copyBtn(text) {
  const b = el("button", "copy-btn", "copy");
  b.type = "button";
  b.setAttribute("aria-label", "Copy " + text);
  b.addEventListener("click", () => {
    if (navigator.clipboard) navigator.clipboard.writeText(text).catch(() => {});
  });
  return b;
}

function renderHeader(overview) {
  const hdr = document.getElementById("safety-header");
  const fieldsEl = document.getElementById("sh-fields");
  const pill = document.getElementById("overall-pill");
  const refresh = document.getElementById("last-refresh");
  fieldsEl.textContent = "";

  const eh = (overview.panels && overview.panels.executive_header) || {};
  const f = eh.fields || {};
  const liveOn = f.live_executor === "ON";
  hdr.classList.toggle("live-on", liveOn);

  const kvs = [
    ["mode", f.mode || "—"],
    ["live_exec", (liveOn ? "⚠ ON" : "OFF"), liveOn],
    ["scheduler", f.scheduler_state || "—"],
    ["branch", (f.branch || "?") + "@" + (f.head_short || "?")],
    ["gov", f.canonical_governance_state || "—"],
    ["watchdog", f.watchdog_ok ? "ok" : "MISSING", !f.watchdog_ok],
    ["open C/H/M", (f.open_finding_counts ?
      `${f.open_finding_counts.CRITICAL||0}/${f.open_finding_counts.HIGH||0}/${f.open_finding_counts.MEDIUM||0}` : "—")],
    ["oldest_age", fmtAge(overview.oldest_artifact_age_seconds)],
  ];
  for (const [k, v, warn] of kvs) {
    const s = el("span", "kv");
    s.appendChild(el("span", null, k + ":"));
    const b = el("b", warn ? "warn" : null, " " + v);
    s.appendChild(b);
    fieldsEl.appendChild(s);
  }

  const sev = (overview.overall_severity || "unknown").toLowerCase();
  pill.className = "pill pill-" + sev;
  pill.textContent = sev + (overview.critical_blocker_count ? ` · ${overview.critical_blocker_count} blocker` : "");
  const iso = overview.generated_at_iso || "";
  refresh.textContent = "refresh: " + (iso ? iso.replace("T", " ").slice(0, 19) + "Z" : "—");
}

function renderPanel(key, p) {
  const trust = (p.trust_state || "current").toLowerCase();
  const sev = (p.severity || "ok").toLowerCase();
  const panel = el("section", "panel " + (NON_CURRENT.has(trust) ? trust : ""));
  panel.setAttribute("aria-labelledby", "h-" + key);

  const hd = el("div", "panel-hd");
  hd.appendChild(el("span", "sev-dot sev-" + sev));
  const h = el("h2", null, p.title || key);
  h.id = "h-" + key;
  hd.appendChild(h);
  const tb = el("span", "trust-badge trust-" + trust, trust);
  hd.appendChild(tb);
  panel.appendChild(hd);

  const body = el("div", "panel-body");
  if (p.summary) body.appendChild(el("div", "panel-summary", p.summary));

  // fields
  const fields = p.fields || {};
  const keys = Object.keys(fields);
  if (keys.length) {
    const dl = el("dl", "kv-grid");
    for (const fk of keys) {
      let v = fields[fk];
      dl.appendChild(el("dt", null, fk));
      let cls = null;
      if (v === true) cls = "true"; else if (v === false) cls = "false";
      if (v !== null && typeof v === "object") v = JSON.stringify(v);
      dl.appendChild(el("dd", cls, v === null || v === undefined ? "—" : String(v)));
    }
    body.appendChild(dl);
  }

  // rows table (deterministic column order from first row)
  if (Array.isArray(p.rows) && p.rows.length) {
    const wrap = el("div", "rows-wrap");
    const t = el("table", "rows");
    const cols = Object.keys(p.rows[0]);
    const thead = el("thead"); const trh = el("tr");
    cols.forEach((c) => trh.appendChild(el("th", null, c)));
    thead.appendChild(trh); t.appendChild(thead);
    const tb2 = el("tbody");
    p.rows.forEach((r) => {
      const tr = el("tr");
      cols.forEach((c) => {
        let cell = r[c];
        if (cell !== null && typeof cell === "object") cell = JSON.stringify(cell);
        tr.appendChild(el("td", null, cell === null || cell === undefined ? "—" : String(cell)));
      });
      tb2.appendChild(tr);
    });
    t.appendChild(tb2); wrap.appendChild(t); body.appendChild(wrap);
  }

  // trust meta line
  const meta = el("div", "meta-line");
  meta.appendChild(el("span", null, "trust=" + trust));
  meta.appendChild(el("span", null, "parse=" + (p.parse_status || "?")));
  meta.appendChild(el("span", null, "prov=" + (p.provenance_status || "?")));
  meta.appendChild(el("span", null, "age=" + fmtAge(p.age_seconds)));
  if (p.freshness_threshold_seconds != null)
    meta.appendChild(el("span", null, "threshold=" + fmtAge(p.freshness_threshold_seconds)));
  if (p.source_path) {
    const sp = el("span", null, "src=" + p.source_path);
    meta.appendChild(sp);
    meta.appendChild(copyBtn(p.source_path));
  }
  body.appendChild(meta);

  // reason codes
  if (Array.isArray(p.reason_codes) && p.reason_codes.length) {
    const rc = el("div", "reasons");
    p.reason_codes.forEach((c) => rc.appendChild(el("span", "reason", c)));
    body.appendChild(rc);
  }

  // raw evidence expandable
  if (p.raw_evidence !== null && p.raw_evidence !== undefined) {
    const d = el("details", "raw");
    d.appendChild(el("summary", null, "raw evidence"));
    d.appendChild(el("pre", null, JSON.stringify(p.raw_evidence, null, 2)));
    body.appendChild(d);
  }

  panel.appendChild(body);
  return panel;
}

function render(overview) {
  renderHeader(overview);
  const grid = document.getElementById("panels");
  grid.textContent = "";
  const order = overview.panel_order || Object.keys(overview.panels || {});
  for (const key of order) {
    const p = (overview.panels || {})[key];
    if (p) grid.appendChild(renderPanel(key, p));
  }
}

async function refresh() {
  try {
    const resp = await fetch("/api/overview", { method: "GET", headers: { "Accept": "application/json" } });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    render(data);
  } catch (e) {
    const grid = document.getElementById("panels");
    const err = el("p", "grid-error", "Failed to load overview: " + e.message + " (data not shown as fresh).");
    grid.textContent = "";
    grid.appendChild(err);
    const refreshEl = document.getElementById("last-refresh");
    if (refreshEl) refreshEl.textContent = "refresh: FAILED";
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
