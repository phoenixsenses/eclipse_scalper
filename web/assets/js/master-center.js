/* ============================================================
   ECLIPSE — Master Center demo view
   Renders a sample operator console. Every value in this file is
   illustrative sample data written by hand. Nothing here reads a
   database, a ledger, an exchange or a live process.
   ============================================================ */
(function () {
  "use strict";

  /* A PROJECTION, not a report. None of the agents below exist yet; only the
     Master Center and the event bus are under construction. Every status,
     version and number in this file is invented to show the console's shape. */
  var AGENTS = [
    { n: "Master Center",       id: "mc-core",    s: "system",   v: "0.1.0", note: "registry, policy, journal" },
    { n: "Alpha Engine",        id: "alpha-s34",  s: "active",   v: "0.9.4", note: "candidate generation" },
    { n: "Market Intelligence", id: "intel-glb",  s: "active",   v: "0.2.1", note: "context only" },
    { n: "Research Engine",     id: "res-lab",    s: "research", v: "0.6.0", note: "experiments running" },
    { n: "Risk Governor",       id: "risk-gov",   s: "active",   v: "0.4.2", note: "authorising" },
    { n: "Security Guardian",   id: "sec-grd",    s: "active",   v: "0.3.0", note: "deny by default" },
    { n: "Data Guardian",       id: "data-grd",   s: "warning",  v: "0.5.1", note: "1 feed gap quarantined" },
    { n: "Execution Gateway",   id: "exec-gw",    s: "idle",     v: "0.2.0", note: "no approved orders" },
    { n: "PR Guardian",         id: "pr-grd",     s: "active",   v: "0.1.4", note: "org-wide" },
    { n: "Observability",       id: "otel-hub",   s: "active",   v: "0.2.2", note: "collecting" }
  ];

  var STATS = [
    { k: "Agents online",     v: "9",  tone: "green" },
    { k: "Research jobs",     v: "14", tone: "cyan" },
    { k: "Frozen arms",       v: "3",  tone: "blue" },
    { k: "Trade candidates",  v: "1",  tone: "blue" },
    { k: "Approved orders",   v: "0",  tone: "zero" },
    { k: "Security alerts",   v: "0",  tone: "zero" }
  ];

  var LOG = [
    { t: "14:44", k: "system",   m: "<b>PR Guardian</b> reviewed eclipse-research #31 — 24/24 tests, no risk-limit change" },
    { t: "14:43", k: "research", m: "<b>Research Engine</b> created dataset snapshot <b>ds-20260821-4471</b>" },
    { t: "14:42", k: "blocked",  m: "<b>Risk Governor</b> rejected candidate — macro event proximity (8 min)" },
    { t: "14:42", k: "risk",     m: "<b>Risk Governor</b> began evaluation of <b>ETH-20260821-001</b>" },
    { t: "14:42", k: "alpha",    m: "<b>Alpha Engine</b> published <code>eclipse.alpha.trade_candidate</code>" },
    { t: "14:39", k: "system",   m: "<b>Data Guardian</b> quarantined a 24-minute book feed gap" },
    { t: "14:31", k: "research", m: "<b>Research Engine</b> walk-forward complete — arm held, no promotion" },
    { t: "14:12", k: "security", m: "<b>Security Guardian</b> issued scoped identity to <b>intel-glb</b>" }
  ];

  var ARMS = [
    { n: "E-DER V1",     st: "frozen",   pop: "25 events / 6 cascades",   note: "strict original definition" },
    { n: "E-DER V2 (A2)", st: "frozen",  pop: "83 events / 8 cascades",   note: "controlled relaxation arm" },
    { n: "E-DER V3",     st: "research", pop: "198 events / 10 cascades", note: "recovery variant" }
  ];

  var DECISION = [
    ["Trade id",     "ETH-20260821-001"],
    ["Alpha arm",    "E-DER V1 (frozen)"],
    ["Timing",       "frozen in advance; specifics not published"],
    ["Market context", "risk-neutral; DXY flat; no regime change"],
    ["Risk verdict", "REJECTED — macro event within 8 minutes"],
    ["Position",     "none opened"],
    ["Journal ref",  "dj-20260821-0f3c91"]
  ];

  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; }); }

  function render(host) {
    var h = [];

    /* rides with the reader: the colours below belong to components that do
       not exist, so the disclaimer must never scroll away from them */
    h.push('<div class="proj-bar"><i></i>Projected — nothing here is running</div>');

    h.push('<div class="note"><div class="note-t">Projection — none of this is running</div>' +
      '<b>Every agent, version, status and number below is invented</b> to show the shape of the ' +
      'console. None of these agents has been built: today only the Master Center and the event ' +
      'bus exist, and both are under construction. See the ' +
      '<a href="changelog.html" style="color:var(--dim)">changelog</a> for what is actually ' +
      'implemented. This page reads no database, ledger, exchange or running process, and the real ' +
      'console will live on the private network only.</div>');

    h.push('<div class="dash-bar" style="margin-top:24px">');
    STATS.forEach(function (s) {
      h.push('<div class="stat"><div class="stat-k">' + esc(s.k) + '</div>' +
             '<div class="stat-v" data-tone="' + s.tone + '">' + esc(s.v) + '</div></div>');
    });
    h.push("</div>");

    h.push('<div class="split-even" style="margin-top:24px">');

    /* agents */
    h.push('<div class="panel"><div class="panel-hd">Agent registry — projected</div><div class="kv">');
    AGENTS.forEach(function (a) {
      h.push('<div class="kv-row"><div class="kv-k">' +
        '<span class="tag" data-s="' + a.s + '">' + esc(a.s) + "</span></div>" +
        '<div class="kv-v">' + esc(a.n) + '<br><span style="color:var(--faint)">' +
        esc(a.id) + " &nbsp;v" + esc(a.v) + " &nbsp;· " + esc(a.note) + "</span></div></div>");
    });
    h.push("</div></div>");

    /* event log */
    h.push('<div class="panel"><div class="panel-hd">Event log — projected</div><div class="log">');
    LOG.forEach(function (e) {
      h.push('<div class="log-r" data-k="' + e.k + '"><div class="log-t">' + esc(e.t) +
        '</div><div class="log-d"></div><div class="log-m">' + e.m + "</div></div>");
    });
    h.push("</div></div>");

    h.push("</div>");

    /* decision journal */
    h.push('<div class="split-even" style="margin-top:24px">');
    h.push('<div class="panel"><div class="panel-hd">Decision journal — projected example</div><div class="kv">');
    DECISION.forEach(function (r) {
      h.push('<div class="kv-row"><div class="kv-k">' + esc(r[0]) + '</div><div class="kv-v">' + esc(r[1]) + "</div></div>");
    });
    h.push("</div></div>");

    /* research arms */
    h.push('<div class="panel"><div class="panel-hd">Frozen research arms</div><div class="kv">');
    ARMS.forEach(function (a) {
      h.push('<div class="kv-row"><div class="kv-k"><span class="tag" data-s="' + a.st + '">' +
        esc(a.st) + '</span></div><div class="kv-v">' + esc(a.n) +
        '<br><span style="color:var(--faint)">' + esc(a.pop) + " · " + esc(a.note) + "</span></div></div>");
    });
    h.push('</div><p class="mono-note" style="margin:16px 0 0">Arms are tracked side by side. ' +
      'None supersedes another, and none is promoted by this view.</p></div>');
    h.push("</div>");

    host.innerHTML = h.join("");
    host.setAttribute("tabindex", "-1");
    host.focus({ preventScroll: true });
  }

  var btn = document.getElementById("open-demo");
  var gate = document.getElementById("gate");
  var host = document.getElementById("console");
  if (!btn || !gate || !host) return;

  btn.addEventListener("click", function () {
    gate.hidden = true;
    render(host);
  });
})();
