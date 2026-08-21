# Eclipse — website

Static, dependency-free public site for the Eclipse platform. No build step, no npm,
no external requests (no CDN fonts, scripts or images) — it opens from disk or from
any static host.

## Run

```powershell
cd D:\eclipse_scalper\web
python -m http.server 8799 --bind 127.0.0.1
```

Then open <http://127.0.0.1:8799>.

## Pages

| File | Purpose |
|---|---|
| `index.html` | Landing. 15 sections on the rail: Master Center → Event bus → Path of a trade → Alpha → Market Intelligence → Research → Risk → Execution → Security → Data → PR Guardian → Observability → Dynamic agents → Infrastructure → Roadmap |
| `architecture.html` | Topology, agent contract, event model, environments, repositories, workflow, the 8 principles |
| `research.html` | Standard of evidence, frozen arms, open hypotheses, burned-sample discipline, graveyard |
| `research-e-der.html` | E-DER concept, design constraints, the three arms, refutation conditions, pipeline position |
| `agents.html` | Registry — role plus publish / read / denied for each agent |
| `security.html` | Onboarding sequence, ten layers, permission matrix, secrets, research↔execution border, CI checks |
| `infrastructure.html` | Stack, node roles, private mesh, storage model, source-control decision |
| `status.html` | Master Center demo console (sample data, behind a demo gate) |
| `changelog.html` | Agent, arm and schema versions plus the versioning rules |
| `404.html` | Not-found page. Uses root-absolute links so it works from any depth |

## Assets

- `assets/favicon.svg` — the eclipse disc, linked from every page
- `assets/og-eclipse.png` — 1200×630 social preview card
- `tools/make_og.py` — regenerates the card: `python web/tools/make_og.py`.
  Uses Bahnschrift's Bold Condensed instance so the card matches the site.

## Before deploying

Every page carries `og:url` / `canonical` pointing at the placeholder
`https://eclipse.example`. **Replace it with the real origin** before publishing:

```powershell
cd D:\eclipse_scalper\web
(Get-ChildItem *.html) | ForEach-Object {
  (Get-Content $_ -Raw).Replace('https://eclipse.example','https://REAL-ORIGIN') |
    Set-Content $_ -Encoding utf8
}
```

Serve `404.html` as the not-found document. Nothing else needs configuring —
there is no backend, no API and no build.

## Design system

`assets/css/eclipse.css` — one file, tokens at the top.

- **The Rail.** A hairline down the left gutter of every page is the global event bus.
  Each section is a tap point with a lamp; slow packets travel it. The metaphor is the
  information architecture, not decoration.
- **Rail lamps show IMPLEMENTATION STATE, never runtime health.** Four states, none of
  them a health colour: `building` (exists as code, under construction) · `design`
  (specified, not built) · `planned` (neither) · `concept` (the section describes an
  idea, rule or method rather than a component). The bulb strips use the same
  vocabulary. A section that is concept-only must not claim health of any kind.
- **Type.** Display face `Bahnschrift` → `DIN Alternate` → `Arial Narrow` (condensed
  DIN-descended grotesque, ships on Windows and macOS — no downloaded fonts). Body is
  system-ui. All labels, states and numbers are monospace.
- **Colour is semantic only, and health colours are reserved.** `green`, `amber` and
  `red` mean healthy / warning / blocked and may **never** label a component, a lamp,
  an agent or a section — nothing on this site is running, so any such use is a false
  claim. They appear in exactly two places, both independently reviewed and accepted as
  non-health semantics: the approve / reduce / reject verdict legend, and the completed
  steps in the E-DER pipeline position. Everything structural uses the implementation
  states above; cyan and violet remain available for category accents that carry no
  status meaning.
- Single dark theme, 2px radius, hairlines at 9% white.
- Responsive to 360px, keyboard focus visible, `prefers-reduced-motion` honoured
  (packets and corona freeze, reveals become instant).

## Content policy — read before adding anything

This site is written to be **public**. Two rules govern what may appear:

**Never claim health**

Nothing in Eclipse is running. No lamp, chip, tag or label anywhere on this site may
read `Active`, `Healthy`, `Running` or carry a green/amber/red status colour for a
component. Use `Building` / `Design` / `Planned` / `Not implemented`. This was the
subject of review finding W2 and its two follow-ups; it keeps regressing because the
palette makes green easy to reach for.

**Never publish**

- Entry/exit rules, offsets, horizons, thresholds, feature definitions, formulas
- Any performance figure — bps, win rate, profit factor, drawdown, totals, or a
  comparison that implies one
- Anything derived from a sealed forward arm, in any aggregated form
- Hostnames, IPs, ports, credentials, real network layout, live positions

**Safe to publish**

- Architecture, agent roles, permission model, security philosophy
- Concept-level descriptions of research ideas
- Population sizes (event and cascade counts) and implementation-state labels
- Component choices and logical topology

`changelog.html` deliberately has no results column, and `status.html` states that the
console withholds sealed aggregates from operators too. Keep both.

## The demo console

`status.html` renders sample data from `assets/js/master-center.js`. Every value in
that file is written by hand. There is no authentication, no fetch, and no database —
the gate is a button and says so. If a real console is ever wired up it belongs on the
private mesh, not here.
