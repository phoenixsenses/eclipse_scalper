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
| `roadmap.html` | Twelve phases in order — what each must deliver, the five-step review gate that closes one, and the current state of each. No dates, ever |
| `arms.html` | Arms & lanes — every arm, lane and closed idea as a card: name, concept description, implementation state. No results column |
| `methodology.html` | Nine concept sections on how the evidence is handled — no formula, threshold or measured value |
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
- **Rail lamps show IMPLEMENTATION STATE, never runtime health.** Seven states, none of
  them a health colour: `accepted` (built **and** passed an independent review gate) ·
  `building` (exists as code, under construction) · `design` (specified, not built) ·
  `planned` (neither) · `concept` (the section describes an idea, rule or method rather
  than a component) · `refuted` (closed by a test, kept on the record so it is not
  rediscovered) · `parked` (not refuted, but blocked on something that does not exist
  yet). **`accepted` is ink, never green:** it describes the review state of the code and
  says nothing about a running thing or a market result — write that sentence next to it
  wherever it appears. The bulb strips and tags use the same vocabulary.
  A section that is concept-only must not claim health of any kind. `refuted` is
  deliberately **not** red: a closed idea is not a failure state of a running thing.
- **Type.** Display face `Bahnschrift` → `DIN Alternate` → `Arial Narrow` (condensed
  DIN-descended grotesque, ships on Windows and macOS — no downloaded fonts). Body is
  system-ui. All labels, states and numbers are monospace.
- **Colour is semantic only, and health colours are reserved.** `green`, `amber` and
  `red` mean healthy / warning / blocked and may **never** label a component, a lamp,
  an agent or a section — nothing on this site is running, so any such use is a false
  claim. **Do not try to state this as a count.** It has been written as "exactly two
  places" and then as "exactly three", and both were false the day they were written —
  the site actually carries around twenty-five coloured declarations. What holds is the
  *kind* of thing being coloured, and it is now machine-checked by
  `tools/check_policy.py`, which refuses any occurrence it cannot place in one of these:
  · **prohibition** — red on something forbidden or wrong (`Denied`, `Never versioned in
  place`, the deliberately wrong example) · **custody** — amber on what an agent holds ·
  **verdict** — the approve / reduce / reject legend · **progress** — a completed step in
  the E-DER pipeline position · **gate** — a promotion step that needs a person, and only
  inside a step that declares `data-gate` · **stage** — a node in a drawn flow, and only
  when that node names a *market state* rather than an Eclipse component. None of those
  says a component is well.
  Colour also arrives through attributes, not just inline styles: `data-s`, `data-accent`,
  `data-k` and `data-gate` are painted by the stylesheet, and the checker derives that set
  *from the stylesheet* so a new health-coloured state cannot be added behind its back.
  That is how three component nodes in the landing page's path-of-a-trade flow were found
  wearing amber and green after four review passes had read straight past them.
  The **one** genuine exception is the **projected demo console** in `status.html`: a
  mockup of a future operator screen whose `active` / `warning` / `idle` chips and green
  count belong to agents that do not exist. It is allowed only while it is fenced three
  ways — the demo gate, panel headings that all read *projected*, and a sticky `Projected
  — nothing here is running` bar that cannot scroll away from the colours it disclaims.
  The checker fails if that bar disappears. **No page outside that console may use this
  exception.** Everything structural uses the implementation states above; cyan and violet remain available for category accents that carry no
  status meaning.
- Single dark theme, 2px radius, hairlines at 9% white.
- Responsive to 360px, keyboard focus visible, `prefers-reduced-motion` honoured
  (packets and corona freeze, reveals become instant). Every page opens with a
  `Skip to content` link that stays off-screen until it takes focus, jumping to
  `<main id="main">` so a keyboard user need not walk the nav on every page.
  **Check 360px after adding any chip or tag:** a `.tag` is `white-space: nowrap`, and
  before `.eyebrow` was given `flex-wrap`, one `Not implemented` tag pushed
  `agents.html` 98px wider than the viewport.

## Check it, do not eyeball it

```powershell
python web/tools/check_policy.py     # 0 = clean, 1 = at least one violation
```

Run it after touching any page. It enforces the content policy below plus the structural
invariants — health colour by category, banned labels, performance figures, horizon
suffixes, ranking vocabulary, unclosed tags, duplicate ids, dead links and fragments,
undefined CSS classes, the skip link, and external requests. It exists because this
policy has failed independent review four times while being read carefully every time;
the two rules it would have caught immediately are W2 (`Active` under a green chip) and
the README's own false count of where colour appears.

It also enforces **one component, one state**: the same component may not be `Building`
on one page and `Accepted` on another. That rule exists because the site had drifted into
exactly that — `Alpha` was claimed in three different states at once, and `agents.html`
contradicted *itself*, its bulb strip saying `Design` while the section it links to said
`Not implemented`.

It is deliberately mutation-tested: **twenty-one** deliberate violations are injected into
a scratch copy and all twenty-one must be caught — a green `Active` chip, a health state
carried by `data-s` with innocuous text, a health colour on a component node in a flow, a
bps figure, a win rate with the number written first, a figure spelled out in words, a
`4H` suffix, a ranking word, the console fence removed, a banned label hidden in a script,
a dead link, a duplicate id, an unclosed tag, an undefined class, a missing skip link, an
external request, a stale roadmap phase, and two pages disagreeing about a state.

**Extend the checker and you must mutate it again.** When these rules were widened, one
of them was silently broken: the `` in two patterns was written as a literal backspace
character, so the performance-figure rule matched nothing at all and reported a clean
site. It was caught only because the mutants were re-run afterwards. A rule that never
fires reads exactly like a rule that passes.

## Content policy — read before adding anything

This site is written to be **public**. Two rules govern what may appear:

**Never claim health**

Nothing in Eclipse is running. No lamp, chip, tag or label anywhere on this site may
read `Active`, `Healthy`, `Running` or carry a green/amber/red status colour for a
component. Use `Building` / `Design` / `Planned` / `Not implemented`. This was the
subject of review finding W2 and its two follow-ups; it keeps regressing because the
palette makes green easy to reach for.

The **one** exception is the projected demo console described under *Design system*, and
it holds only inside `#console` on `status.html`, only behind the demo gate, and only
while the sticky `Projected — nothing here is running` bar renders above it. If that bar
is ever removed, the console's colours and `active` chips must go with it — and
`tools/check_policy.py` will fail until one of the two is put back.

**Never publish**

- Entry/exit rules, offsets, horizons, thresholds, feature definitions, formulas
- **Horizon suffixes in arm names.** An internal name like `Operational Control 4H`
  publishes as `Operational Control`. The suffix is a horizon, and a horizon is a
  threshold. Strip every `4H` / `6H` / `+31m` / `30m` from names and descriptions
- **Any ranking or comparison between arms** — "strongest", "most promising", "leading",
  "best". A ranking is a performance comparison with the number removed. List arms in a
  given order and describe each on its own terms. Saying an idea was *refuted* is safe;
  saying one is *promising* leaks the edge
- Any performance figure — bps, win rate, profit factor, drawdown, totals, or a
  comparison that implies one
- Anything derived from a sealed forward arm, in any aggregated form
- Hostnames, IPs, ports, credentials, real network layout, live positions

**Keep one state per component**

A component's implementation state appears on several surfaces — the landing bulb strip,
`agents.html`, `roadmap.html`, `changelog.html`. They must agree. When a phase closes,
grep for the component and change every surface in the same pass, then run the checker;
it compares them for you. Updating one page and not the others is how the site ends up
telling two stories about itself.

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

The console is the site's only sanctioned use of health colour, so it carries a sticky
`Projected — nothing here is running` bar (`.proj-bar`, rendered first by
`master-center.js`). It is sticky on purpose: the earlier disclaimer was a note at the
top that scrolled away, leaving a reader looking at green `active` chips with no visible
statement that none of those agents exist.
