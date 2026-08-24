# REPOSITORY METADATA RECOMMENDATIONS

The GitHub **About** panel is the second thing a visitor reads and the only thing a
search result shows. At the time of this audit it was empty: no description, no website,
no topics, no social preview.

**Nothing in this document has been applied.** Repository settings are changed through
the GitHub web interface or the API, and doing that without being asked would be an
outward-facing change to an account this work has no mandate over. Everything below is a
recommendation with exact copy, ready to paste.

---

## 1. Description

GitHub's description field allows 350 characters. Shorter is better — it is truncated in
search results and in the repository card.

**Recommended (183 characters):**

```
Mechanism-first market microstructure research and execution framework for perpetual futures — built around reproducibility, forward validation and risk-governed execution.
```

**Shorter alternative (118 characters),** if you prefer the card to read in one glance:

```
Mechanism-first market microstructure research and execution framework for perpetual futures.
```

Deliberately absent, and worth stating so it does not get added back later: no
performance claim, no "profitable", no "AI", no symbol name, and no adjective doing work
a number should be doing.

---

## 2. Website field

Three options, in order of preference.

**(a) Leave it empty for now.** The static site in `web/` is not deployed, and every page
still carries the placeholder origin `https://eclipse.example` in its `og:url` and
`canonical` tags. Pointing the About panel at a site that does not resolve is worse than
pointing it nowhere.

**(b) Publish `web/` and point at it.** It is a static site with no build step, no npm and
no external request, so GitHub Pages serves it directly. **Before publishing**, replace
the placeholder origin — `web/README.md` carries the exact one-line command — and serve
`404.html` as the not-found document.

**(c) Point at the documentation index:**

```
https://github.com/phoenixsenses/eclipse_scalper/blob/main/docs/public/README.md
```

Honest but redundant, since the visitor is already on the repository.

**Recommendation: (a) now, (b) once the origin placeholder is replaced.**

---

## 3. Topics

GitHub allows up to 20 topics. Each must be lowercase, may contain hyphens, and is capped
at 35 characters. Accuracy matters more than volume — a topic that does not describe the
repository makes every other one less credible.

**Recommended set (12):**

```
market-microstructure
quantitative-finance
quantitative-research
algorithmic-trading
order-book
execution
risk-management
reproducible-research
experimental-design
backtesting
binance-futures
python
```

**Notes on what was left out and why:**

| Rejected | Reason |
|---|---|
| `cryptocurrency` | true but low-signal; `binance-futures` already carries the venue |
| `trading-bot` | attracts exactly the audience this README is written to disappoint |
| `machine-learning` | the repository does not currently centre on model fitting |
| `hft` | wrong scale, and a claim about latency infrastructure this project does not make |
| `alpha` | a claim, not a topic |
| `fastapi`, `react`, `sqlite` | implementation details of a subsystem, not what the project is |

`experimental-design` and `reproducible-research` are the two that make this repository
findable by the people most likely to value it, and are the two most projects in this
space cannot honestly claim.

---

## 4. Social preview

GitHub recommends 1280 × 640 px and accepts PNG, JPG or GIF (**not SVG**), up to 1 MB.
It is uploaded manually under *Settings → General → Social preview*.

**Source, authored at exactly 1280 × 640:**

```
docs/assets/07_social_preview.svg
```

**To produce the PNG.** No rasterisation library is installed in this repository's
interpreter — `Pillow` and `cairosvg` are both absent — so this is a manual step. Three
ways, in order of least effort:

1. **Browser.** Open the SVG, set the window to 1280 × 640, screenshot. Fastest, and
   fonts render exactly as a viewer would see them.
2. **`docs/assets/make_social_preview.py`.** Draws the card directly with Pillow at
   1280 × 640, mirroring `web/tools/make_og.py`. Needs `pip install Pillow`.
3. **Any SVG converter** — `rsvg-convert`, Inkscape, `cairosvg` — at 1280 × 640.

**Interim option that needs no work at all:** `web/assets/og-eclipse.png` already exists
at 1200 × 630, is Eclipse-branded, and carries no numbers by the same policy. GitHub
accepts it; it is 80 px narrower than the recommendation and will be letterboxed
slightly.

---

## 5. Things that are owner decisions, not metadata

Collected here so they are in one place rather than scattered through the risk register.

| # | Decision | Why it cannot be made here |
|---|---|---|
| 1 | **Licence.** There is no `LICENSE` file, and the README's MIT badge — which claimed terms that were never granted — has been removed. Until a licence is added, default copyright applies and nobody has any grant. | Licensing is an ownership decision with legal effect. Choosing one silently would be worse than the badge was. |
| 2 | **`docs/protocols/`.** Seven frozen mini-protocols stating complete executable rules would be newly published if the current branch merges. See risk register §P-1. | This is the single blocking item on the public surface. |
| 3 | **`reports/research/s34/`.** 816 files already public, containing figures today's policy would not publish. §P-3. | Deleting research receipts to tidy a front page is the wrong trade; the alternative is moving the corpus to a private remote. Both are owner calls. |
| 4 | **`SYSTEM_STATE.md`.** Already public, and would grow from 846 KB to 2.71 MB. §P-5. | Moving it private has a real operational cost. A genuine trade-off, not an oversight. |
| 5 | **The remaining 849 newly-added docs.** §P-6. | Needs a file-by-file pass through `docs/research/**` before the branch merges. |
| 6 | **`eclipse_scalper/localtests/`.** A nested directory sharing the repository's own name, holding 14,175 tracked files — 74% of the tracked tree, and the first thing a visitor sees when browsing the root. `.gitignore` already lists it; the files were tracked before the rule existed. | Untracking is a history-and-size decision with consequences for every existing clone. |
| 7 | **A tracked-link check in CI.** Eight module paths cited across the tracked public docs do not resolve, including inside two contract documents. Audit §T-11. | Adding a CI job is a change to `.github/workflows/`, outside this work's permitted change set. |

---

## 6. Applying the recommendations

Web interface: *Settings → General* for the description, website and social preview; the
gear icon beside **About** on the repository home page for topics.

Or, with the `gh` CLI — for the owner to run, not this work:

```bash
gh repo edit phoenixsenses/eclipse_scalper \
  --description "Mechanism-first market microstructure research and execution framework for perpetual futures — built around reproducibility, forward validation and risk-governed execution." \
  --add-topic market-microstructure \
  --add-topic quantitative-finance \
  --add-topic quantitative-research \
  --add-topic algorithmic-trading \
  --add-topic order-book \
  --add-topic execution \
  --add-topic risk-management \
  --add-topic reproducible-research \
  --add-topic experimental-design \
  --add-topic backtesting \
  --add-topic binance-futures \
  --add-topic python
```

The social preview cannot be set from the CLI — it is a manual upload.
