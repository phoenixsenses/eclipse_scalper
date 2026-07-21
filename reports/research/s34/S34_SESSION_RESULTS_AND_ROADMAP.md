# S34 Session Results & Ripple Roadmap — for Codex

State as of 2026-06-29. Hand to Codex to continue. Live executor / `execution/` /
`risk/` are UNTOUCHED and must stay that way unless the operator explicitly
promotes (see the shadow protocol). Methodology guardrails at the bottom are
non-negotiable — they are how the old evidence got contaminated.

---

## 1. What is settled (dead vs real)

DEAD (do not re-mine; settled):
- Continuation at the threshold cross — 12 routes RESEARCH_ONLY/BLOCKED, 0 PAPER_CANDIDATE.
- Conditional continuation (13 knowable features) — no cal+hold-positive subset.
- Fade at scalp horizons (<=1h) — the move is spent before the knowable cross.
- Onset / early-build velocity precursor — move happens faster than it can be confirmed.
- Order-flow (OFI) lead — direction RIGHT (gross ~+1-2bps, 52-58% win) but < taker cost.
- PRECURSOR, at every measurable resolution — own-flow, seed (butterfly), order-flow magnitude, and cross-asset lead-lag (1min AND 5s are contemporaneous, corr ~0.84 @ lag 0, no lead). The move is simultaneous across assets and spent by confirmation. The precursor does not exist tradeably.
- Tail-cut by selection — price-stop (whipsaw), resumption (universal, no specificity), butterfly own-seed (no separation), reclaim-stop (slow winners look like runaways), confirmation-entry (waiting forfeits the snap-back). The -410 tail is IRREDUCIBLE by entry/early selection.
- Regime gate (day-trend / BTC-trend) — looked positive in-sample, FAILS the chronological holdout (multiple-comparison artifact).

REAL (the standing lead):
- Big liquidation cascades mean-revert at swing scale (1h-24h), beta-controlled. Deep-V overshoot (>=28bps) sharpens it.
- The BRIDGE (user's idea) recovered Feb-Mar from mark data (book_ticker only covers Apr+Jun) -> 4 months / 169 events. Deep-V SELL fade: 4-month cal +10.4 / hold +6.6 mean, 3/4 months positive, cost-robust to ~14bps RT. Real-fill OOS (June): median +16, win 56%, sequential single-capital +628.
- SYNCHRONIZATION GATE (user's connection vision, validated): the outcome depends on the cascade's CROSS-ASSET connection. SYNCHRONIZED cascades (BTC+SOL also sell-liquidating, concurrent >=200K in prior 10min): N=97, mean +16/trade, win 60%, T3R +559, cal+hold positive. IDIOSYNCRATIC (isolated ETH flush): dead (mean +0.1, T3R -906). Best knowable filter; CONCENTRATES the edge but does NOT cut the -410 tail.

LIMITS on the lead: single ETH-SELL lane (does NOT generalize to BTC; SOL tiny-N); ~2 months real book + bridged modeled spread for Feb-Mar; the -410 tail is irreducible; sync gate found on the same data (mechanistically motivated, lower overfit risk, but unconfirmed forward / real-fill).

## 2. Standing lead & shadow status

Frozen, OBSERVATION-ONLY: `docs/protocols/S34_ETH_SELL_DEEP_V_FADE_V0_1.md` (v0.2 = + synchronization gate). NOT live. Promotion = operator decision after forward OOS across >=2 regimes + real-fill confirmation + sizing for the -410 tail.

## 3. Ripple / Wave-Dynamics roadmap (glossary -> testable question -> run)

The user's connection vision already produced the synchronization gate (the best filter). The wave-dynamics glossary maps to the remaining grounded questions:

| Glossary term | Question | Concrete run |
| --- | --- | --- |
| Cascade Effect / Chain Reaction / Resonance | Is there a CRITICAL synchronization (tipping point) where the edge peaks? Does 3-asset sync (BTC+ETH+SOL) > 2-asset? | sweep sync threshold and asset-count; find where mean/win/T3R peaks (phase transition) |
| Constructive Interference | Does the bounce AMPLIFY with the number of resonating assets? | bounce magnitude vs sync degree (sizing signal) |
| Destructive Interference | Does the fade FAIL when the ETH cascade opposes BTC's direction (counter-wave)? | condition fade on BTC concurrent direction; SELL-cascade while BTC rising = destructive? |
| Wave Absorption | Does order-book ABSORPTION predict revert vs runaway (the tail-separator we still lack)? Deep book absorbs the shock -> reverts; liquidity vacuum -> runs. | book depth / spread / book_imbalance at the cross -> outcome; this is the one tail-separator NOT yet tested |
| Attenuation / Decay / Ripple Decay Rate / Energy Dissipation | Does a faster-decaying shock revert more cleanly? Measure the move's half-life. | post-cascade price half-life / decay rate -> outcome |
| Positive Feedback / Amplification | Does post-entry liquidation ACCELERATION predict the tail magnitude (for sizing, since the tail is irreducible)? | liq_accel (already separates: winners 0.0 / runaways 0.2) -> tail-aware position sizing |
| Echo Effect / Wave Reflection | After the bounce, is there a secondary re-test (reflected wave) -> better exit / re-entry timing? | post-exit path: does price re-test the low? |
| Negative Feedback | (this IS the fade thesis — the system self-corrects; mean-reversion = negative feedback). Confirmed at swing scale. | n/a — it's the edge |
| Momentum Propagation / Shockwave | (settled — contemporaneous co-movement, no propagation delay at tradeable scale). | n/a — closed |

## 4. Prioritized next runs

1. **Wave Absorption (highest leverage):** book depth / spread / book_imbalance at the cross as a tail-separator. This is the one mechanism that could cut the -410 tail (a liquidity vacuum = runaway), and it is the ONLY separator class not yet tested. New tool.
2. **Real-fill + cross-asset sync gate:** does the sync gate survive real bid/ask fills (Apr+Jun), and does it RESCUE BTC (BTC was dead ungated — fade BTC only when ETH+SOL also cascading)? Reuse `research_s34_reversal_backtest.py` + a sync filter.
3. **Tipping point / dose-response (Resonance):** sweep sync threshold + asset-count; find the critical synchronization.
4. **Feedback-based sizing (Positive Feedback):** since the tail is irreducible, use liq_accel for tail-aware sizing rather than selection.
5. **Forward shadow:** keep accumulating true OOS on the sync-gated ETH-SELL fade.

## 5. Tools built this session (reuse)

`research_s34_{horizon_decay, onset_precursor, early_build_entry, orderflow_lead, liq_swing_event, reversal_backtest, reversal_regime_diag, reversal_stop_backtest, vshape_conditioning, tail_cut, butterfly_seed, confirmation_entry, bridge_backtest, failure_geometry, regime_gate, regime_gate_validate, cross_asset_leadlag, subminute_leadlag, convexity_flip, synchronization_gate}.py` plus the navigation dashboard/survey. Reports in `reports/research/s34/`.

## 5b. Absorption findings (2026-06-29, Codex) + v2 expansion

Wave Absorption test (`tools/research_s34_maker_absorption_permission.py`) on the
ETH-SELL 200K maker-LONG route: `deep_bid` (cascade hits a deep resting bid =
absorbed) is the strongest gate -- deep_bid N=11 sum +1081 med +46 T3R +402
max_loss +13 (NO losing trade); shallow_bid/vacuum carry the tail (T3R -126,
max_loss -144). deep_bid + state_machine: N=11 med +165 max_loss +6. Mechanism
CONFIRMED: absorbed shock reflects (revert); unabsorbed shock propagates (runaway).

**CRITICAL CAVEAT: N=11.** No holdout; book covers only ~Apr+Jun (likely
June-regime concentration); "no losing trade on 11" is small-sample selection, not
a validated tail-cut (the -410 tail was irreducible over 169 events). The
deep_bid+state_machine +165 median is NOT believable yet -- it is a hypothesis at
N=11 with multi-parameter overfit risk. Codex correctly froze it SHADOW_ONLY
(`S34_ABSORPTION_PERMISSION_V0_1`, deep_bid @ T=0 + state_machine). NOT live.

Unified model emerging: revert = ABSORBED reflection (deep bid); runaway =
UNABSORBED propagation amplified by RESONANCE (vacuum + synchronized cross-asset
deleveraging). Absorption (Wave Absorption) and synchronization (Resonance) are
two distinct mechanisms that may combine.

v2 next runs (the central problem is STATISTICAL POWER -- N=11 must become N>=50):
1. **Cross-asset pooling of the absorption signal:** BTC+SOL deep-V SELL with book -> many more deep_bid events. The only way to get N for a real holdout. Highest priority.
2. **Continuous absorption signal:** regress net_bps on book_imbalance / bid_qty / bid-depth across ALL 22 book-covered events (not an 11/11 binary split) -> more power + a sizing curve, less overfit than buckets.
3. **Absorption x Synchronization 2x2:** are deep_bid and synchronized redundant or complementary? Does book depth SUBSUME the sync proxy (the book IS the connection)? Test (deep_bid & sync) vs each alone.
4. **Energy Dissipation / replenishment (glossary):** static bid depth is absorption; the SPEED of bid replenishment after the cascade is dissipation -> does fast replenishment predict a cleaner bounce?
5. **Wave Reflection / Echo:** after an absorbed bounce, does price re-test the low (a reflected wave)? -> exit / re-entry timing.
6. **Forward shadow** the deep_bid + state_machine rule; the book-coverage limit means forward accumulation is the real validator.

Note: absorption REQUIRES book data, which is the binding constraint (~2 months).
The signal cannot be back-extended to Feb-Mar (no book). So forward shadow +
cross-asset pooling are the only ways to reach a trustworthy N.

## 5c. v2 pooling result + v3 Network roadmap (2026-06-29, Codex)

v2 cross-asset absorption pooling (N=541) DISPROVED the global deep_bid rule:
pooled sum -2615, T3R -4497; "absorbed" pooled is BAD (-7315). The ETH deep_bid
result was route-specific / small-N, NOT a global law (the N=11 caveat held).
NEW signal that DOES survive cross-asset: shock ENERGY, not static structure --
running_accel_z high quartile T3R +1481, running_notional_z +1112, while static
bid_depth_z is -6896. Edge lives at CONFLUENCES (sync+mixed holdout +1386/T3R +2.9;
sync+absorbed -7356), not single gates. Conclusion: route-specific state model >
global pooled gate.

### DISCIPLINE WARNING (read first)
We are now at the dangerous stage: many routes x many conditions = a multiple-
comparison OVERFIT FACTORY. At this scale you will ALWAYS find some N=11-14 route
that looks spectacular in-sample. Every candidate MUST clear: (a) chronological
holdout, (b) cross-route / cross-regime persistence, (c) a minimum N (>=40 filled
on each split), and (d) forward shadow. Treat any single-route N<40 result as a
hypothesis, never an edge. The binding constraint is still N / data / book
coverage, not model sophistication.

### Network roadmap (user's Connection/Network glossary -> run)

| Glossary term | Question | Run |
| --- | --- | --- |
| Critical Nodes / Strong vs Weak Links | Which routes are reliable NODES (strong edge) vs dead? Build the route map. | score every (symbol x threshold x depth) route: holdout expectancy + tail + N; rank the strong nodes |
| Hidden Connections / Latent Variables | Is shock ENERGY (running_accel_z, running_notional_z) the universal latent variable across routes (where static book failed)? | route-normalized continuous model on energy features; test cross-route persistence |
| Confluence Zones / Intersection Points | The edge is at intersections (sync+mixed). Which 2-/3-way confluences hold OUT-OF-SAMPLE? | systematic confluence search WITH holdout + cross-route guard (beware combinatorial overfit) |
| Path Dependency / Dependency Chains | Dynamic > static. Does the cascade's PATH (acceleration, dissipation) determine the bounce? | energy dissipation: post-entry bid replenishment + liq deceleration speed -> outcome |
| Influence Network / Network Effects | Does one route's state PREDICT another's outcome (route->route influence)? | cross-route conditioning: does a concurrent BTC cascade state predict the ETH fade outcome (beyond price co-move)? |
| Butterfly Effect | Small initial ENERGY differences -> divergent outcomes (seed didn't separate, but energy/accel does). | accel/notional at the cross as the divergence seed (refines the failed butterfly_seed) |
| Blind Spots | Where does measurement fail (no-book periods, regime concentration)? Don't trust those. | flag and exclude no-book / thin-regime windows from every verdict |
| Leverage Points | Per strong node, the single condition that most improves expectancy. | route-specific one-condition leverage scan (holdout-gated) |

### v3 prioritized next runs
1. **Route node map (Critical Nodes):** rank all routes by holdout expectancy + tail + N. Keep strong nodes SEPARATE (no pooled rule). ETH-SELL standing route stays its own frozen candidate.
2. **Energy latent model (Hidden Connections):** route-normalized running_accel_z / running_notional_z continuous model -- is shock energy the cross-route universal? This is the most promising NEW lead from v2.
3. **Energy dissipation (Path Dependency):** post-entry bid replenishment / liq deceleration -> tail separator (the dynamic version of absorption).
4. Independent frozen SHADOW candidates per strong node (ETH, and any BTC/SOL node that clears holdout + N>=40), never a single global rule.

## 5d. v3 result + v4 Management pivot (2026-06-29, Codex)

v3 with full discipline (holdout, N>=40 each split, cross-route): ROUTE NODE MAP =
**0 STRONG_NODE** -- nothing passes; the overfit-factory warning held; the small-N
"winners" (SOL 2/12, BTC 4/8) are invalid. There is NO validated ENTRY edge.
Energy latent = weak but CONSISTENT (Pearson ~0.10 cal AND hold); not a standalone
gate; best as a CONFLUENCE context/sizing signal (energy_high + not_absorbed: hold
N=42 sum +1663 T3R +610 -- first N>=40 holdout-positive-with-positive-T3R, but tail
-460 remains). Energy DISSIPATION is the new lead but is POST-ENTRY (management,
not entry): total_replenish_120s_pct q90 hold-high N=31 sum +2268 med +65 tail=1
vs hold-low N=344 sum -6636. Fast post-entry book replenishment -> the bounce
holds; slow -> runaway.

### THE PIVOT: entry -> management
We cannot PREDICT at entry which cascade reverts (butterfly/seed/absorption all
route-specific/small-N). But we can OBSERVE at +60-120s whether the absorbed
reflection is materializing (book refilling, liquidations decelerating) and CUT
the ones turning into propagation (-460 runaways) before they hurt. The edge is in
DETECTING the failure fast, not predicting the success. This converts the
irreducible entry-tail into a manageable exit-tail.

### v4 prioritized next runs (management, not entry)
1. **Dissipation management BACKTEST (core):** enter all deep-V SELL fades; at +120s measure total_replenish_120s_pct + liq_deceleration; EXIT EARLY / tighten the low-replenish ones; hold the high-replenish ones to 4h. Does this management overlay improve REALIZED total P&L and cut the tail vs baseline hold-4h, on a chronological holdout? (Descriptive q90 split is NOT enough -- backtest the actual rule's P&L impact.)
2. **Observation horizon sweep:** is +120s optimal, or +60/+90/+180s? When does the replenishment/deceleration signal become reliable enough to act without whipsawing the slow-but-real reverters?
3. **Dual confirmation:** book replenishment AND liq deceleration together as the "bounce confirmed" signal -> exit if NEITHER materializes by tau.
4. **Energy confluence as SIZING (not a gate):** energy_high + not_absorbed as a position-size multiplier; test sized P&L / Sharpe, not a binary filter.
5. Write the dissipation observer SHADOW-ONLY (Codex's plan): at +120s after a logged signal, simulate hold / tighten / exit-early. NOT live.

Honest meta: no validated entry alpha survives discipline; the remaining value is
MANAGEMENT (dissipation-based exit) and CONTEXT/SIZING (energy confluence), both
promising but small-N and forward-unvalidated. Binding constraint still N / data /
book coverage. Do NOT let route x condition combinatorics manufacture a false edge.

## 5e. v4 result + CONVERGENT VERDICT (2026-06-29, Codex)

v4 dissipation management backtest (`tools/research_s34_v4_dissipation_management.py`,
full discipline): pooled baseline holdout is NEGATIVE (-4367). Management (+120s
replenishment + liq deceleration) CUTS THE TAIL hard (<-100: 90->15; <-200: 62->8)
but does NOT validate as an edge: primary rule breaks calibration (+1740->-1132);
the best-holdout config (+2263) is overfit (cal -1134, "cal bad/hold good" regime-
fit); the only 3 cal+hold-both-positive configs leave the managed holdout still
NEGATIVE (cleanest: tau180_replenish_only -2343). On the live ETH lane (N=16)
management HURTS (+825 -> +388). Codex's call (shadow observer only, no order
change) is correct.

### CONVERGENT VERDICT
After ~25 disciplined tools across entry, conditioning, fade/maker/convexity,
tail-cut, butterfly, confirmation, bridge, cross-asset, regime gates, lead-lag,
synchronization, absorption, energy, and dissipation management: there is NO
validated, robust, out-of-sample edge in the S34 liquidation-cascade family --
neither ENTRY nor MANAGEMENT. The recurring pattern is mechanistically-plausible
signals that look great in-sample / small-N and FAIL proper holdout + cross-route +
N>=40 discipline. The tail is cuttable (management) but cutting the tail on a
zero/negative-edge base does not create an edge. This family is EXHAUSTED under
discipline. Treat this as settled, hard-won knowledge -- not a prompt for more
condition combinatorics (that is the overfit factory).

### What the program produced (value)
Rigorous no-lookahead research infrastructure (~25 tools); the navigation/permission
dashboard; observation-only shadow protocols; and a definitive map of what does NOT
work and WHY -- which prevents burning real capital.

### Genuine next steps (only these)
1. Forward-shadow the existing observers (dissipation observer + ETH-SELL lane) to
   accumulate TRUE out-of-sample. Codex's shadow-observer plan is endorsed. NOT live.
2. If continuing active RESEARCH: pivot to a genuinely DIFFERENT signal class
   (funding-rate momentum / basis convergence / OI dynamics) -- fresh data, not
   overfit on cascades. Do NOT mine more cascade conditions.
3. Consolidate / stop. The cascade research is complete.

## 5f. Funding lead (2026-06-29, Codex) — develop with discipline

Fresh signal class (funding-rate mean-reversion). First result LOOKS strong and
is much larger-N than cascades: best config funding_abs_z>=1 MR 24h -- cal N=3335
sum +70359 med +13.4; hold N=926 sum +48463 med +54.9; both sides (pos funding ->
SHORT, neg -> LONG) holdout positive. Symbol split: ETH hold +44197, BTC +14281,
SOL **-10016 (broken)**. Tools: `tools/research_funding_extreme_mean_reversion.py`.

**#1 RED FLAG (must resolve before believing it): OVERLAPPING-WINDOW INFLATION.**
"every funding snapshot with |z|>=1 = a trade" means consecutive snapshots are
highly autocorrelated (funding barely moves over hours) and their 24h forward
returns massively overlap. So N=3335 is NOT 3335 independent trades -- it is maybe
a few hundred independent funding regimes oversampled thousands of times. The
+48k sum / +54.9 median are inflated by summing overlapping correlated windows;
+54.9 bps/trade is mechanistically implausible for a pure funding signal.

Develop (in order):
1. **Non-overlapping trades (make-or-break):** one trade per funding period, hold to next funding / 8h, no re-entry while open. Does the edge survive at TRUE independent N (~hundreds)? Until this passes, do not believe the headline.
2. **Pure funding vs directional beta:** "short when funding positive" may = "short in bull regimes." Remove market beta (regress out market return / use BTC-SOL difference) -- does a funding-specific edge remain, or is it a regime bet?
3. **Horizon match:** funding is 8h; a 24h forward mixes 3 funding periods. Match 8h funding -> 8h forward to avoid contamination.
4. **Cost realism:** funding trades often -> cumulative fee+spread matters; net per-trade after cost.
5. **Liquidation independence:** does it work in calm (no-cascade) periods, or only in liquidation regimes?
6. **SOL failure / walk-forward:** SOL is the canary (BTC was for cascades); is ETH/BTC robust month-by-month or regime-concentrated?

Same discipline as the cascade family: chronological holdout, real cost, cross-asset
persistence, and TRUE independent N. The non-overlapping test is the first gate.

Live note: the single armed executor (PID, rule ...DEEPBID, active=null) trades the
UNVALIDATED cascade rule if a signal fires -- keep it minimum-size + kill-criteria;
funding research stays RESEARCH (do not wire to live).

## 5g. Funding non-overlap result + v5 META questions (2026-06-29)

Non-overlapping funding test (`tools/research_funding_nonoverlap.py`) DEMOLISHED
the +48k headline (it was overlapping-window inflation): calibration negative
everywhere (BTC -2802, ETH -491, SOL -179), holdout small/inconsistent, neither
SHORT nor LONG cleanly positive on both splits, only small-N positives. So the
FRESH signal class (funding) ALSO fails discipline -- the meta-pattern now holds
across two independent signal families. Strong inference: this dataset (~2-4mo,
ETH/BTC/SOL, available features) contains NO simple robust validated edge that
clears cost. In-sample edge-hunting manufactures artifacts; stop it.

v5 next questions are META (do NOT mine another in-sample well -- it will produce
another artifact):
1. **Permutation-null (highest value):** for every candidate (cascade fade, sync, absorption, funding), shuffle labels (direction/time) 1000x; is the real result beyond the 95th percentile of the null? If NOTHING beats permutation -> definitive proof the "edges" are noise -> stop hunting. If something does -> it is real and worth pursuing. This answers "keep researching or stop?" with data.
2. **Artifact-detector harness:** codify the manual discipline into ONE automated gauntlet -- any candidate signal -> {non-overlap, holdout, real cost, cross-asset, beta-control, N>=40, permutation-null} -> PASS/ARTIFACT. Reusable; prevents future artifact-chasing.
3. **Data SNR ceiling:** from realized vol + noise floor, the max achievable Sharpe at retail latency/cost -- is the data simply efficient for us, or is there headroom?
4. **Data-expansion scoping:** backfill funding/OI/basis to 12+ months from exchange APIs -- which backfill gives the most regime diversity per unit effort? (The binding constraint is data/N, not models.)
5. **Forward-shadow ledger:** a proper real-time hypothetical-P&L log per candidate so true OOS accumulates; estimate the minimum forward weeks to separate edge from noise.

Live: operator has decided the single armed executor keeps running on the
unvalidated cascade rule. Honest counsel given (no validated edge anywhere; keep
minimal / kill-criteria). Do NOT touch it; do NOT wire research to it.

## 5h. v6 — Live-rule MANAGEMENT system (2026-06-29)

Pivot: not "find new alpha" (permutation-null settled: none) but "MANAGE the live
rule that is running." The permutation-null killed PREDICTION, not RISK MANAGEMENT.
Glossary categories are reframed from new-alpha lenses to RISK/FAILURE-MODE/
EXECUTION lenses. HARD CONSTRAINT: management is NOT entry-filtering -- the live-rule
profile showed the -410 losers are indistinguishable from winners at entry (sync,
depth, BTC all identical). So you cannot "skip the bad trades"; the only defenses
are sizing, defensive exit, monitoring, and kill criteria.

Management system components (build these; observation/risk only, no order-logic
changes without operator sign-off):
1. **Tail-aware position sizing:** size so the worst observed -410 (and a margin)
   is survivable -- risk-of-ruin / fractional-Kelly bounded. This is the #1 defense
   because the tail is unpredictable and irreducible.
2. **Defensive-exit observer (shadow-only):** the dissipation observer -- at +60/120/180s
   measure book replenishment + liq deceleration; LOG a hold/tighten/exit recommendation.
   Note: it did NOT validate as a P&L improver, so it is a DEFENSIVE/diagnostic observer,
   not an edge; do not let it change live orders without forward proof.
3. **Regime-degradation monitor:** the profile shows the live (June) regime is the WEAKEST
   (median +12 vs +40-51 earlier). Alert when rolling win-rate / median degrades vs the
   historical baseline -> scale down / pause trigger.
4. **Failure-mode classifier (Traps glossary, DESCRIPTIVE):** classify each -100+ loss by type
   (stop-hunt / whipsaw / exhaustion-continuation / market-wide deleveraging). Informs sizing
   and the kill threshold; NOT an entry filter (they are unpredictable).
5. **Explicit kill criteria:** forward 30/60-day live sum < 0, or rolling tail exceeds the
   pre-accepted risk budget, or regime-degradation monitor trips -> reduce/stop. No size
   increase until forward OOS positive across >=2 regimes.

Legitimate 600GB use (EXECUTION, not alpha): tick-level fill/queue realism for the maker
execution (extend `research_eth_provision_realism.py`). The 600GB is VOLUME not calendar --
it improves execution/fill modeling, it does NOT back-extend the calendar or create
directional alpha (that is settled).

Do NOT: mine M/W-pattern / network / system-dynamics for new DIRECTIONAL alpha (overfit
factory; permutation-null is definitive). Do NOT build an entry filter from the trap features.

## 5i. v7 — Execution-management çevre completion (observation/risk/recommendation only; keep 40x)

Inputs (from the protective-stop sweep + execution-management audit):
- Least-destructive stop = `fixed_sl_150` (150 bps). Worst REAL taker fill -175.7 bps (gaps through the nominal 150).
- At the live $1190 notional, the 150bps stop loss ~= $20.9 = **59.7% of the $35 equity**. At the $11 tail-budget notional ~= $0.2 = 0.6%. The stop bounds the tail but does NOT make the current size safe.
- Bracket is NOT atomic: entry limit fills, stop is placed on the next 2s poll loop -> a short but real unprotected window.
- Tail frequency 18.7%; P(>=1 tail) = 46% / 65% / 87% over 3 / 5 / 10 trades.
- Leverage stays 40x (operator decision; revisit last).

Tasks for Codex (ALL observation/risk/recommendation; NO live order-logic/size/config change without explicit operator sign-off; do not touch `.env`, `tools/s34_v_engine_live_executor.py`, live state):
1. **Sizing recommender (#1):** from equity + tail-budget + the stop's worst real fill (-175.7), output recommended max margin/notional per trade; flag current env size vs recommendation (oversize multiple). Keep 40x leverage; vary MARGIN only.
2. **Stop recommender + gap-through flag:** confirm 150bps as least-destructive; quantify protection at each candidate size; flag that worst real fill (-175.7) > nominal (150) => stop protection is PARTIAL (gap-through), not exact.
3. **Atomicity-gap monitor (observation):** log the entry-fill -> stop-placement window per forward/paper signal; alert if an adverse move begins within it. DOCUMENT the gap and RECOMMEND making entry+stop atomic (or placing the stop with the entry) -- but that is a live-logic change requiring operator sign-off; do NOT change it.
4. **Alerting layer:** wire oversize / tail-event / atomicity-gap / regime-degradation / kill-trip to the existing `tools/telegram_bot.py` + `tools/telemetry_*` -> operator notifications. Observation -> notify, no actions.
5. **Consolidated execution-management panel:** one readout -- live state, size vs tail-budget, stop config + real-fill protection, atomicity-gap status, tail-frequency context, kill status. Host: `tools/s34_cascade_navigation_dashboard.py`.
6. **Forward-validation decision-gate doc:** 30/60-day ledger review -> validate (operator may resize) or noise (disarm).

Honest frame: the stop helps but the real defense at current size is sizing; the atomicity gap is a genuine uncapped window. The çevre makes the risk VISIBLE and RECOMMENDED; only the operator changes live size/logic/leverage.

## 6. Guardrails (non-negotiable)

- No lookahead: only data with ts <= decision time; run feature sets through `tools/s34_feature_availability.py`.
- Chronological holdout; judge on the held-out later period.
- Median is a trap: always report TOTAL sum + skew + max_loss + T3R. This family wins often but loses big.
- Pay for fills (taker bid/ask + fee; maker modeled conservatively). Mark-to-mark is not tradeable; the bridge's modeled 2bps spread is optimistic.
- Beware regime/in-sample artifacts: a filter found by searching many candidates must clear a chronological holdout AND ideally cross-asset before it is believed.
- Machine limits: no parallel Python/PS (RAM); max 2 test files per pytest; D: drive; pytest `--basetemp=D:/...`; avoid full-table scans on the ~600GB DB (bound queries by symbol+time).
- DO NOT wire research into the live executor. Research plane is isolated (repo doctrine SAF-02). Promotion is an explicit operator decision.
