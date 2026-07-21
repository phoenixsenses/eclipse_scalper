# Liquidation-Driven Flow — Quote Suppression Policy Diagnostic

**Generated:** 2026-07-17 · **Symbol:** ETHUSDT · **RESEARCH_ONLY / READ-ONLY**
**Verdict:** `LIQUIDATION_QUOTE_SUPPRESSION_NO_INCREMENTAL_VALUE`
(secondary framing: any loss-reduction vs a naïve always-on maker is `REDUCES_LOSSES_ONLY` and is fully generic to high-volatility, not liquidation-specific)

Holdout NOT opened · deep-bid NOT simulated · no top-of-book-external fills assumed · directional 2h return NOT added to maker PnL · DB/repo/runtime/strategy read-only (`mode=ro`). This study does **not** touch, soften, or re-optimise the two preserved Phase-1 verdicts (`PRICE_MOVEMENT_WITHOUT_HARVESTABLE_ALPHA`, `SELL_BURST_MAKER_EDGE_NOT_IDENTIFIED`).

Machine artifacts: `LIQUIDATION_QUOTE_SUPPRESSION_POLICY_DIAGNOSTIC.json` (all numbers below regenerated deterministically; stage-1 fills sha256 `4b49aaf1d2b5f87b`).

---

## 0. Question

Not a directional strategy. A **maker risk-gating** question: during BUY or SELL liquidation-driven aggressive flow, does temporarily **withdrawing passive quotes** (P2–P5) improve inventory-adjusted maker PnL **relative to an always-two-sided maker (P1)** — and, critically, does it add value **beyond doing the same thing in any matched high-volatility non-liquidation window**?

## 1. Data reality & power gate (criterion #10 — PASS)

- `book_ticker` (identifiable top-of-book: `bid/ask price+qty`, `spread_pct`) covers **2026-04-11 → 2026-07-17** — the binding executable window. `liquidations`/`agg_trades`/`mark_prices` run 2026-02-15→07-17 but pre-book data is unusable for maker fills.
- SELL-liq collection hole **2026-04-27→06-07 (41d)** and **07-06→07-10 (4d)** confirmed and excluded from day counts.
- Frozen primary trigger = **first single liquidation print ≥ 100K USD, 5-min debounce, per direction**; `side='SELL'`(forced sell) attacks the **BID**, `side='BUY'`(forced buy) attacks the **ASK**. Overlapping ±5min windows merged into canonical clusters.

| Trigger | canonical clusters | distinct days | non-H17 days | ≥40-day power |
|---|---:|---:|---:|:--:|
| SELL 100K | 575 | 55 | 55 | ✅ |
| BUY 100K | 639 | 56 | 56 | ✅ |
| SELL/BUY 200K | 340 / 392 | 53 / 53 | 53 / 53 | ✅ |
| SELL/BUY 1M (diagnostic) | 68 / 101 | 36 / 39 | 35 / 39 | ⚠️ below 40 |

Analysis set (book present both event & matched-control): **1068 events + 1068 matched controls, 51 event-days.** HOUR17 flagged per-window (not a whole-day exclusion — H17 triggers co-occur with non-H17 triggers on the same days).

## 2. Method (top-of-book only)

Continuous two-sided maker, **own = 0.5 ETH**, joined at best, **re-quoted at best after each fill** (queue resets), inventory backstop 40 fills/side (=20 ETH). Fills are driven by real `agg_trades` against `book_ticker`:

- **F3 queue-conservative (primary):** fill when cumulative aggressive volume through our level exceeds **queue-ahead = visible top qty at placement** + own size. Levels below best are NOT identifiable → not attempted.
- **F2 trade-through (secondary family):** fill on a trade strictly through our level.
- **F1 touch:** upper bound only — **excluded from all evidence.**

Per-fill maker PnL (bps of notional): `spread_capture = pos·(mid_fill − price)` + `adverse_markout = pos·(mid_{fill+h} − mid_fill)`; realistic **net_cross** unwinds by crossing the spread (taker) at `fill+h`; **net_marked** marks at mid. Policies suppress fills by side within `[latency, off_end)`; fills in `[0, latency)` are the cancel/fill race (not suppressed). Matched control = same clock-time on nearest ±day with **no liq ≥50K within ±5min** and book present (records pre-5min realized vol). Primary config: L=250 ms, off-dur=30 s, eval-window=30 s, markout h=30 s, maker 2 bps / taker 3.05 bps.

`net_suppression_value = Delta_Pk = Pk − P1 = gross_avoided_loss − missed_profitable_fills` (identity holds by construction).

---

## A. Policy P0–P5 comparison (per event, bps of 0.5-ETH notional, day-clustered CI90)

**F3 — primary evidence family**

| Policy | abs net_cross | Δ vs P1 [CI90] | matched-ctrl Δ | **incremental (ev−ctrl) [CI90]** |
|---|---:|---:|---:|---:|
| P0 no-quote¹ | −363.1 | +142.0 [131,154] | +125.2 | +11.7 [−2.5, +26.0] |
| **P1 always two-sided** | **−501.1** | 0 | 0 | 0 |
| P2 both-off | −363.1 | +142.0 [131,154] | +125.2 | **+11.7 [−2.5, +26.2]** |
| P3 attacked-off | −444.8 | +55.5 [47,65] | +63.2 | −4.5 [−15.8, +7.8] |
| P4 opposite-only | −444.8 | +55.5 [47,65] | +63.2 | −4.5 [−16.2, +7.3] |
| P5 flow-renorm | −363.1 | +142.0 [130,154] | 0² | +137.3 [127,147]² |

¹ Strict P0 ≡ no quotes at all ⇒ PnL 0 by definition; the row measures the pre-trigger+race residual (≡ P2 at dur=W). **Pk − P0 = abs net_cross, which is negative for every policy** — the least-loss absolute maker policy is *do nothing*.
² **Artifact:** controls have no trigger, so P5 re-enters immediately and suppresses nothing in controls (baseline 0). P5's apparent incremental is not liquidation alpha — on events P5 ≈ P2, and P2's incremental against the *fair* generic baseline is ≈0.

**F2 — secondary family (rejects harder):** P2 incremental **−78.9 [−98, −59]**, P3/P4 **−61.2 [−74, −48]** — under trade-through fills, liquidation-triggered suppression is *significantly worse* than the same suppression in matched high-vol windows.

## B. Avoided adverse-selection decomposition (F3, per event)

| Policy | gross avoided loss | − missed profitable fills | = net suppression value |
|---|---:|---:|---:|
| P2 | 173.9 | 36.0 | **+138.0** |
| P3 | 64.9 | 8.6 | +56.3 |

The gross avoided "loss" is real vs P1 — but see F/I: ~90% of it is avoided **round-trip fees on churned fills**, not avoided adverse selection.

## C. Missed-fill opportunity cost
Included in every net figure above (net_suppression_value is *after* subtracting missed profitable fills). At dur≥60 s the missed post-burst mean-reversion fills dominate → incremental goes **negative** (−50, §F).

## D. Cancel/fill race (§4)
Latency shifts which race-window fills leak into P1 on the attacked side; effect is small and monotone (§I latency row). No policy's conclusion depends on the 0-ms theoretical upper bound.

## E. Inventory-adjusted PnL
`net_cross` already crosses the spread + pays taker on unwind (inventory-adjusted). **Every policy's inventory-adjusted absolute PnL is negative** (criterion #4: value is not preserved — there is no positive value to preserve).

## F. Quote-off duration sweep (F3, W=90 s, incremental ev−ctrl)

| off-dur | ev Δ | ctrl Δ | **incremental [CI90]** |
|---:|---:|---:|---:|
| 1 s | 14.5 | 4.8 | +10.1 [7.1, 13.0] |
| 5 s | 61.4 | 21.0 | +35.4 [29.0, 41.8] |
| 30 s | 142.3 | 126.2 | +11.0 [−3.5, +25.6] |
| 60 s | 170.2 | 216.8 | **−50.5 [−70.5, −30.8]** |

Non-monotone: short off-windows show a fee-inflated positive, long off-windows destroy value.

## G. BUY/SELL symmetry (criterion #6 — FAIL for the incremental)
Δ vs P1 is directionally consistent (SELL P2 +149, BUY P2 +127), but the **incremental is not**: SELL P2 **+35.6**, BUY P2 **−2.05**. The small SELL-side positive is not mirrored on BUY ⇒ not a robust cross-directional mechanism.

## H. Matched non-event controls (criterion #5 — the decisive gate, FAIL)
Suppressing quotes in matched high-vol **non-liquidation** windows captures **~90% of the same benefit** (P2 ctrl Δ +125 vs event Δ +142). The liquidation-specific increment is +11.7 with **CI including 0**, and **max-stat FWER p = 0.69** (P3/P4 p = 0.96). Quote suppression derives its value from *high volatility*, not from *liquidation information*.

## I. Latency / fee sensitivity (F3 P2 incremental)

| latency | 0 ms | 50 ms | 100 ms | 250 ms | 500 ms |
|---|---:|---:|---:|---:|---:|
| incr | +14.3 | +13.4 | +13.0 | +11.7 | +8.9 |

| fee (maker/taker) | 0 / 0 | 2 / 3.05 | 3 / 4.05 |
|---|---:|---:|---:|
| incr | **−0.6** | +11.7 | +16.6 |

**The entire positive incremental is a fee/churn artifact — it scales with fees and vanishes at zero fees** (fee0 P2 incr −0.6 [−6.9, +5.6]; the best-case dur=5s at fee0 = +4.4 [−0.9, +9.9], CI includes 0). At zero fees all six policies cluster **−125…−140** bps/event (best/worst gap collapses from ~173 to ~15). Criterion #7 fails: at realistic 250 ms the incremental CI already includes 0, and at fee0 it is null regardless of latency.

## Not-just-lower-uptime (criterion #9 — FAIL)
Per-fill net_cross of **suppressed** burst fills = **−5.5** vs **kept** fills **−7.7** (F3, P2). Suppression removes fills that are *less* adverse than average — it is **not** targeting the worst fills; the benefit is reduced churn/uptime, not smarter selection.

## J. Deep-bid evidence boundary
`DEEP_BID_FILL_ASSUMPTIONS_NOT_VALIDATABLE_FROM_TOP_OF_BOOK`. This study used only identifiable best-of-book. The existing `S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_*_DEEPBID` protocol places maker fills **below best** — that fill level sits in the NOT_IDENTIFIABLE zone; top-of-book data can neither confirm nor deny its fills. Required to validate: L2/order-book depth, depth update sequence, queue additions/cancellations, own queue-position reconstruction. Absent that data, its fill claims remain **UNVERIFIED**. This work neither accepts nor rejects deep-bid.

## K. Phase-2 candidate
**None.** No quote-suppression policy is a Phase-2 candidate.

---

## Success-criteria scorecard (§10)

| # | Criterion | Result |
|---|---|:--:|
| 1 | F3 Pk − P1 > 0 | ✅ (P2 +142) |
| 2 | Day-clustered CI lb ≥ 0 (for the *incremental*, the meaningful quantity) | ❌ incl 0 |
| 3 | Missed-fill opportunity cost included | ✅ |
| 4 | Value preserved after inventory unwind | ❌ (all abs negative) |
| 5 | **Incremental over matched high-vol baseline** | ❌ **decisive** (+11.7 CI[−2.5,26]; fee0 −0.6; FWER 0.69) |
| 6 | BUY & SELL mechanically consistent (incremental) | ❌ (SELL +35.6, BUY −2.0) |
| 7 | Survives latency < 250 ms | ❌ (CI incl 0; fee0 null) |
| 8 | Survives multiple-testing | ❌ (max-stat FWER p 0.69/0.96) |
| 9 | Not merely lower quote uptime | ❌ (suppressed fills less adverse) |
| 10 | ≥ 40 independent non-burned days | ✅ (51 event-days) |

## Verdict

`LIQUIDATION_QUOTE_SUPPRESSION_NO_INCREMENTAL_VALUE`

Making continuously into these windows (P1) loses heavily, so any quote-off policy *looks* good against that strawman (`REDUCES_LOSSES_ONLY`). But the loss-reduction (a) is ~90% avoided churn fees, not avoided adverse selection; (b) is fully reproduced by suppressing in **matched high-volatility non-liquidation** windows; (c) is not direction-consistent; (d) does not survive latency, zero-fee, or multiple-testing controls; (e) does not target the worst fills. The **liquidation trigger carries no incremental maker risk-gating value** over a generic volatility gate. This is the risk-gating mirror of the preserved `SELL_BURST_MAKER_EDGE_NOT_IDENTIFIED` finding: "don't make into aggressor flow" is generic, not liquidation-specific.

**Falsification / what would reopen this:** L2 depth data enabling below-best queue reconstruction (would test deep-bid and a genuine queue-position edge), or a matched-control design showing event-minus-control incremental with CI excluding 0 at zero fees and consistent across BUY/SELL on an untouched holdout. Until then: RESEARCH_ONLY, no Phase-2, operator gate stands.

## Limitations (honest)
- Single-shot-per-side generalised to repeated fills with a 40-fill/side inventory cap; cap applied identically to events and controls (delta-neutral).
- P5 re-entry uses a past-only "liquidation silence ≥60 s" proxy (+spread-band via pre-event point samples), not the full 4-condition microprice/imbalance band — a documented simplification; P5's headline is dominated by the control-baseline artifact regardless.
- Eval fills detected within +90 s (bursts are seconds-scale); +5 m priced by terminal point-lookup. 5-minute-window fills beyond +90 s not enumerated (adverse selection is a +seconds phenomenon; 30/60/90 s carry the result).
- Matched controls preserve UTC hour + time-of-day + closest pre-5min vol; not jointly matched on all §7 axes simultaneously.

---

## Addendum — Accounting sanity re-audit (2026-07-17, independent, read-only)

Re-derived independently from the **frozen stage-1 fills** (sha256 `4b49aaf1d2b5f87b…`, byte-identical to the file that produced this report) and the stage-2 per-window PnL. No DB access, no re-simulation. Purpose: audit the `−501 bps/event` P1 headline for unit / double-count error before treating the liquidation maker family as closed.

**Headline reproduced exactly:** P1 net_cross = **−501.05**, P2/P0/P5 = −363.09, P3/P4 = −444.78 (events = 1068, controls = 1068). All identities hold.

**Decomposition of the −501 (F3, per-event mean):**

| Component | value | note |
|---|---:|---|
| mean fills/event (P1) | **72.16** | of a hard cap of 80 (40/side) |
| events at n_fill ≥ 40 | 984 / 1068 | inventory cap binds for most events |
| events at n_fill ≥ 79 | 775 / 1068 | **both sides saturated** |
| fee burden (= 5.05 × fills) | **−364.4** | **72.7 % of the −501** |
| spread capture | +10.2 | maker earns thin half-spread |
| adverse markout | −144.9 | true adverse selection |
| net_marked (mid, no taker) | −279.0 | |
| cross+taker on unwind | −222.1 | each of ~72 lots unwound independently |

**Identity & double-count checks (all PASS):** `P2.kept + P2.suppressed == P1.n_fill` (0 violations); `delta_cross == avoided_loss − missed_profit` (max residual 0.001); no fill counted twice; n_fill capped at 80 exactly.

**Independent fee=0 reconstruction** (add back 5.05 × n_fill per policy): all policies collapse to **−125…−137** (event) / −3…−15 (control); **P2 incremental @fee0 = −0.03** (report −0.6); P2 incremental @fee = +10.1 (report +11.7). The report's decisive claim — *the entire positive incremental is a fee/churn artifact and is null/negative at zero fees* — is **independently reproduced from the frozen fills.**

**Verdict of the re-audit — NO verdict-changing error.** No unit or double-count bug; every identity holds and the fee0/incremental conclusion reproduces exactly. The liquidation maker family stays **CLOSED**.

**One honest labelling caveat (does not reopen the family):** the "−501 **bps/event**" is a *sum of ~72 independent 0.5-ETH round-trip returns under a saturated inventory cap*, each normalised by its own lot notional — **not** a per-notional return. The turnover-normalised figure is ≈ **−6.9 bps per 0.5-ETH round trip**. Two design choices inflate the absolute magnitude — (a) MAXFILLS saturation (73 % of events cap out), (b) per-lot independent taker unwind (pays taker ~72× instead of one netted inventory exit). Both apply **symmetrically to events and controls**, so the incremental analysis (the load-bearing result) is unaffected; only the absolute strawman magnitude of P1 is exaggerated. Recorded for hygiene; the preserved verdicts `PRICE_MOVEMENT_WITHOUT_HARVESTABLE_ALPHA`, `SELL_BURST_MAKER_EDGE_NOT_IDENTIFIED`, `LIQUIDATION_QUOTE_SUPPRESSION_NO_INCREMENTAL_VALUE`, `DEEP_BID_FILL_ASSUMPTIONS_NOT_VALIDATABLE_FROM_TOP_OF_BOOK` are untouched.
