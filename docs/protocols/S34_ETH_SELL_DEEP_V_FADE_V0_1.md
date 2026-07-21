# S34 ETH-SELL Deep-V Fade v0.1 — Forward Shadow Protocol

Status: `EXPLORATORY_FROZEN — OBSERVATION ONLY`

Protocol id: `S34_ETH_SELL_DEEP_V_FADE_V0_1`

Created: 2026-06-29

## Authorization

This protocol **DOES NOT authorize live execution**. It freezes a research
candidate for forward, observation-only shadowing so that a true out-of-sample
sample accumulates from zero. Promotion to the live executor is an explicit
operator decision, gated by the kill criteria below AND by manual review — it is
never performed automatically from research. No `execution/`, `risk/`, or
`brain/` file is touched by this protocol.

## Frozen Rule

| Field | Value |
| --- | --- |
| Symbol | ETHUSDT |
| Trigger | SELL liquidation cascade, running-notional threshold cross at 200K |
| Filter | deep-V overshoot >= 28 bps (mark move from cascade start to cross) |
| Direction | FADE = LONG (bet the down-spike reverts) |
| Entry | at the threshold cross (no confirmation wait — the edge is in the snap-back) |
| Exit | fixed 4h from cross |
| Fills | taker, cross the spread (book_ticker bid/ask) |
| Cost assumed | ~6.1 bps fee RT + real spread |

## Honest Evidence (and its limits)

Bridged mark-based backtest (Feb-Jun, modeled 2bps spread):
- 4-month FADE positive in 3/4 months; chronological holdout cal +10.4 / hold +6.6 mean (win 56.8% / 52.9%); cost-robust to ~14 bps RT.

Real-fill (book bid/ask, Apr+Jun only):
- ETHUSDT: holdout (June) median +16.2, win 55.6%, sequential single-capital sum +628 (~+10.5 bps/trade). Calibration (April) flat (med -0.6).

**Red flags — why this is NOT yet tradeable:**
- Cross-asset generalization FAILS as a clean edge: ETH-SELL hold +16, but BTC hold med -3 (does not replicate), SOL hold +77 but on ~16 trades (small-N). Across all 3 assets calibration (April) is weak/negative and the positive P&L concentrates in the June holdout => the apparent edge is largely a **June regime effect**, not a durable cross-asset alpha.
- The runaway tail (~-410 bps) is irreducible: 5 selection methods failed to cut it (price-stop, resumption, butterfly-seed, reclaim-stop, confirmation-entry).
- Tradeable data is ~2 months of real book (Apr+Jun); the bridge to Feb-Mar uses an optimistic modeled spread.

## v0.2 Refinement — Synchronization Gate (BEST CASE, observation-only)

Added 2026-06-29. The cross-asset connection determines the outcome: fade only
when the market is SYNCHRONOUSLY deleveraging, not on an idiosyncratic ETH flush.

| Field | Value |
| --- | --- |
| Sync gate | concurrent cross-asset SELL-liquidation (BTC + SOL) in the prior 10 min >= 200K notional |
| Effect | keeps the synchronized cascades (the reliable bounces), drops the idiosyncratic ones (dead, ~0 contribution) |

Historical (bridged, modeled 2bps spread, 4 months):
- synchronized bucket: N=97, mean +16.0 bps/trade, win 59.8%, T3R +559.6, cal sum +1215 (win 62.7%) / hold sum +335 (win 53.3%) -> cal&hold positive.
- idiosyncratic bucket: N=72, mean +0.1, win 50.0%, T3R -905.6, cal/hold flip -> dead.

Note: the sync gate concentrates the edge (higher win rate / per-trade) but does
NOT cut the -410 tail (synchronized still has max_loss -410). It is a quality
filter, not a tail-cut. Still observation-only; same kill/promote criteria. The
sync gate must itself be confirmed forward and with real fills (it was found on
the bridged modeled-spread sample).

Tool: `tools/research_s34_synchronization_gate.py`.

## Forward Shadow Procedure

- Log every deep-V SELL cross (>=28 bps) as a paper LONG, entry at cross, exit 4h, taker fills from live book.
- Record per-trade net bps, win/loss, max adverse excursion, and the realized tail.
- Report weekly: N, sum, mean, median, win rate, max loss, and a chronological view. Keep the regime context (BTC trend, realized vol) per trade.
- Forward observations start from zero on the commit date of this protocol; do not re-tune the threshold, depth, or horizon on forward data.

## Kill / Promote Criteria

- KILL if 60-day forward sum is negative, OR forward win rate < 50%, OR a single tail exceeds the risk budget the operator has not pre-accepted.
- DO NOT promote to live on the historical sample alone. Promotion requires:
  1. forward out-of-sample positive across at least two distinct regimes (not just one favorable month), AND
  2. an independent-lane corroboration (BTC or SOL turning genuinely positive, not small-N), AND
  3. an explicit operator decision with position sizing set so the -410 tail is survivable, hand on the kill-switch.

## Tools

- Backtests: `tools/research_s34_reversal_backtest.py` (real fill), `research_s34_bridge_backtest.py` (bridged), `research_s34_failure_geometry.py`, `research_s34_confirmation_entry.py`.
- Navigation/permission layer: `tools/s34_cascade_navigation_dashboard.py` (would surface this as FADE only after forward validation).
