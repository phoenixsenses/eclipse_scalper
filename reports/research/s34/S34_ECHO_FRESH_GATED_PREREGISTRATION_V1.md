# S34 ECHO — FRESH GATED PRE-REGISTRATION V1

**Date:** 2026-07-20 · **Author:** Opus 4.8 [1M] · **Status:** `PREREGISTERED_AWAITING_INDEPENDENT_REVIEW`
**Governs:** the last un-burned alpha lead `echo_30_90 + regime` (memory `project_canonical_alpha_status_jul2026`, §147).
**Discipline:** OD-028 (new lead = fresh prereg + independent historical holdout, N≥100 independent cascades, pre-frozen features/thresholds, no retrofit) · OD-029 (no new optimization on burned samples) · CLAUDE.md (no lookahead; thresholds frozen; FEE=5bps net; MC permutation standard).

---

## 1. FROZEN RULE (no parameter may change after this document is accepted)

Source of truth: `tools/research_s34_echo_live_gauntlet.py::cand_9090`. Copied here verbatim so the rule is frozen independently of the file.

| Component | Frozen value |
|---|---|
| Symbol / side | ETHUSDT, LONG fade of SELL-flush |
| Anchor | `reconstruct_anchors(bucket_sec=300, min_gap_sec=900, thresholds=(200_000,), accel_window_sec=30)` (canonical, causal, no-lookahead) |
| Gate `regime` | `btc4h_bps < 0 OR btc7d_bps < 0` |
| Gate `echo_30_90` | ∃ a prior SELL-anchor timestamp in window **[T−90m, T−30m)** (echo of an earlier cluster) |
| Gate `not bull` | NOT (`eth1h_bps > 20` AND `btc4h_bps > 50`) |
| Gate `session` | `session != EUROPE` (EUROPE = UTC hour ∈ [7,13)) |
| Gate `not noisy` | NO ETH SELL ≥50K liq in **(T+60s, T+30m)** |
| Gate `dow` | weekday ∉ {0 (Mon), 2 (Wed)} |
| Entry | **T0** (anchor timestamp), mark fill |
| Hold / exit | **4h**, no stop (primary). Secondary registered variant: 4h + hard stop −150bps |
| Cost | FEE = **5 bps net** round-trip |
| Realistic frequency | **no-overlap** (4h busy-lock), reported per-month |

Any change to any cell above = a NEW prereg (V2), not this one.

## 2. BURNED vs UN-BURNED — HOLDOUT PARTITION (the binding constraint)

- The discovery run (WR≈81%, tail 0, mc_p 0.0, WF 5/5, ~7/mo) used `LOOKBACK=400d` = the **entire** available liquidation history: **2026-02-15 → 2026-07-19** (ETH SELL≥200K liqs exist only from 2026-02-15). book_ticker only from 2026-04-11.
- **Therefore there is NO un-burned historical slice.** OD-028's "independent historical holdout N≥100" **cannot be satisfied from existing data** — the whole population is burned.
- **Verdict: holdout is FORWARD-ONLY.** The un-burned test set = anchors with `anchor_ts_ms > 2026-07-20T00:00Z` (strictly after this prereg is written), recorded by a frozen-rule forward ledger, never mined.
- **Maturity:** at ~7 qualifying no-overlap events/month, N≥100 independent cascades ≈ **~14 months** forward (or blended with the registered `echo_30_120` sibling ~9.7/mo → faster but a separate registered arm). This is the honest cost of the OD-028/029 gate; no shortcut exists without burning the lead.

## 3. SUCCESS CRITERIA (evaluated ONCE, on the matured forward holdout only)

Pre-declared PASS (all must hold on the forward holdout, N≥100 no-overlap, FEE=5bps net):
1. WR ≥ 70% (discovery 81%; a real edge should not collapse below 70).
2. Net avg > 0 AND MC permutation `mc_p ≤ 0.05`.
3. Tail (`net < −100bps`) count ≤ 2 per ~100 events.
4. No-overlap per-month ≥ 4 (executable frequency).
5. Walk-forward: ≥ 4/5 forward folds net-positive.

FAIL / graveyard if any of (1)-(3) breaks materially. PARTIAL (revisit, not deploy) if frequency/WF weak but WR+mc pass.

## 4. FORBIDDEN (contamination guards)

- No threshold/feature/session/dow re-optimization — §1 is frozen (OD-029).
- No peeking at the forward ledger to tune anything; a single terminal evaluation at maturity.
- No re-running discovery on the burned Feb–Jul sample and calling it validation.
- Live executor stays default-OFF, triple-gated; this prereg authorizes **no** deploy — only a forward measurement.

## 5. NEXT PHASES (gated, one at a time, operator sign-off between each)

1. **This prereg → independent read-only review** (does §1 faithfully freeze `cand_9090`? is §2 partition honest? are §3 criteria pre-committed and falsifiable?).
2. On acceptance → build a **frozen-rule echo forward ledger** (append-only, OPEN at T0 / CLOSE at +4h, records gate booleans + net; mirrors `liq_tip_forward.py` discipline) → operator activation.
3. Forward accumulation (~months) → single terminal evaluation vs §3.

**No verdict is possible today. Any today-verdict would burn the lead — which is exactly what this prereg exists to prevent.**
