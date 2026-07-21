# S34 ECHO — Core Strategy Specification (causal core)

**Status:** RESEARCH_ONLY · NOT deployed · NO holdout · forward-validation-only (OD-028/029)
**Date:** 2026-07-20 · **Source of truth:** `tools/research_s34_echo_live_gauntlet.py` (gate `cand_9090` MINUS the `not noisy` lookahead)
**Supersedes:** the frozen `echo_30_90+regime` rule whose `not noisy` gate was a T+30m LOOKAHEAD (removed).

> This is the bare causal core that the 2026-07-20 microstructure investigation converged on. All
> numbers are IN-SAMPLE (burned Feb–Jul 2026) = necessary-not-sufficient. No gate, proxy, reactive
> overlay, or stop improved it (all dead/negative in-sample; see S34_ECHO_REACTIVE_ARM_SPEC).

---

## 1. Thesis
LONG fade of an ETH SELL-liquidation cascade that **echoes** a prior cascade, in a BTC downtrend —
buy the bottom of a repeat forced-selling flush and harvest the mean-reversion bounce.

## 2. Anchor (trigger event)
- Symbol **ETHUSDT**, side **SELL** liquidations.
- `reconstruct_anchors(bucket_sec=300, min_gap_sec=900, thresholds=(200_000,), accel_window_sec=30)`.
- Fires the instant cumulative SELL-liq notional within a 5-min bucket crosses **200,000 USD**, with
  15-min min-gap suppression between anchors.

## 3. Entry gates (ALL T0-knowable — NO lookahead)
| Gate | Rule |
|---|---|
| echo_30_90 | a prior anchor exists in `[T0−90m, T0−30m)` (the echo) |
| regime | `btc4h_bps < 0` OR `btc7d_bps < 0` (BTC downtrend) |
| not_bull | NOT (`eth1h_bps > 20` AND `btc4h_bps > 50`) |
| session | session ≠ EUROPE (not 07:00–13:00 UTC) |
| dow | weekday ∉ {Mon(0), Wed(2)} |

All five true → signal. (`cand_causal = not bull ∧ sess≠EUROPE ∧ dow∉{0,2} ∧ echo_30_90 ∧ regime`.)

## 4. Execution
- **Entry:** T0 (anchor timestamp), **LONG**, at mark price.
- **Hold:** 4h (primary); 6h marginally better in-sample.
- **Exit:** time-based at T0+hold. **NO stop, NO reactive cut** — both lower net in-sample (whipsaw).
- **Fee:** 5 bps net (baked in).

## 5. Removed vs the old frozen rule
- `not noisy` (∃ ≥50K ETH SELL liq in (T0+60s, T0+30m)) — **T+30m LOOKAHEAD, removed.** It produced the
  spurious "+92.5bps/tail0" by removing tails with hindsight (quantified: it did ~half the per-trade
  edge and 100% of tail removal; see S34_ECHO_CAUSAL_VS_LOOKAHEAD).

## 6. In-sample numbers (BURNED Feb–Jul 2026, causal core, hold 4h)
| Metric | Value |
|---|---|
| N | 118 (~26/mo raw; no-overlap fewer) |
| WR | 69.5% |
| Mean net | +41.2 bps/trade |
| Sum | +4856 bps |
| Worst | −338.9 bps |
| Tails (net<−100) | 14 (~12%) |
| No-overlap (real) | +560 (4h) · +1335 (6h) bps over 4.5 mo |

## 7. Tail nature (why there is no overlay)
- T0-irreducible: no clean-causal feature separates tails after BH-FDR (S34_ECHO_SEPARABILITY_STATS).
- Detectable at T+6 (own-P&L AUC ~0.82) but NOT actionable — the "looks-bad-at-T+6" set is mostly
  recoverers; cutting them (reactive or stop) destroys more than it saves (all arms Δ<0 in-sample).
- Mechanism: tails = a continued systemic BTC liquidation cascade in [T0, T0+10m]; unactionable.
- Only legitimate risk add: a very wide catastrophic stop (~−250bps) for max-drawdown control only
  (~5% return cost in-sample), forward-verify.

## 8. Validation status
- NO deploy, NO holdout — discovery burned all Feb–Jul 2026.
- Forward-only via `tools/research_s34_echo_forward_ledger.py` (records `qualified_t0` causal;
  currently empty, counts only anchors after 2026-07-20; ~months to accumulate ≥100 independent cascades).
- Live executor default-OFF, triple-gated. This spec authorizes NO deploy.
- Calibrated forward-survival estimate ~25–40%.

---
*Companion reports (reports/research/s34/): S34_ECHO_CAUSAL_VS_LOOKAHEAD, S34_ECHO_SEPARABILITY_STATS,*
*S34_ECHO_BE_RATIO_INFOCURVE, S34_ECHO_TAIL_FORENSICS, S34_ECHO_REACTIVE_ARM_SPEC, ECHO_LIQ_IMPACT_PROXIES_v1.*
