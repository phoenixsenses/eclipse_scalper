# EXECUTION HARDENING PLAN — V1

*Generated: 2026-02-21 | Based on RANK_V3_costgrid.json + triage_capacity output*

---

## 1. Situation Summary

Forward validation of `micro_edge_v3_passive_alpha` on ETHUSDT (horizon=120s) shows:

- **Raw edge exists**: `score_raw_core` is positive (~1e-4 per fill), confirming a directional
  signal survives the train/val split and the passive fill proxy.
- **NPA is negative**: Across all cost settings tested (fee=1.0 bps, adv_mult=1.0–1.5),
  `median_net_per_attempt` is negative, meaning costs dominate the raw edge.
- **Capacity is thin**: Attempts per minute range ~0.03–0.10. The rarest pocket
  has `p_fail_insufficient ≈ 43%`. Most pockets see 3–20% fail rate due to
  too few fills per validation split.
- **Relaxed thresholds survive**: At `splits=3, min_n=20, min_n_frac=0.00010,
  pass_threshold=0.33`, at least 1 pocket survives fee≥1.0 with pass_rate≥0.5.

**Conclusion**: The microstructure signal is real but too expensive to monetise
at current maker fee + adverse selection assumptions.  The path to profitability
requires either (a) reducing cost drag or (b) improving signal filters.

---

## 2. Cost Breakdown (from analyze_cost_breakdown.py)

Run:
```bash
python -m tools.analyze_cost_breakdown \
  --rank-json reports/RANK_V3_costgrid.json \
  --out-md    reports/COST_BREAKDOWN_V1.md \
  --out-json  reports/COST_BREAKDOWN_V1.json
```

Key output fields per pocket:

| Field | Meaning |
|---|---|
| `raw_edge_proxy_bps` | Estimated gross edge (NPA at zero cost), in bps |
| `fee_drag_bps_implied` | Cost attributed to round-trip maker fee, in bps |
| `adverse_drag_bps_implied` | Cost attributed to adverse selection, in bps |
| `dominator` | `fee` / `adverse` / `balanced` |
| `break_even_fee_bps` | Max fee at which pocket becomes profitable (at ref adv_mult) |
| `break_even_adv_mult` | Max adv_mult at which pocket becomes profitable (at ref fee) |
| `gap_to_breakeven_fee_bps` | How far current fee is from break-even (negative = already profitable) |

### Decision rules

| `dominator` | Action |
|---|---|
| `fee` | Pursue maker rebate tier or reduce entry frequency |
| `adverse` | Apply `--mitigation-profile anti_adverse_v1` |
| `balanced` | Address both; start with the larger `gap_to_breakeven_*` |

---

## 3. Adverse Selection Calibration (from fit_adverse_model.py)

The passive simulator uses a synthetic `base_adverse_bps` inferred from
training samples.  This may not reflect the true conditional adverse selection.

Run:
```bash
python -m tools.fit_adverse_model \
  --db data/microstructure.db \
  --symbol ETHUSDT \
  --lookback-min 1440 \
  --bucket-sec 1 \
  --rule micro_edge_v3_passive_alpha \
  --h 120 \
  --out-json reports/ADVERSE_MODEL_ETH_120.json \
  --out-md   reports/ADVERSE_MODEL_ETH_120.md
```

Key output fields:

| Field | Meaning |
|---|---|
| `global_mean_adverse_bps` | Empirical average adverse on signal-firing buckets |
| `implied_adverse_mult_vs_1bps` | Suggested `passive_adverse_mult` given 1 bps baseline |
| `by_spread_quartile.{Q1..Q4}.mean` | Conditional adverse by spread regime |
| `by_intensity_quartile.{Q1..Q4}.mean` | Conditional adverse by intensity regime |

**How to use the model in ranking:**

The calibrated model JSON is loaded via
`execution.passive_execution_simulator.load_adverse_model(path)`.
When passed to `simulate_passive_fill(..., adverse_model=model)`, the simulator
uses the empirical conditional adverse instead of the internal blend — giving
more realistic NPA estimates.

---

## 4. Mitigation Experiments (--mitigation-profile)

### anti_adverse_v1

Tightens pre-attempt filters to trade fill-rate for lower adverse selection:
- `min_imbalance_strong = min_imbalance × 1.25` — requires stronger directional signal
- `max_spread_tight     = max_spread × 0.75`     — avoids wide-spread / low-liquidity bars

Run:
```bash
python -m tools.rank_passive_pockets_forward \
  --candidates-md reports/RANK_V3_splits4.md \
  --db data/microstructure.db \
  --lookback-min 1440 \
  --splits 3 \
  --min-n 20 \
  --min-n-frac 0.00010 \
  --maker-fee-bps-grid "0.5,1.0,1.5" \
  --passive-adverse-mult-grid "0.8,1.0,1.2,1.5" \
  --pass-threshold 0.33 \
  --mitigation-profile anti_adverse_v1 \
  --out-md   reports/RANK_V3_anti_adverse_v1.md \
  --out-json reports/RANK_V3_anti_adverse_v1.json
```

Compare `npa_core` and `pass_rate_core` between `baseline` and `anti_adverse_v1` runs.

### Expected trade-offs

| Metric | Baseline | anti_adverse_v1 | Expected direction |
|---|---|---|---|
| `attempts_per_min` | ~0.05–0.10 | Lower | Fewer signals pass gate |
| `attempt_fill_rate` | ~0.60–0.70 | Similar or higher | Tighter spread = easier fill |
| `npa_core` | Negative | Less negative or positive | Lower adverse drag |
| `pass_rate_core` | ~0.33 | Higher (target ≥0.50) | Better signal quality |

---

## 5. Pockets to Promote to Paper Trading

**Promotion criteria** (update after running the experiments above):

1. `pass_rate_core ≥ 0.50` at `fee=1.0, adv_mult=1.0` (or adv_mult from model)
2. `npa_core > 0` (positive net per attempt, including unfilled=0)
3. `attempt_fill_rate ≥ 0.50` (fills at least half the time)
4. `insufficient_fill_rate < 0.30` (not starved of data)
5. `gap_to_breakeven_fee_bps ≤ 0` (already profitable at reference cost)

**Candidate ranking** (fill after running the tools above):

| Rank | Symbol | h | imb | int | spr | Profile | npa_core | pass_rate_core | Status |
|---:|---|---:|---:|---:|---:|---|---:|---:|---|
| 1 | ETHUSDT | 120 | TBD | TBD | TBD | anti_adverse_v1 | TBD | TBD | Pending |

---

## 6. Recommended Next Steps

```
Step 1  Run analyze_cost_breakdown.py on RANK_V3_costgrid.json
        → Identify whether fee or adverse dominates per pocket

Step 2  Run fit_adverse_model.py on microstructure.db
        → Get empirical global_mean_adverse_bps and by-quartile breakdown

Step 3  Compare global_mean_adverse_bps vs the simulator's default base_adverse_bps
        → If empirical > assumed, the current NPA estimates are optimistic

Step 4  Re-run ranking with anti_adverse_v1 profile
        → Measure npa_core improvement vs baseline

Step 5  If any pocket achieves npa_core > 0 and pass_rate_core >= 0.5:
        → Promote to paper trading via run-bot.ps1 with SCALPER_DRY_RUN=1
        → Monitor paper_scoreboard.json for 48h

Step 6  After 48h paper run: compare paper NPA vs backtest NPA
        → If within 2× of backtest NPA, promote to live with FIRST_LIVE_SAFE=1
```

---

## 7. File Inventory

| File | Purpose |
|---|---|
| `tools/analyze_cost_breakdown.py` | Fee vs adverse decomposition, break-even analysis |
| `tools/fit_adverse_model.py` | Empirical adverse calibration from microstructure DB |
| `execution/passive_execution_simulator.py` | Upgraded with `load_adverse_model` + `get_conditional_adverse_bps` |
| `tools/rank_passive_pockets_forward.py` | Extended with `--mitigation-profile` |
| `tests/test_analyze_cost_breakdown.py` | Unit + smoke tests for cost breakdown tool |
| `tests/test_fit_adverse_model.py` | Unit tests for adverse model fitter |
| `reports/COST_BREAKDOWN_V1.json` | Generated by analyze_cost_breakdown |
| `reports/ADVERSE_MODEL_ETH_120.json` | Generated by fit_adverse_model |
| `reports/RANK_V3_anti_adverse_v1.json` | Generated by rank with anti_adverse_v1 |
