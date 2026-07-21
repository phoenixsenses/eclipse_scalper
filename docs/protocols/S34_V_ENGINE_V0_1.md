# S34 V Engine v0.1 Mini-Protocol

Status: `EXPLORATORY_FROZEN`

Protocol id: `S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D`

Created: 2026-06-28

## Purpose

Freeze the first clean V-engine exploratory candidate so future observations are counted from zero. This protocol is not a live rule and not a paper-candidate. It exists to prevent post-hoc edits after the multi-feature diagnostics pass.

## Frozen Rule

| Field | Value |
| --- | --- |
| Symbol | `ETHUSDT` |
| Liquidation side | `SELL` |
| Trigger | running liquidation notional first crosses `200000 USDT` |
| V-depth | `28 <= vdepth_bps < 40` |
| Prior trend | `prior_4h_bps < -50` |
| Entry model | maker `LONG` |
| Limit offset | `20 bps` below anchor mark/extreme |
| Conservative fill | future mark must cross `2 bps` beyond the limit |
| Exit | fixed `2h` after maker fill |
| Fees | maker entry fee/rebate model + taker exit fee as configured in research harness |
| Permission | `EXPLORATORY_V_FADE_V0_1` observation only |

`vdepth_bps` is computed from cascade start to the knowable threshold-cross anchor:

```text
vdepth_bps = (mark_at_first_liq - mark_at_anchor) / mark_at_first_liq * 10000
```

`prior_4h_bps` is computed point-in-time:

```text
prior_4h_bps = (mark_at_anchor - mark_4h_before_anchor) / mark_4h_before_anchor * 10000
```

## Evidence That Motivated The Freeze

Source report: `reports/research/s34/S34_MAKER_FADE_DIAGNOSTICS.md`

Exploratory cell:

```text
H2_O20_C2
vdepth_bucket=v28_40 & prior_4h_bucket=prior4h_down
cal:  N=6,  sum=+220.0 bps, median=+28.3 bps, T3R=+79.8 bps
hold: N=14, sum=+681.9 bps, median=+43.2 bps, T3R=+160.6 bps
```

This is not sufficient for deployment because calibration N is only 6. The result is a forward-observation hypothesis, not a validated edge.

## Guardrails

- No live trading.
- No normal paper-trade acceptance counter.
- No parameter edits inside v0.1 after this freeze.
- Any sensitivity work is labeled v0.2 exploratory and cannot retroactively modify v0.1.
- Forward observations must start from zero after this document.
- Report total P&L, median, top-3-winner-removed, fill rate, and no-fill count.

## Forward Observation

For the first 30-60 days, this protocol may be shown in dashboard/navigation only as:

```text
EXPLORATORY_V_FADE_V0_1
```

The permission means "observe/log only". It does not authorize live capital or paper-candidate promotion.

## Monitoring

Observation ledger:

```text
reports/research/s34/S34_V_ENGINE_V0_1_OBSERVATION_LEDGER.jsonl
reports/research/s34/S34_V_ENGINE_V0_1_OBSERVATION_LEDGER.csv
```

Weekly brief:

```text
reports/research/s34/S34_V_ENGINE_V0_1_WEEKLY_BRIEF.md
reports/research/s34/S34_V_ENGINE_V0_1_WEEKLY_BRIEF.json
```

Refresh command:

```text
python tools/s34_v_engine_shadow_observer.py
```

The observer is append/dedupe by `observation_id`. It records knowable signal fields at the threshold-cross anchor and outcome labels only after the data window exists. It does not write to `microstructure.db`, does not create paper trades, and does not authorize live orders.

Required review fields:

- signal count
- closed fill count and fill rate
- total net bps
- median net bps
- top-3-winner-removed net bps
- closed no-fill counterfactual
- latest winner/loser observations

## Kill Criteria

Keep this frozen candidate exploratory unless a later forward report clears a separately pre-registered gate. A conservative kill condition for the observation phase:

```text
60-day forward T3R < 0
```

If killed, archive as `RESEARCH_ONLY_SKEW_DEPENDENT`.
