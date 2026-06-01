# Untapped Paths Run Results

Date: 2026-06-01

Scope: execution pass over the seven highest-value paths from `docs/ALPHA_COMPETITIVE_EDGE_MAP_2026-06-01.md`.

## Verdict Ranking

| Rank | Path | Run Verdict | Why |
|---:|---|---|---|
| 1 | SOL forced-flow transfer | SHADOW_CANDIDATE | Fresh SOL run found BUY liquidation -> SHORT at 15m: N=46, WR=73.91%, mean=+15.78 bps |
| 2 | S34 single-large liquidation composition | GO / promote to serious validation | Single-large ETH branch: simulated filled subset N=10, WR=80.0%, net NPA=+5.11 bps |
| 3 | S34 basis / geometry / entropy branches | MONITOR / candidate filters | Positive basis branch N=30, WR120=66.7%, mean=+5.33 bps; cluster quality separates |
| 4 | Venue / fee routing | CONDITIONAL | Existing high-frequency ETH onset edge remains fee-gated; viable only if all-in taker cost is below gross edge |
| 5 | True bookTicker / queue state | DIAGNOSTIC YES, ALPHA NO | bookTicker data is live and large, but deep Stage C found no edge uplift |
| 6 | Passive-then-taker refresh | KILL current 21D claim | Pocket B and tight-mid both failed 21D refresh: pass_count=0/12 under baseline and PTT |
| 7 | DeFi liquidation linkage | BLOCKED | No local DeFi liquidation table or tokenized The Graph collection yet |
| 8 | Event-lane conditional discovery | BLOCKED / stale tooling | 24h lane check timed out; 60m returned no_data |

## What Was Run

### 1. Book / Queue State

Commands:
- `python tools/bookticker_validation.py`
- `python tools/analyze_bookticker_recovery.py --db-path data/microstructure.db --limit 500`
- `python tools/research_s69_obi_entry.py`
- `python deep_discovery/scripts/stage_c_book_ticker.py`

Results:
- `book_ticker` is live for BTC/ETH/SOL.
- BookTicker validation:
  - BTC last age: 19.336s, 6h ticks: 8,812,600
  - ETH last age: 4.652s, 6h ticks: 8,663,532
  - SOL last age: 4.532s, 6h ticks: 2,522,440
- S34 book recovery analysis processed 73 detector rows, updated 17 with book-state fields.
- Updated detector rows mostly show `entry_book_state` as `recovered` or `partial`; measured spread recovery times are near-zero seconds in the current implementation.
- `research_s69_obi_entry.py` returned `INSUFFICIENT_EVIDENCE` because it expects an `orderbook` snapshot table, not the existing `book_ticker` table.
- Deep Discovery Stage C:
  - Apr11+ baseline: +3.1233 bps
  - `book_imb_aligned`: N=251, +2.576 bps, worse than baseline
  - high quote intensity: N=133, +3.472 bps, only +0.349 bps uplift, not enough
  - microprice aligned: N=301, +3.217 bps, not enough
  - verdict: FAIL

Interpretation:
- True top-of-book collection is now operational.
- Simple bookTicker features do not rescue or amplify the existing ETH microstructure edge.
- The useful next step is not another simple `book_imbalance` filter. It is a better book-state model: recovery clocks, depth shock, quote refill, queue asymmetry, and state transitions around single-large cascades.

### 2. Venue / Fee / Routing

Source:
- `execution_engineering/FINAL_EXECUTION_ENGINEERING_REPORT.md`
- `core/fee_model.py`
- current positive branches from this run

Current economics:
- `EE_TAKER_VIP_ONSET`: +3.12 bps gross, +1.12 bps net only at 2.0 bps/side taker.
- Standard taker economics remain non-viable.
- S49 single-large passive simulation showed +5.11 bps net on filled signals under the script's 2.0 bps/side fee assumption, but only 47.6% fill rate.
- SOL BUY->SHORT at 15m is gross mark-return only in this run; it still needs execution/fee modeling.

Interpretation:
- Routing remains a real edge lever, but only after a candidate survives gross and fill tests.
- The strongest new SOL result must be execution-modeled before any fee claim.

### 3. S34 Bad-Fingerprint / Branch Rejection

Commands:
- `python tools/research_s53_purity_ofi.py`
- `python tools/research_s54_entropy_basis.py`
- `python tools/research_s49_single_large_validation.py`
- `python tools/research_s44_btc_contagion.py`
- `python tools/research_s45_fragility_reversal.py`

Positive:
- S49 single-large:
  - matched 30/52 historical pkl signals
  - single_large: N=13, WR=84.6%
  - clustered: N=17, WR=52.9%
  - passive simulation across all 21 single_large signals:
    - filled: 10/21 = 47.6%
    - filled WR: 80.0%
    - mean gross: +7.11 bps
    - mean net: +5.11 bps
    - NPA per attempt: +2.43 bps
  - verdict: GO
- S54:
  - basis > 0.5 bps: N=30, WR120=66.7%, mean120=+5.33 bps
  - basis < 0: N=21, WR120=23.8%, mean120=-15.64 bps
  - cluster_A_high_quality: N=5, WR60=80.0%, mean60=+8.91 bps
  - cluster_C_low_quality: N=23, WR60=52.2%, mean60=+1.53 bps
- S53:
  - ETH OFI > 0.3: N=8, WR120=75.0%, mean120=+16.87 bps
  - ETH OFI < -0.3: N=4, WR120=0.0%, mean120=-11.47 bps
  - BTC OFI < -0.3: N=10, WR120=80.0%, mean120=+6.18 bps

Negative:
- S44 BTC contagion:
  - isolated was not better than contagion
  - isolated WR60=59.7%, contagion WR60=66.7%, but contagion N=6
  - verdict: NO EDGE
- S45 fragility reversal:
  - prev EXTREME/COLD -> GOLDILOCKS: N=15, WR60=33.3%
  - prev GOLDILOCKS -> GOLDILOCKS: N=18, WR60=77.8%
  - verdict: NO EDGE for the proposed reversal thesis

Interpretation:
- The best S34 next path is not BTC isolation or bad-zone reversal.
- It is composition/fingerprint: single-large vs clustered, basis sign, OFI sign, and cluster quality.
- Promote `single_large` to a dedicated forward-validation package.

### 4. DeFi Liquidation Linkage

Commands:
- repo search and `docs/research/DEFI_DATA_PLAN.md` review

Result:
- No local `defi_liquidations` table exists.
- No runnable collector exists in the current tool surface.
- Prior plan requires `THEGRAPH_API_TOKEN` and tokenized gateway access.

Interpretation:
- This path is still high-value but blocked. It should be scaffolded as preview-only before persistence.

### 5. Passive-Then-Taker Refresh

Commands:
- 21D Pocket B baseline and passive-then-taker
- 21D tight-mid baseline and passive-then-taker

Results:
- Pocket B baseline: pass_count=0/12, insufficient_fill_rate=100%
- Pocket B passive-then-taker: pass_count=0/12, insufficient_fill_rate=100%
- Tight-mid baseline: pass_count=0/12, insufficient_fill_rate=100%
- Tight-mid passive-then-taker: pass_count=0/12, insufficient_fill_rate=100%

Interpretation:
- The old 7D passive-then-taker rescue did not survive this 21D refresh.
- Freeze the current promotion claim. Do not spend more cycles on PTT until the validation harness explains why all validation rows have zero attempts/fills under current data.

### 6. Event-Lane Conditional Discovery

Commands:
- `python -m tools.check_event_lanes --db data/microstructure.db --symbol ETHUSDT --lookback-min 1440 --json`
- `python -m tools.check_event_lanes --db data/microstructure.db --symbol ETHUSDT --lookback-min 60 --json`

Results:
- 24h check timed out.
- 60m check returned `gate=UNKNOWN`, `reason=no_data`.

Interpretation:
- Current event-lane live gate is not usable for historical discovery in this run.
- Treat event lanes as tooling debt before alpha search.

### 7. SOL Forced-Flow Transfer

Command:
- `python tools/research_sol_forced_flow_transfer.py`

Results:
- Coverage:
  - SELL >=25k: N=114; >=50k: N=39; >=100k: N=23
  - BUY >=25k: N=76; >=50k: N=46; >=100k: N=24
- Best N>=20:
  - BUY liquidation -> SHORT
  - threshold: >=50k
  - horizon: 900s
  - N=46
  - WR=73.91%
  - mean=+15.78 bps
  - median=+16.55 bps
- Also good:
  - BUY >=25k -> SHORT 900s: N=76, WR=68.42%, mean=+13.22 bps
  - BUY >=100k -> SHORT 900s: N=24, WR=70.83%, mean=+14.41 bps
- SELL->LONG is weaker:
  - SELL >=50k -> LONG 900s: N=39, WR=58.97%, mean=+5.90 bps

Interpretation:
- This is the strongest fresh untapped result from the run.
- It contradicts some earlier notes that emphasized SOL SELL->LONG as the candidate. Current data says SOL BUY->SHORT is cleaner.
- Promote SOL BUY-liquidation short continuation to shadow research, with immediate execution modeling and walk-forward discipline.

## Final Action List

1. Build a dedicated `SOL_BUY_LIQ_SHORT_15M` hypothesis package.
   - Freeze threshold grid: 25k, 50k, 100k.
   - Primary candidate: BUY >=50k, h=900s.
   - Add walk-forward and fee/slippage modeling before any runtime wiring.

2. Build `S34_SINGLE_LARGE_V1`.
   - Use single-large composition as a pre-entry gate.
   - Validate against clustered branch.
   - Add fill-rate and passive/taker execution comparison.

3. Freeze passive-then-taker promotion language.
   - Current 21D refresh failed hard.
   - Investigate validator zero-attempt behavior before rerunning broader grids.

4. Keep bookTicker, but pivot the model.
   - Stop testing raw `book_imbalance` as the main uplift.
   - Build state-transition features: quote refill, depth recovery, spread shock persistence, and recovery-before-entry labels.

5. Defer DeFi until token access is available.
   - Add preview collector only when `THEGRAPH_API_TOKEN` exists.

6. Repair event-lane discovery tooling.
   - 24h timeout and 60m no_data means this lane is not ready for alpha search.

## Artifacts

- `reports/BOOKTICKER_VALIDATION.md`
- `deep_discovery/reports/stage_c_book_ticker.md`
- `reports/S49_SINGLE_LARGE_VALIDATION.md`
- `reports/S53_PURITY_OFI.md`
- `reports/S54_ENTROPY_BASIS.md`
- `reports/S37_MIRROR_LONG.md`
- `reports/S44_BTC_CONTAGION.md`
- `reports/S45_FRAGILITY_REVERSAL.md`
- `reports/S70_FUNDING.md`
- `reports/PTT_REFRESH_21D_B_BASELINE.md`
- `reports/PTT_REFRESH_21D_B_PTT.md`
- `reports/PTT_REFRESH_21D_TIGHTMID_BASELINE.md`
- `reports/PTT_REFRESH_21D_TIGHTMID_PTT.md`
- `reports/SOL_FORCED_FLOW_TRANSFER.md`
