# CVD / TAKER-VOLUME DATA READINESS AND REPAIR RECONCILIATION — READ-ONLY AUDIT (2026-07-05)

**Scope:** read-only audit only. No outcomes read, no MFE/MAE, no thresholds chosen, no experiments written, no canonical.sqlite write, no backfill/repair executed, no collector change, no production/risk/execution/runtime code touched.

## 1. Source inventory

| Object | Location | Role |
|---|---|---|
| `agg_trades` | `data/microstructure.db` | Raw per-trade record: `ts_ms, symbol, price, quantity, notional, is_buyer_maker`. The ONLY raw source of taker-side information in the project. No dedup/UNIQUE constraint beyond autoincrement `id`. |
| `ami_candles.taker_buy_volume` / `.taker_sell_volume` | `data/ami/canonical.sqlite` | Candle-level aggregate, `SUM(quantity)` grouped by `is_buyer_maker` within `[open_ts_ms, close_ts_ms)`. **ETHUSDT only**, timeframes `1m`/`5m`. |
| `gaps` (`stream='agg_trades'`) | `data/microstructure.db` | Legacy heartbeat-based gap registry. 20 rows total, all dated 2026-04-12→04-24. Dead since (see §4). |
| `book_ticker.book_imbalance` | `data/microstructure.db` | Quote-side (bid/ask) imbalance — **not** a taker/CVD signal; different mechanism, must not be conflated. |
| No dedicated CVD table/view/cache exists anywhere in the repo | — | Verified by exhaustive grep across `ami/`, `tools/`, `reports/research/s34/`. |

**Existing CVD/order-flow-adjacent research (must not be duplicated):**
- `tools/research_s34_orderflow_lead.py` + `reports/research/s34/S34_ORDERFLOW_LEAD.md`: rolling-OFI (order-flow-imbalance) momentum strategy, agg-trade-derived, tested ETHUSDT/SOLUSDT/BTCUSDT across window/quantile grid. **Net-negative after cost on every configuration tested; "Leads (both-split positive @30s): none".** Graveyarded outcome — any future CVD work must not re-test raw OFI-momentum betting without new justification.
- `tools/orderflow_chart.py`: live dashboard (port 5051, ETHUSDT hardcoded). Its `cvd_series()` (lines 105-124) computes 1-minute-bucketed `SUM(notional) WHERE is_buyer_maker=0` (buy) minus `=1` (sell), then a running `cum` that **resets to 0 at whatever `start_ms` the caller passes** (the live UI always calls it with a 1-hour rolling lookback) — **ephemeral only, never persisted to any table, no fixed epoch/anchor.** This is the only place in the repo where a "CVD" series is actually computed end-to-end, and it confirms a definitional gap: **nothing in this codebase currently defines what a persistent, anchored CVD series should be cumulative *since*** (unlike liquidations' `reconstruct_anchors()`, which has a well-defined "previous accepted anchor" concept). Any birth-truncated CVD feature work must make this anchor choice explicitly, as a frozen design decision — it does not already exist implicitly anywhere to inherit.
- `tools/state_reconstruct.py`: replay-based state-vector reconstruction (`_get_trade_side()`, line 29) — confirms the same `is_buyer_maker` sign convention independently, used for a rolling (default 30s) local order-flow feature, not a persisted CVD series.
- `tools/research_ami_mfe50_experiment.py` (Paket 3, AMI Research OS-governed, already preregistered/run): uses `ofi10m`/`taker_sell10m` as **feature inputs** (not the primary research subject) via a direct ad-hoc query — `SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END)` over a fixed **10-minute trailing window ending at the feature-read timestamp**, bypassing `ami_candles.taker_buy_volume` entirely (reads raw `agg_trades` directly). This is a **third, independent, already-in-production pattern** for consuming taker-side data, alongside `candle_builder.py` and `orderflow_chart.py` — any new CVD/quality-contract work should reconcile with (or at minimum not silently diverge from) this precedent's own window/known-at conventions.
- `ami/chart/candle_builder.py` + `ami/chart/candle_gap_repair_rehearsal.py`: the only existing **canonical, tested** taker-volume computation + external-repair mechanism. Any CVD work should reuse these, not reimplement.
- No historical CVD-specific collector ever existed (verified via `git log -S cvd` across all history) — `agg_trades` has always been the sole raw source.

## 2. Field-semantics matrix

| Field | Meaning | Notes |
|---|---|---|
| `agg_trades.is_buyer_maker` | Binance aggTrade convention: `1` → resting order was a buy limit (maker was the buyer) → **taker was the seller** → taker-sell volume. `0` → **taker was the buyer** → taker-buy volume. | Confirmed against `candle_builder.py`'s own derivation code, matches Binance docs. |
| `agg_trades.ts_ms` | Per-trade event timestamp (not batched/binned at the raw level). | Safe as a `<=T` truncation boundary. |
| `ami_candles.taker_buy_volume/taker_sell_volume` (agg_trades-derived rows) | Exact `SUM(quantity)` over the candle's own `[open_ts_ms, close_ts_ms)` window. | **Exact, not approximate, conditional on the underlying agg_trades window itself being gap-free.** Inherits any agg_trades gap unchanged — it is the SAME rows, not an independent corroborating source. |
| `ami_candles.taker_buy_volume` (Binance-fapi-repair rows, `candle_definition_version=candle-binance-fapi-repair-v1`) | Binance kline field `[9]` (`takerBuyBaseAssetVolume`) — Binance's own authoritative aggregate over ALL trades in that 1m window, independent of local collector coverage. `taker_sell_volume` = candle `volume` (kline field `[5]`) − `taker_buy_volume` (both Binance-authoritative, so the subtraction is exact). | **Independent, authoritative repair source — but 1-minute-granularity only.** No intra-minute trade sequence is recoverable this way. |
| Venue/scope | Binance **USDT-M Futures** (`fapi`) exclusively, matching the project's established source discipline. No spot-market data collected at all. | |
| Symbol scope | `agg_trades`: ETHUSDT + BTCUSDT (from 2026-02-15), SOLUSDT (from 2026-04-18). `ami_candles` taker fields: **ETHUSDT only.** | |

## 3. Temporal coverage matrix (measured, not assumed)

| Symbol | agg_trades rows | agg_trades coverage | ami_candles (taker-bearing) coverage |
|---|---|---|---|
| BTCUSDT | 193,634,020 | 2026-02-15 14:26 → 2026-07-05 13:41 | **none built** |
| ETHUSDT | 174,465,734 | 2026-02-15 14:26 → 2026-07-05 13:41 | 1m: 199,180 rows / 5m: 40,093 rows, 2026-02-15 14:25 → 2026-07-03 22:05 |
| SOLUSDT | 20,121,325 | 2026-04-18 08:41 → 2026-07-05 13:41 (later start — added to the per-symbol subscription list mid-April) | **none built** |

## 4. Exact gap inventory

**Registry (`gaps` table, `stream='agg_trades'`):** 20 rows total, 18 resolved + 2 unresolved, ALL dated 2026-04-12→2026-04-24. **Nothing logged after 2026-04-24 10:27** — the same "gap-registry stopped being authoritative" cutoff already established for the liquidations stream in the prior source-quality-contract-v2 batch (the shared heartbeat/gap-detection code has been removed from the live collector entirely; no stream's registry is trustworthy past this point).

**Real, measured cadence gaps found in THIS audit (registry blind to all of these):**

| Window (approx., UTC) | Duration | Symbols affected | Evidence |
|---|---|---|---|
| 2026-04-30 11:00/18:07 → 05-02 14:19 | 44–51h | ETH + BTC simultaneously | recovery within 9s of each other |
| 2026-05-08 02:17/11:41 → 05-09 12:54 | 25–35h | ETH + BTC simultaneously | recovery within 13s |
| 2026-05-15 01:31 → 11:36 | 10.1h | SOL | |
| 2026-05-21 17:02–17:12 → 05-22 21:50–21:51 | 28.6–28.8h | **ETH + BTC + SOL simultaneously** | recovery within ~60s across all three |
| 2026-05-26 01:05/12:09 → 05-28 17:23 | 53–64h | ETH + BTC simultaneously (SOL continues into a further 3h gap right after) | recovery within 1s of each other |
| **2026-06-01 16:53 / 06-02 09:44 → 06-05 16:44** | **79–96h (3.3–4 days)** | ETH + BTC simultaneously — **the single largest identified outage** | recovery within 3s |

The near-identical recovery timestamps across symbols for every one of these windows is direct evidence of **collector-process-wide outages** (the whole collector was down/reconnecting), not per-symbol network issues. All of these fall inside or immediately adjacent to the already-known liquidations-stream 40.1-day blackout (2026-04-27→06-06) — corroborating one broader collector-instability episode across April–June 2026, not an isolated liquidations-specific bug.

**Open item, not resolved in this audit:** only the top-5 largest holes per symbol (post-2026-04-24) were enumerated; whether any material gaps exist **after** 2026-06-06 (once liquidations had also resumed) was not exhaustively checked — the pattern found so far suggests the post-06-06 era is materially cleaner, but this is a hypothesis, not yet measured.

## 5. Collector regime history

- One collector module: `data/microstructure_collector.py`, single combined websocket, `StreamParser` handling `forceOrder`/`aggTrade`/`markPrice@1s`.
- `agg_trades` subscription is **per-symbol** (`{sym}@aggTrade`) — unlike liquidations (which had a per-symbol→all-market mode transition), agg_trades has always used per-symbol subscriptions; no equivalent stream-mode cutoff applies here.
- SOLUSDT joined the symbol list ~2026-04-18 (matches its agg_trades start date exactly); BTC/ETH have been subscribed since 2026-02-15.
- The shared gap-detection/heartbeat code (`_check_stream_gap`/`_resolve_gap`, recovered from git stash `07e1a1f9`) has been **removed** from the current collector — no live gap-tracking exists for any stream today.
- No duplicate/late-arrival handling logic is visible in the schema (no UNIQUE constraint on `agg_trades` beyond the autoincrement `id`); a genuine duplicate/ordering-integrity scan has **not** been performed — flagged as an open item, not assumed clean.

## 6. Data-quality assessment

- **Absence of rows is not evidence of completeness** — doubly proven here: the newly-found multi-day May/June gaps were completely invisible to the dead registry.
- No positive completeness proof exists past ~2026-04-24 via the registry alone. A cadence-based proof (mirroring the already-frozen `liquidation-source-quality-contract-v2` methodology: max inter-arrival gap ≤ threshold within a required window) is demonstrably achievable — this audit performed a coarse version of it — but has **not** been formalized into a versioned, frozen contract for agg_trades/CVD the way it was for liquidations.
- **Candle-derived taker volume cannot repair trade-level gaps in the same source** — it is computed from the identical rows; a gap in `agg_trades` is inherited exactly, never independently corroborated.
- **Binance-kline-repair CAN provide an independent, authoritative repair — at 1-minute granularity only.** It recovers net taker buy/sell volume per closed 1m bar, never the intra-bar trade sequence. **Classification: exact repair of 1-minute net taker imbalance; permanently approximate (unrecoverable) relative to true tick-level CVD path inside any repaired bar.**
- The May/June gaps identified in this audit have **not** been addressed by the existing candle-repair batch (that batch targeted the earlier gaps discovered while the registry was still live, through 2026-04-27) — these are new, unaddressed findings.

## 7. Known-at safety assessment

- `agg_trades.ts_ms` is a per-trade event timestamp; `ts_ms <= T` truncation is safe by construction (same discipline already proven for the liquidation geometry work).
- `ami_candles.known_at_ts = close_ts_ms` is an already-tested, established invariant (`candle_builder.py`/`timing_contract.py`) — a candle's taker volume is knowable only at its own close, never earlier. No lookahead risk in the existing mechanism.
- **No birth-truncated CVD reconstruction function exists yet** (verified via grep — no hits combining "cvd"/"taker" with "birth"/"truncat" anywhere in the codebase). The required building blocks (known-at truncation, closed-candle discipline, `as_of_ms` lookup) are all already proven elsewhere and directly reusable — this would be new composition, not new invention.

## 8. Independent-cycle joinability

- **100% of canonical signals (LONG=220, SHORT=104) and canonical events (252) are ETHUSDT** — this exactly matches `ami_candles`' current ETHUSDT-only taker-volume scope. No immediate cross-symbol join gap exists for the *current* canonical population; it would only appear for a future BTC/SOL-scoped signal population.
- Identity keys (`signal_id → source_event_id → independent_cycle_id`) and the cycle-grouped train/test split machinery (`w8_short_expanded_baseline.compute_global_cycle_split`, already reused across 6+ modules) are directly reusable with zero redesign.
- **Not yet measured:** what fraction of the 220+104 canonical signals' own required source windows overlap the May/June gap windows found in §4. This is the natural next step — deliberately not performed here per the "no experiments" constraint on this audit.

## 9. Blockers

| ID | Blocker |
|---|---|
| B1 | No frozen, versioned source-quality contract exists for agg_trades/CVD (unlike liquidations' contract-v2) |
| B2 | Newly-discovered May–June multi-day outages are unquantified against the canonical signal population (no per-signal overlap check performed) |
| B3 | `ami_candles` taker-volume coverage is ETHUSDT-only — not currently blocking (matches signal population), but blocks any future BTC/SOL CVD work until candles are built from the already-available raw `agg_trades` for those symbols |
| B4 | No duplicate/ordering-integrity audit performed on `agg_trades` |
| B5 | No birth-truncated CVD reconstruction function exists — needs building (all prerequisites already proven/reusable) |
| B6 | Whether material gaps exist after 2026-06-06 has not been exhaustively checked |
| B7 | No frozen definition exists anywhere in the codebase for what a persistent CVD series should be cumulative *since* (no anchor/epoch concept) — the only working `cvd_series()` implementation (`orderflow_chart.py`) is an ephemeral, rolling-window-reset visualization value, never persisted; a genuine research-grade CVD feature needs this as an explicit, frozen design decision, not an inherited convention |

## Recommended next controlled batch

**CVD/TAKER-VOLUME SOURCE-QUALITY CONTRACT + BIRTH-TRUNCATED FEATURE DEFINITION-AUDIT** — mirroring the already-accepted liquidation-geometry precedent exactly:
1. Extend the gap/cadence audit to the full April–July range (not just the top-5-per-symbol sample used here), explicitly resolving B6.
2. Freeze a field-level, versioned source-quality contract for CVD windows (contract-v2-style), addressing B1/B4.
3. Freeze the CVD anchor/epoch definition explicitly (e.g. cumulative since signal-birth-minus-N, since the reconstructed bucket start, or another frozen choice) — addressing B7 — before defining the birth-truncated reconstruction algorithm + schema (disposable rehearsal only), addressing B5.
4. Run the per-signal overlap check against the 220+104 canonical ETHUSDT signals to measure the real, current quality distribution, addressing B2.
5. Reconcile the new definition against the 3 existing independent consumer patterns found in this audit (`candle_builder.py`, `orderflow_chart.py`, `research_ami_mfe50_experiment.py`) so the frozen contract doesn't silently diverge from already-in-production conventions.

All of the above are audit/definition/rehearsal work — no outcome, threshold, or experiment work is implied or recommended at this stage.

## Final verdict

**`CVD_DATA_REPAIR_REQUIRED`**

Not ready for preregistration as-is (real, substantial, previously-undocumented multi-day gaps exist in the exact window relevant to the canonical signal population, with no frozen quality contract to classify them). Not fully blocked either — raw `agg_trades` coverage is otherwise deep and long-running for all three symbols, ETHUSDT candle-level taker data already exists and is exactly scoped to the current signal population, and every methodological building block needed (source-quality contract pattern, birth-truncated reconstruction discipline, cycle-split joinability) is already proven and reusable from the liquidation-geometry batch. The path forward is well-defined and tractable.
