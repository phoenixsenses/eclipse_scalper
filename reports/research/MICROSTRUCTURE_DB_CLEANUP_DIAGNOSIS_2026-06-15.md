# Microstructure DB Cleanup Diagnosis - 2026-06-15

Scope: read-only diagnosis only. No files were deleted, no `VACUUM` was run, and the S34 runner/config were not touched.

## 1. Table-Level Size And Coverage

### SQLite File State

| Metric | Value |
|---|---:|
| DB path | `data/microstructure.db` |
| Current file size | 488.9 GB decimal / 455.4 GiB |
| WAL file | ~4.4 MB during audit |
| Journal mode | `wal` |
| `page_count` | 119,347,622 |
| `page_size` | 4,096 |
| `freelist_count` | 0 |
| `dbstat` | unavailable (`no such table: dbstat`) |
| D: free space | ~1.02 TiB |

`page_count * page_size` matches the main DB file closely, and `freelist_count=0` means there is no meaningful already-free space inside SQLite to reclaim. Shrinkage requires deleting/copying data and then either `VACUUM` or a clean rebuild.

Exact `COUNT(*)` on the largest tables did not finish within 5 minutes while the runner was active. For autoincrement tables, I used `sqlite_sequence.seq` as the low-risk row-count proxy; with `freelist_count=0`, it is a useful estimate, but not a formal exact count.

### Large Table Inventory

| Table | Count Basis | Approx Rows | First UTC | Last UTC | Days | Approx Disk Share |
|---|---:|---:|---|---|---:|---:|
| `book_ticker` | `sqlite_sequence` | 2,996,199,747 | 2026-04-11 17:08:41 | 2026-06-15 07:55:09 | 64.6 | ~380-420 GiB |
| `detector_heartbeat` | `sqlite_sequence` | 551,629,265 | 2026-04-02 18:10:52 | 2026-05-31 10:07:16 | 58.7 | ~20-40 GiB |
| `agg_trades` | `sqlite_sequence` | 326,902,561 | 2026-02-15 14:26:27 | 2026-06-15 07:55:08 | 119.7 | ~20-35 GiB |
| `mark_prices` | `sqlite_sequence` | 15,602,544 | 2026-02-15 14:26:28 | 2026-06-15 07:55:08 | 119.7 | ~1-3 GiB |
| `liquidations` | `sqlite_sequence` | 535,757 | 2026-02-15 14:30:18 | 2026-06-15 07:54:54 | 119.7 | <0.2 GiB |

The main space driver is not `agg_trades`; it is almost certainly `book_ticker`. The second surprise is `detector_heartbeat`: 551M rows, no code references found in `tools/`, `data/`, or `tests/`, and no rows after 2026-05-31 in rowid order.

### Smaller Tables

| Table | Exact Rows | Coverage / Notes |
|---|---:|---|
| `detector_log` | 260,811 | 2026-02-15 to 2026-05-31 by rowid |
| `event_diary` | 133,999 | 2026-03-14 to 2026-06-05 |
| `spot_prices` | 172,862 | 2026-03-07 to 2026-06-05 |
| `vol_state` | 53,648 | exact count only |
| `basis_reversion_candidates` | 1,430 | PR #7 artifact |
| `gaps` | 812 | gap records |
| `open_interest` | 501 | PR #8 artifact |
| `funding_rates` | 178 | PR #7 artifact |
| `liq_heatmap` | 133 | small aggregate |
| `detector_signals` | 73 exact current count | `sqlite_sequence` showed historical seq 125 |
| `sol_s35_candidates` | 25 | small research artifact |

These smaller tables do not explain the 455 GiB file. Cleaning them is not worth risk unless tied to a separate schema cleanup.

## 2. S34 Forward Test Dependency

### Tables The Runner Reads

From [tools/s34_shadow_paper_runner.py](D:/eclipse_scalper/tools/s34_shadow_paper_runner.py):

| Purpose | Table | Code Evidence | Window |
|---|---|---|---|
| Signal detection | `liquidations` | `_bucket_events()` groups liquidation buckets | cursor/start to latest |
| Regime trend/range | `mark_prices` | `_regime_snapshot()` first/last/min/max | UTC day start to signal time |
| Regime buy-liq notional | `liquidations` | `_regime_snapshot()` count/sum BUY | UTC day start to signal time |
| Regime agg count | `agg_trades` | `_regime_snapshot()` count/sum | UTC day start to signal time |
| Entry/exit reference | `mark_prices` | `_mark_at()` and `_evaluate_trade()` | signal/mark cursor to max horizon |
| Executable fills | `book_ticker` | `_book_ticker_at()` via `_fill_quote()` | nearest quote before fill timestamp |

The regime filter is explicitly `utc_day_so_far`; it does not need months of raw rows for live operation. Current open-trade evaluation uses a monotonic cursor and max horizon from the trade rule; the observed S34 rule uses a 1-hour max horizon.

### Hot Data Window

| Data | Minimal Live Need | Practical Safe Window | Forward-Test Caveat |
|---|---:|---:|---|
| `mark_prices` | current UTC day + open trade horizon | 7 days | needed for entry refs and exit replay |
| `liquidations` | current UTC day + signal cursor onward | 30 days or since validation start | old rows useful for audit/replay |
| `agg_trades` | current UTC day for count/sum | 7 days | S34 only needs counts, not old raw rows |
| `book_ticker` | fill timestamps for open/new trades | since validation start until N=100 | do not delete validation-period quotes before final recompute |

For pure live S34, old raw `agg_trades` are not needed after the UTC day rolls over. For the validation program, the safer rule is: keep all S34 validation-period `book_ticker`, `liquidations`, `mark_prices`, and `agg_trades` until N=100 and final reports are frozen.

Deleting `liquidations` older than 30 days should not break the live runner if no open trade points into that period, but it destroys retrospective S34/research ability for that period. During active pre-registration, do not delete data newer than the pre-reg start.

## 3. Other Files

| Path | Size | Code Reference Found? | Active Use Verdict | Delete/Archive Risk |
|---|---:|---|---|---|
| `logs/smoke_microstructure.db` | 16.96 GiB | no exact reference found | likely old smoke artifact | low, but take backup/move first |
| `data/lead_lag_work.db` | 1.92 GiB | no exact reference found | PR #9 work DB artifact | low, not used by runner |
| `localtests/` | 0.01 GiB total | tests create it | test artifacts | safe but tiny gain |
| `data/test_*.db` and old test DBs | 417 files, 0.012 GiB | test artifacts | not meaningful space | safe but tiny gain |
| `logs/archive/telemetry.20260310_010349.jsonl.gz` | 0.16 GiB | archive | not active | low |
| `logs/archive/execution_journal_20260219_065600.jsonl` | 0.12 GiB | archive | not active | low |
| `*.bak`, `_old`, `_backup`, `_copy` in logs/data | ~0.014 GiB excluding archives above | mostly test/archive | not meaningful space | safe but tiny gain |

The only immediate non-main-DB win is `logs/smoke_microstructure.db` plus `data/lead_lag_work.db`, about 18.9 GiB total. Everything else outside `microstructure.db` is too small to matter.

## 4. `agg_trades` Archive Scenario

Assumption requested: hot = last 7 days, cold = older than 7 days. Code says S34 only requires UTC day-so-far for `agg_trades`, so 7 days is already conservative for live forward test.

| Item | Estimate |
|---|---:|
| `agg_trades` total rows | ~326.9M |
| Estimated rows/day | ~2.7M over full coverage |
| Hot 7-day rows | ~19M |
| Cold rows to archive | ~308M |
| Approx reclaimable from `agg_trades` cold | ~20-33 GiB |
| INSERT SELECT archive time | rough 1-4 hours on SSD |
| DELETE cold rows time | rough 1-4+ hours |
| DB shrink after DELETE alone | none meaningful until VACUUM/rebuild |

Archiving `agg_trades` alone will not solve the disk issue because `book_ticker` is the dominant table. A 7-day `book_ticker` hot policy would be the real lever, but it must not cut through the current S34 validation period unless trade fill snapshots are fully materialized and accepted as sufficient for recompute.

### VACUUM vs Clean Rebuild

| Method | Pros | Cons | Runner Impact |
|---|---|---|---|
| DELETE cold rows + `VACUUM` | simple conceptually; keeps same DB path | needs roughly another DB-sized copy during vacuum; long exclusive lock; high interruption risk | stop runner for hours |
| Archive old rows + clean rebuild DB | can copy only hot data into a new DB; often safer than vacuuming 455 GiB | requires careful schema/index recreation and atomic swap; still needs validation | stop runner for final swap, likely shorter than full vacuum |
| Online archive only, no delete/vacuum | no immediate shrink risk; preserves data | no main DB space reduction | runner can continue, but lock pressure possible |

D: has ~1.02 TiB free, so one full-size rebuild/vacuum is feasible on paper. It is not riskless: with archives and WAL/temp files, keep at least 550-650 GiB free before attempting.

## 5. Three-Tier Cleanup Plan

### Kademe A - Risksiz, Hemen Yapılabilir

| Candidate | Est. Gain | Forward Test Impact | Action |
|---|---:|---|---|
| Move/delete `logs/smoke_microstructure.db` | 16.96 GiB | none found | safe after one manual confirmation |
| Move/delete `data/lead_lag_work.db` | 1.92 GiB | none for S34 | safe after one manual confirmation |
| Compress/move old `logs/archive/*` large files | 0.28 GiB | none | optional |
| Clean `localtests/` and `data/test_*.db` | ~0.02 GiB | none | not worth prioritizing |

Kademe A expected gain: ~18-19 GiB. This does not touch the live runner or `microstructure.db`.

### Kademe B - Dusuk Risk, Runner Calisirken Planlanabilir

| Candidate | Est. Gain After Final Shrink | Forward Test Impact | Risk |
|---|---:|---|---|
| Archive `agg_trades` older than 7 days | ~20-33 GiB | none for live S34 if current day retained | lock/I/O pressure during copy |
| Archive stale `detector_heartbeat` | ~20-40 GiB | none for S34; monitor impact unconfirmed | needs exact owner decision |
| Archive old non-validation `book_ticker` | potentially hundreds of GiB | dangerous if validation/recompute needs it | only after explicit validation-period cutoff |

Kademe B should start with archive-only copies, not deletes. Deleting while runner writes can work in WAL mode, but it increases lock and checkpoint risk; I would not do mass DELETE during an active high-value validation window.

### Kademe C - Dikkatli, Runner Kisa Sure Durdurulmali

| Candidate | Est. Gain | Runner Downtime | When |
|---|---:|---:|---|
| Clean rebuild keeping hot/validation data only | 100-350+ GiB depending bookTicker retention | hours | regime-dışı day, no open S34 position |
| DELETE + VACUUM after archive | same logical gain | many hours, worst lock profile | only if rebuild is rejected |

Kademe C is the only path that materially reduces `microstructure.db`. Since `freelist_count=0`, nothing shrinks until data is actually removed and the file is rebuilt/vacuumed.

## Recommendation

1. Do Kademe A first if you need quick space: ~19 GiB with no forward-test impact.
2. Do not start by archiving `agg_trades` expecting a huge win; it is probably only ~20-33 GiB.
3. The real cleanup PR should target `book_ticker` retention plus the stale `detector_heartbeat` table, but only after defining an S34 validation retention boundary.
4. Prefer clean rebuild over full `VACUUM` for Kademe C: copy schema + only retained rows into a new DB, validate counts/ranges, then stop runner for a controlled swap.

Estimated total gain:

| Tier | Gain |
|---|---:|
| A only | ~18-19 GiB |
| A + B without bookTicker | ~60-90 GiB |
| A + B + C with bookTicker retention policy | potentially 200-350+ GiB |

Final answer: Forward test is not affected by Kademe A. Kademe B archive-only is not supposed to affect it, but mass deletes should wait. Kademe C affects the forward test because the runner must be stopped for the final rebuild/swap; do it only on a regime-dışı window with no open S34 trade.
