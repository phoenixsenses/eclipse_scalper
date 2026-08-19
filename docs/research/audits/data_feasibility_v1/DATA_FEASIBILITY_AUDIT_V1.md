# Eclipse Scalper / E-DER — Data Feasibility Audit V1

**Audit date:** 2026-08-19  
**Mode:** read-only, outcome-blind data-feasibility audit  
**Status:** `COMPLETE — STOPPED BEFORE MODELING`  
**Canonical Bible SHA-256:** `6BA44FEB5942018AD413ECF22DC9C361E17AA459D3F84061A9106E73B475182D`

No existing database, table, collector, trading/execution file, configuration, frozen E-DER
definition, historical event file, or existing research report was modified. The only writes are
new files in this audit directory and the authorized byte-identical Bible copy at
`docs/research/ECLIPSE_RESEARCH_BIBLE.md`.

## 1. Executive conclusion

The stored historical data can scientifically support the following, with the stated limits:

- **VERIFIED FROM CODE/DATA:** all 25 frozen E-DER identities have a complete 241-row, one-minute
  event-to-fixed-boundary OHLCV path, the corresponding BTC path, and observed force-order rows.
- **VERIFIED FROM CODE/DATA:** the liquidation variable is `forceOrder o.p * o.q`, where `q` is
  original order quantity. It is an **observed forced-liquidation pressure proxy**, not complete
  executed liquidation volume.
- **PROXY ONLY:** historical Flow Surprise can be defined only on that force-order pressure proxy.
  Historical Impact Surprise can be defined only on one-minute OHLCV/OPEN-type prices, not true
  quote mid.
- **UNSUPPORTED for the frozen 25:** aggressive-trade flow, quote mid, spread, L1 depth, multi-level
  depth, classical OFI, MLOFI and displayed-book recovery. The retained aggTrade and bookTicker
  archives contain only BTCUSDT, ETHUSDT and SOLUSDT; none of the 12 E-DER event symbols is covered.
- **NOT IDENTIFIABLE:** exact additions, cancellations, exact replenishment, hidden liquidity, true
  executed liquidation volume, and historical tick-regime conditioning.

The audit therefore permits a narrowly named, outcome-blind **liquidation-pressure-proxy Flow
Surprise** and **one-minute-OHLCV Impact Surprise** measurement contract. It does not permit a
historical LOB absorption claim, exact replenishment claim, OFI/MLOFI analysis, or dynamic-exit
experiment.

## 2. Canonical research understanding

1. The current question is whether adverse forced SELL pressure becomes progressively less
   price-effective after E-DER LONG entry, consistent with a conditional response anomaly rather
   than automatically with thesis deterioration.
2. Channel A rejected its frozen ordering: renewed post-entry SELL did not separate outcomes in the
   expected direction. It did **not** establish that SELL liquidation is bullish, that absorption
   occurred, that the relationship is causal, or that an exit should be reversed.
3. The 25 events were already observed before the new compression mechanism was formulated. They
   are therefore exploratory/post-hoc for that mechanism; confirmation requires new forward events.
4. Flow Surprise is abnormal observed forced flow relative to a pre-event expectation; Impact
   Surprise is abnormal price response conditional on generic market expectations; Response Path is
   their time-ordered joint evolution; Measurement Sensitivity asks whether conclusions survive
   defensible measurement choices; Mechanism Evidence is a compatible observable pattern;
   Mechanism Identification rules out observationally equivalent explanations.
5. Frozen population/timing, no return-driven feature or threshold selection, no random CV, no
   in-sample residuals, no future variables in states/matching, trade/event as independent unit,
   explicit multiplicity control, and `SUPPORTED / NOT SUPPORTED / NOT IDENTIFIABLE` language are
   binding.
6. Data Feasibility Audit V1 is the stage gate because a formula is not a valid feature until its
   source fields, clocks, coverage and identification limits are proven.

## 3. Evidence-status convention

| Label | Meaning in this report |
|---|---|
| `VERIFIED FROM CODE/DATA` | Directly established from stored schema/rows, manifests, version history or collector code. |
| `INFERRED` | Best explanation supported by indirect evidence; not encoded sufficiently to prove row by row. |
| `UNKNOWN / NOT RECOVERABLE` | Missing source payload/version/clock information prevents a historical ruling. |

The observability matrix uses the separate Bible vocabulary: `VALID`, `PROXY ONLY`, `UNSUPPORTED`,
and `NOT IDENTIFIABLE`.

## 4. Repository and data inventory

### 4.1 Canonical/primary assets

| Asset | Relevant contents | Census / coverage | Finding |
|---|---|---|---|
| `data/microstructure_02.db` | Current live SQLite: liquidation, aggTrade, mark price, bookTicker and operational tables | 157,465,911,296 bytes. SQLite max assigned IDs: liquidation 892,356; aggTrade 39,873,579; mark 6,770,345; bookTicker 1,074,737,334. These are **not exact row counts**. | `VERIFIED` sizes/schema; counts `INFERRED` only |
| `data/keeper_frozen_smalltables.db` | Pre-rotation small tables, including liquidation | Liquidations: **1,722,645** rows, **761** symbols, `1771165818195..1784839238819` | `VERIFIED FROM CODE/DATA` |
| `data/xsec_klines.db` | One-minute OHLCV and ingest receipts | Klines: **11,790,940** rows, **109** symbols, `1777593600000..1784937540000` | `VERIFIED FROM CODE/DATA` |
| `data/archives/parquet_v1/agg_trades` | Pre-cutoff legacy aggregate trades | **427,185,688** rows, 415 partitions, BTC/ETH/SOL, 2026-02-15..07-23 | `VERIFIED FROM CODE/DATA` |
| `data/archives/parquet_v1/book_ticker` | Pre-cutoff top-of-book ticks | **5,723,357,020** rows, 305 partitions, BTC/ETH/SOL, 2026-04-11..07-23 | `VERIFIED FROM CODE/DATA` |
| `data/archives/parquet_v1/mark_prices` | Pre-cutoff mark price/funding updates | **24,441,427** rows, 415 partitions, BTC/ETH/SOL, 2026-02-15..07-23 | `VERIFIED FROM CODE/DATA` |
| `reports/research/s34/mechanism_store.sqlite` | Derived ETH event/control feature store | 836 rows; not raw market data | `VERIFIED FROM CODE/DATA` |
| `reports/research/s34/S34_ALL.db` | Research result/ledger consolidation | 28,960 result rows; not raw market data | `VERIFIED FROM CODE/DATA` |
| `data/funding_history.db` | Funding, futures/spot 1h, limited OI history | 20,218 funding; 161,129 futures 1h; 161,862 spot 1h; 1,500 OI | `VERIFIED FROM CODE/DATA` |
| `data/oi_history.db` | OI and futures 5m history | 37,259 OI; 25,919 futures 5m | `VERIFIED FROM CODE/DATA` |

`data/microstructure.db` is currently a zero-byte/reclaimed placeholder path and must not be treated
as the historical database. `data/rotation_state.json` points readers to `microstructure_02.db` and
the keeper segment. The pre-cutoff large tables exist in the verified Parquet archive, not in the
keeper.

Complete schemas and counts are exported in `schema_inventory.csv`, `database_inventory.csv` and
`archive_inventory.csv`.

### 4.2 Archive integrity and duplicate evidence

- **VERIFIED FROM CODE/DATA:** the accepted rotation-ordering receipt covers 6,726,613,400 rows in
  1,191 partitions with zero physical ordering and physical-ID violations.
- That receipt proves physical archive identity/order, **not semantic feed completeness**.
- **VERIFIED FROM CODE/DATA:** keeper liquidation has 186 economic-key duplicate groups / 186 excess
  rows over its full 172-day extent. The frozen S34 support preflight separately rejects economic
  duplicates inside its bounded support interval; therefore these full-keeper duplicates do not
  silently enter the frozen event reconstruction.
- **VERIFIED FROM CODE/DATA:** all 1,722,645 keeper liquidation rows satisfy stored
  `notional == price * quantity` within the audit tolerance.
- **VERIFIED FROM CODE/DATA:** `xsec_klines` uses `(symbol, open_time)` as a composite primary key, so
  duplicate identities are structurally prevented.
- **UNKNOWN:** semantic duplicate rate in the active live aggTrade/bookTicker tables was not
  established. A disruptive billion-row scan was intentionally not run against the live DB.

### 4.3 Missing intervals and outages

| Feed | Evidence | Finding |
|---|---|---|
| Liquidations | No rows for 2026-04-28..06-05 (39 full UTC days) | `VERIFIED`; first major outage |
| Liquidations | 2026-07-06 10:06:39Z..07-10 11:24:38Z; 07-07/08/09 have zero rows | `VERIFIED`; routed-endpoint regression |
| bookTicker | BTC/ETH/SOL partitions 2026-06-06..06-10 each contain zero rows | `VERIFIED`; five-day outage |
| aggTrade | Nine zero-row symbol/day partitions in May/June | `VERIFIED`; exact list in `archive_inventory.csv` |
| Mark price | No zero-row daily archive partitions | `VERIFIED`; this does not prove sub-day continuity |
| Collector `gaps` | 741 liquidation, 20 aggTrade, 51 mark-price records | `VERIFIED`, but registry ceased being complete and cannot prove absence of unlogged gaps |

The price supplement registry has 977 `COMPLETE` partitions with 3,823,200/3,823,200 expected rows
and 15 `INCOMPLETE` partitions missing 252,065 minutes. Each frozen E-DER event window itself has all
241 required event-symbol bars and all 241 BTC bars.

## 5. Trade-feed audit

### 5.1 What exists

**VERIFIED FROM CODE/DATA:** the historical microstructure feed stores Binance Futures `aggTrade`,
not raw individual trades. No raw-trade table or raw-trade WebSocket stream was found.

Stored columns are:

`id, ts_ms, symbol, price, quantity, notional, is_buyer_maker`.

Current collector mapping:

| Stored field | Payload / transformation | Ruling |
|---|---|---|
| `ts_ms` | `T` trade time | `VERIFIED` |
| `symbol` | `s` | `VERIFIED` |
| `price` | `p` | `VERIFIED` |
| `quantity` | `q` | `VERIFIED` aggregate quantity |
| `notional` | `p*q` | `VERIFIED` |
| `is_buyer_maker` | `m` | `VERIFIED`; 0 → taker BUY, 1 → taker SELL |

The collector discards WebSocket event time `E`, aggregate trade ID `a`, first/last trade IDs `f/l`,
and has no local receive timestamp, quote quantity, `nq`, or RPI field.

### 5.2 Aggregation/source limits

- One `aggTrade` row must **not** be interpreted as one taker order.
- From 2026-07-03 onward the collector can use public REST aggTrades during WebSocket stalls. It uses
  `a` in volatile in-memory cursor state, then discards it before insert. Stored rows contain no
  `source=WS|REST` flag and no aggregate ID.
- **UNKNOWN / NOT RECOVERABLE:** a stored historical row cannot be assigned definitively to WS or
  REST after fallback existed; cross-restart ID continuity cannot be reconstructed from the legacy
  table.
- **UNKNOWN / NOT RECOVERABLE:** exact historical Binance upstream aggregation semantics cannot be
  proven from the legacy stored columns alone. The repository's repaired REST schema preserves
  `a/f/l`, but it is a separate later repair surface and does not retroactively enrich the E-DER
  event symbols.

### 5.3 E-DER relevance

The archive covers BTC/ETH/SOL only. None of ZEC, BANK, NEAR, VELVET, SPCX, ESPORTS, XAG, RE, AAOI,
XRP, KORU or SKHYNIX has historical aggTrade rows in the retained archive. Historical E-DER
aggressive buy/sell notional and minute signed flow are therefore `UNSUPPORTED`.

## 6. Order-book audit

### 6.1 Exact collected product

**VERIFIED FROM CODE/DATA:** `data/bookticker_collector.py` subscribes to
`<symbol>@bookTicker`. It stores only:

`ts_ms, symbol, bid_price, bid_qty, bid_depth_usd, ask_price, ask_qty, mid_price, spread_pct,
book_imbalance`.

Derived fields are:

- `mid_price = (bid_price + ask_price) / 2`
- `spread_pct = (ask_price - bid_price) / mid_price`
- `book_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)`
- `bid_depth_usd = bid_price * bid_qty`

The timestamp is `E`, otherwise `T`, otherwise local wall-clock fallback. The row does not record
which branch was used.

### 6.2 What is absent

- No partial-depth or diff-depth payloads.
- No multi-level bids/asks.
- No `U/u/pu` sequence fields or persisted update ID.
- No snapshot bootstrap/reconstruction procedure.
- No order-book gap detector or sequence recovery proof.
- No local receive timestamp.
- No exact additions/cancellations.

The collector retries database locks, but if a flush remains locked it can retain pending rows and
later trim the oldest rows once the 50,000-row cap is exceeded. There is no per-row missing marker.
Reconnections do not reconstruct a book because only top-of-book snapshots are collected.

### 6.3 Resolution class

Where available, historical book data is **Research Bible Level C**: top-of-book snapshots only.
It is not Level B (multi-level snapshots) and not Level A (incremental L2). For the 25 E-DER events
it is not merely low-resolution; it is absent because the 12 symbols are outside BTC/ETH/SOL.

Consequences:

- L1 spread/mid/depth for E-DER: `UNSUPPORTED`.
- Multi-level depth/MLOFI: `UNSUPPORTED`.
- Exact additions, cancellations and replenishment: `NOT IDENTIFIABLE`.
- Hidden liquidity: `NOT IDENTIFIABLE` even with displayed-book data.

## 7. Liquidation audit

### 7.1 Feed-regime timeline

The historical liquidation architecture has two non-comparable regimes:

1. **Pre-2026-06-06 17:43:52.123Z:** per-symbol forceOrder, effectively 2–3 configured symbols.
2. **From 2026-06-06 17:43:52.123Z:** all-market `!forceOrder@arr`.

The transition timestamp is measured from the first post-blackout row plus 171 distinct symbols in
the following hour. All 25 E-DER events begin on or after 2026-06-07, so they belong to the
all-market regime. The code change was only committed later; repository incident reports preserve
the actual deployment date.

### 7.2 Payload-to-storage semantics

| Stored column | Source | Meaning |
|---|---|---|
| `ts_ms` | top-level `E` | exchange event time |
| `symbol` | `o.s` | liquidation-order symbol |
| `side` | `o.S` | order side |
| `price` | `o.p` | order price |
| `quantity` | `o.q` | original order quantity |
| `notional` | `o.p * o.q` | nominal original-order pressure proxy |
| `trade_time_ms` | `o.T`, fallback `E` | order trade time |

The parser sees examples containing `ap`, `l`, `z` and `X` but discards them. It also discards raw
JSON, order type/time-in-force, and any local receive timestamp.

In the keeper, 1,722,630 of 1,722,645 rows have `E != T`; the observed `E-T` difference among those
rows is 1..10,825 ms. Therefore the clocks are not interchangeable.

### 7.3 Valid interpretation

The all-market stream is documented in the repository's prior semantics audit as a 1000 ms,
per-symbol latest-order snapshot. It is throttled/snapshot-like and does not deliver a complete
liquidation tape. Thus:

> The E-DER liquidation variable measures observed, throttled forced-order snapshots valued at
> order price times original order quantity.

It is **not** true executed liquidation volume. Because `ap/l/z/X` and raw payloads are absent, that
executed quantity is `NOT IDENTIFIABLE` historically.

### 7.4 `q_parent` and `q_echo`

**VERIFIED FROM CODE/DATA:** in the S34 impact-elasticity code:

1. SELL force-order rows are reconstructed into frozen anchors; anchor `running_notional` sums the
   stored `p*q` pressure proxy.
2. `trailing_quote_volume` sums exact one-minute kline quote volume over 15/30/60-minute windows.
3. `q_parent = parent.running_notional / parent trailing quote volume`.
4. `q_echo = echo.running_notional / echo trailing quote volume`.

Therefore neither variable is executed liquidation share. Both are normalized observed
forced-liquidation pressure proxies.

## 8. Symbol-metadata audit

- No tick size, step size, price precision or quantity precision exists in the microstructure,
  keeper or xsec-kline schemas.
- The frozen universe artifact stores a 2026-07-25 `onboardDate`-derived listing-age snapshot and
  explicitly warns that it came from that day's `exchangeInfo`.
- No historical exchange-filter change archive was found.
- Current `exchangeInfo` values cannot be substituted for historical event-time filters.

Ruling:

- Listing date: partially reconstructable only from dated snapshot artifacts, `PROXY ONLY`.
- Historical tick/step/precision and contract/filter changes: `UNKNOWN / NOT RECOVERABLE` from the
  stored E-DER data.
- Tick-regime conditioning for the frozen 25: `NOT IDENTIFIABLE`.

## 9. Timestamp and alignment audit

| Source | Exchange clock stored | Receive clock stored | Boundary rule |
|---|---|---|---|
| forceOrder | `E` plus `o.T` | No | Research generally buckets by `E=ts_ms` |
| aggTrade | `T` only | No | Query windows use trade time |
| markPrice | `E` (REST fallback may use API `time`/local fallback) | No | No source flag |
| bookTicker | `E`, else `T`, else local fallback | No | Branch not recorded |
| one-minute kline | exact UTC `open_time` | No | `(symbol,open_time)` PK |

E-DER time construction is explicit:

- `base_ms = floor(anchor_ts_ms / 60s) * 60s + 60s`, the next one-minute OPEN;
- entry = `base + 31m` OPEN;
- fixed boundary = `base + 240m` OPEN;
- event path = base minute 0 through boundary minute 240 inclusive;
- entry-relative hold is 209 minutes;
- an observation outcome begins at the next exact OPEN.

The frozen 25 manifest exactly matches these rules and every required OPEN is present.

However, without collector receive time and with mixed/fallback clock rules, a row stored at a
similar timestamp cannot prove the exact market state faced by the flow. This is especially material
for matching forceOrder `E` to quote state: historical E-DER quotes are absent, and even where
bookTicker exists its `E/T/local` branch is not recorded.

All times are epoch milliseconds and converted as UTC in research code. No local-time bucketing was
found in the frozen E-DER path.

## 10. Derived-feature provenance

| Field | Source → transformation | Scientific ruling |
|---|---|---|
| liquidation rate/notional | forceOrder snapshots → count or sum `p*q` / window | pressure proxy only |
| `q_parent`, `q_echo` | anchor running `p*q` / trailing kline quote volume | normalized pressure proxy only |
| signed aggTrade flow | `m=0` BUY, `m=1` SELL → signed `p*q` | valid aggTrade imbalance where covered |
| `fl_*_ofi` | `(BUY notional-SELL notional)/(BUY+SELL)` | **not classical OFI**; name is ambiguous |
| book mid/spread/imbalance | L1 bid/ask snapshot formulas | valid Level-C fields where covered |
| `bk_pull` | min pre-1m L1 bid qty / avg pre-10m L1 bid qty | sampled L1 ratio; not cancellations |
| `bk_refill` | avg post-5m L1 bid qty / avg pre-10m L1 bid qty | sampled L1 ratio; not exact replenishment |
| `fl_*_impact` | absolute mark return / total aggTrade notional in millions | price-per-aggregate-flow proxy |
| `px_rv` | root sum of squared one-minute mark-price returns | sampled volatility proxy |

`mechanism_store.sqlite` is a derived ETH-only research table, not a raw source. Its fields cannot
fill the missing E-DER event-symbol archive. Full provenance is in `feature_provenance.csv`.

## 11. Frozen 25-event observability matrix

Legend: `V=VALID`, `P=PROXY ONLY`, `U=UNSUPPORTED`, `NI=NOT IDENTIFIABLE`.

All 25 events have 241/241 event-symbol one-minute bars, 241/241 BTC bars, and at least one
event-window liquidation-proxy row. None has event-symbol aggTrade or bookTicker history.

| Event | Liq | Agg flow | Raw trades | Mid/L1 | Multi-L | OFI | MLOFI | Add/cancel | Flow S. | Impact S. | BTC adj. | Tick | Sens. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E:ZECUSDT:1780845457333 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:BANKUSDT:1780855795565 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:NEARUSDT:1780862453188 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:VELVETUSDT:1781136529140 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:VELVETUSDT:1781213436253 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:VELVETUSDT:1781443683377 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:ZECUSDT:1781619444366 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:SPCXUSDT:1781630527127 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:ESPORTSUSDT:1781753866102 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:XAGUSDT:1781809363726 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:REUSDT:1781850730196 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:REUSDT:1781890124567 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:REUSDT:1781961178284 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:REUSDT:1781993071245 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:REUSDT:1782047077639 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:SPCXUSDT:1782237239120 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:AAOIUSDT:1782241674063 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:XRPUSDT:1782394329538 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:AAOIUSDT:1783013941004 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:XRPUSDT:1783230138080 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:AAOIUSDT:1784266556188 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:KORUUSDT:1784271145642 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:BANKUSDT:1784293554509 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:SKHYNIXUSDT:1784325168738 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |
| E:BANKUSDT:1784466726162 | P | U | U | U | U | U | U | NI | P | P | V | NI | P |

`Flow S.` is `P` only for the forceOrder pressure-proxy definition. An aggressive-trade Flow
Surprise is `U`. `Impact S.` is `P` only for exact one-minute OHLCV/OPEN; quote-mid Impact Surprise
is `U`. The unabridged matrix and event-specific source counts are in
`historical_e_der_observability_matrix.csv` and `event_data_coverage.csv`.

## 12. Feed-semantic regime timeline

| Date | Change | Ruling |
|---|---|---|
| 2026-02-15 | Earliest retained liquidation/aggTrade/mark history | `VERIFIED` |
| 2026-02-20 | microstructure collector first visible in Git | `VERIFIED` |
| 2026-04-11 | BTC/ETH bookTicker archive begins | `VERIFIED` |
| 2026-04-18 | SOL joins aggTrade/bookTicker/mark archive | `VERIFIED` |
| 2026-04-28..06-05 | liquidation blackout | `VERIFIED` |
| 2026-06-06 17:43:52.123Z | per-symbol → all-market forceOrder transition | `VERIFIED` from data/incident contract |
| 2026-06-06..06-10 | bookTicker zero-row partitions | `VERIFIED` |
| 2026-07-03 | REST fallback and routed-WS correction committed | `VERIFIED` |
| 2026-07-06..07-10 | second liquidation outage | `VERIFIED` |
| 2026-07-21 | bookTicker collector first visible in Git | `VERIFIED`; earlier executable provenance not committed |
| 2026-07-23 20:40:46.344Z | SQLite rotation cutoff | `VERIFIED` |
| 2026-08-06 | large frozen DB reclaimed; Parquet + keeper reader path activated | `VERIFIED` |

No per-row collector version, stream mode, transport source or schema version is stored. Therefore:

- exact row-by-row attribution to collector version is `UNKNOWN / NOT RECOVERABLE`;
- current Binance documentation must not be projected backward as proof of every historical
  upstream semantic detail;
- the E-DER support is at least consistently after the measured all-market transition, which avoids
  mixing the earlier narrow subscription regime into these 25 events.

## 13. Historical result reconciliation

This was an isolated provenance check performed **after** all outcome-blind observability
classifications were constructed.

| Version | Contract | Mean | Median |
|---|---|---:|---:|
| Earlier S34/Channel A | same entry OPEN → same fixed-boundary OPEN, minus 10 bps cost | 305.0963027350 | 124.3599546530 |
| Latest frozen outcome | same entry OPEN → same fixed-boundary OPEN, gross | 315.0963027350 | 134.3599546530 |

**VERIFIED FROM CODE/DATA:** both files contain the same 25 event IDs. For every event individually,
`latest gross - earlier net = exactly 10.0 bps`. The earlier `build_event` computes the same log
return then subtracts `COST_BPS=10`; the newer frozen exporter intentionally exports gross return.

The discrepancy is therefore fully explained by cost treatment. It is not an event-population,
entry-alignment, exit-alignment, price-source or rounding change. Neither historical value is
replaced: one is net under the hypothetical 10 bps cost and one is gross.

Event-level proof is in `result_reconciliation_event_level.csv` and summary proof in
`result_reconciliation_summary.json`.

## 14. Claim-support table

| Claim | Ruling | Reason |
|---|---|---|
| Observed forced-liquidation pressure around all 25 events | Potentially `SUPPORTED/testable` | E/S/p/q/T stored; event windows covered |
| Complete executed liquidation volume | `NOT IDENTIFIABLE` | throttled snapshot; fill fields discarded |
| OHLCV response anomaly vs chronological generic benchmark | Potentially `SUPPORTED/testable` | exact 1m price panel; explicitly a proxy |
| Quote-mid marginal impact | Testable but currently `NOT SUPPORTED` | event-symbol quotes absent |
| Aggressive-flow absorption | Testable but currently `NOT SUPPORTED` | event-symbol aggTrades absent |
| Displayed LOB resilience, OFI or MLOFI | Testable but currently `NOT SUPPORTED` | event-symbol LOB absent |
| Exact add/cancel/replenishment | `NOT IDENTIFIABLE` | no incremental L2 sequence |
| Hidden-liquidity absorption | `NOT IDENTIFIABLE` | hidden orders are not in stored displayed data |
| Historical tick-regime effect | `NOT IDENTIFIABLE` | historical filter changes absent |
| Dynamic-exit benefit/mechanism | Not assessed and not authorized | audit is outcome-blind feasibility only |

## 15. STOP / GO stage gate

| Item | Decision | Exact scope |
|---|---|---|
| Flow Surprise V1 — forceOrder pressure proxy | **GO** | `PROXY ONLY`; measurement choices frozen by semantics/stability, never returns |
| Flow Surprise V1 — aggressive-trade definition | **STOP** | historical event-symbol aggTrades absent |
| Impact Surprise V1 — exact 1m OHLCV definition | **GO** | `PROXY ONLY`; never call OHLC4/OPEN a quote mid |
| Impact Surprise V1 — quote-mid definition | **STOP** | historical event-symbol quotes absent |
| OFI | **STOP** | no event-symbol book updates; `fl_ofi` is not OFI |
| MLOFI | **STOP** | no multi-level book |
| Displayed-book resilience | **STOP** | no event-symbol book data |
| Exact replenishment | **STOP** | not identifiable from snapshots; event feed absent |
| Hidden-liquidity claims | **STOP** | not identifiable |
| Dynamic-exit research | **STOP** | no forward-confirmed mechanism; audit does not authorize exit work |

## 16. Known measurement limitations

1. forceOrder is a lossy, throttled pressure proxy and can saturate in cascades.
2. Original quantity/order price are stored; executed quantity/average fill are not.
3. E-DER events are all in the all-market regime, but that does not make the feed complete.
4. No receive timestamps allow latency or exact flow-versus-book ordering reconstruction.
5. Event-symbol microstructure is absent; candles cannot substitute for LOB or aggressor flow.
6. OHLCV prices identify within-minute ranges and OPENs, not the contemporaneous quote state.
7. Existing `fl_ofi`, `bk_pull` and `bk_refill` names are stronger than their stored semantics.
8. The gap registry is incomplete; explicit zero-row periods are evidence of outages, while silence
   outside them is not proof of completeness.
9. Historical tick/filter metadata is absent.
10. The 25 events are exploratory/post-hoc for the new mechanism and cannot confirm it.

## 17. Exact next recommended research step — not executed

Create and independently review an **outcome-blind E-DER Measurement Sensitivity Contract V1**
before any new mechanism diagnostic.

It should freeze, using semantics and measurement stability only:

1. forceOrder pressure representations `sum(p*q)`, `sum(q)` and observed-message count;
2. separate `E`-time and `T`-time bucketing, with disagreement reported rather than optimized;
3. one-minute OPEN/close/range response definitions explicitly labeled as OHLCV proxies;
4. source-quality rejection for windows crossing measured cadence gaps/outages;
5. the existing chronological generic-market OOS benchmark and BTC adjustment without refitting on
   E-DER returns;
6. a prospective collector contract for future E-DER symbols that preserves raw forceOrder fields,
   aggTrade `a/f/l`, exchange and receive timestamps, source/version IDs, historical exchange
   filters, and sequence-valid incremental L2 if OFI/MLOFI/replenishment is to be studied.

Only after that contract is frozen should a historical descriptive proxy diagnostic be run, clearly
labeled exploratory, while genuine confirmation waits for new forward E-DER events. No dynamic exit
should be designed or tested at this stage.

## 18. Audit artifacts

- `audit_evidence_export.py` — audit-only read-only exporter
- `audit_evidence.json` — machine-readable audit summary
- `database_inventory.csv`
- `schema_inventory.csv`
- `archive_inventory.csv`
- `collector_gap_inventory.csv`
- `data_quality_findings.csv`
- `event_data_coverage.csv`
- `historical_e_der_observability_matrix.csv`
- `feature_provenance.csv`
- `feed_semantic_timeline.csv`
- `claim_identifiability.csv`
- `go_stop_matrix.csv`
- `result_reconciliation_event_level.csv`
- `result_reconciliation_summary.json`
- `AUDIT_ARTIFACT_MANIFEST_SHA256.csv`

The report stops here. No next-stage model, feature, threshold, exit, or trading rule was created.
