# LIQUIDATION SOURCE-QUALITY COVERAGE CONTRACT RECONCILIATION — READ-ONLY (2026-07-05)

**Batch:** AMI birth-truncated cascade geometry — source-quality methodology divergence resolution.
**Mode:** READ-ONLY (canonical.sqlite mode=ro + hash-verified unchanged; microstructure.db mode=ro; no outcome read; no migration; no experiment write).
**Script:** `tools/research_s34_source_quality_reconciliation.py` (deterministic; regenerates the per-signal table).
**Per-signal table (220 rows):** `S34_LIQUIDATION_SOURCE_QUALITY_RECONCILIATION_2026-07-05.json` (this directory) — includes both methods, contract-v2, window bounds, months, cycles, touching gap rows, per-window health metrics.

## GOAL A — both methods reconstructed EXACTLY

| Method | Rule | COMPLETE | GAPPED | UNRESOLVED |
|---|---|---|---|---|
| METHOD_A (original audit) | resolved-gap overlap → GAPPED; else `birth >= FIRST open-ended liq-gap start` (2026-04-02 17:58:38.989) → UNRESOLVED; else COMPLETE | **83** | **1** | **136** |
| METHOD_B (rehearsal) | resolved-gap overlap → GAPPED; else `birth >= LAST liq-gap-row start` (2026-04-27 14:27:24.680) → UNRESOLVED; else COMPLETE | **125** | **1** | **94** |

Both reproduce the historical counts bit-exactly. Disagreement set = **exactly 42 signals, all April 2026** (birth in (Apr 2 17:58, Apr 27 14:27)), 28 distinct independent cycles, every one A=UNRESOLVED / B=COMPLETE. The single GAPPED signal is identical in both: `SIG-291486458a347231dcb60109` (2026-04-05 06:13:18, overlaps resolved gap id=114, 06:10:13→06:12:40).

**Neither method uses positive evidence** — both infer completeness from ABSENCE of registry rows, differing only in where they stop trusting the registry's silence.

## GOAL B — what the gap registry can and cannot prove

Writer code recovered from git stash `07e1a1f9` (June 1; `test_collector_stream_gaps.py` — the writer itself was deleted from the live codebase):

- Gap = staleness heartbeat: stream silent >120s while connected → row (`resolved_bool=0, end_ts_ms=NULL, duration_sec`=staleness at detect); >300s → stream-specific reconnect; data resumption → `_resolve_gap` sets `resolved_bool=1, end_ts_ms`.
- **Resolution state was in-memory** — any collector crash/restart orphans open rows forever. `resolved_bool=0 + end_ts_ms=NULL` = "gap opened, resumption never recorded", NOT "gap still open".
- The **current** collector (`data/microstructure_collector.py`) has **no gap code at all** — the registry is dead. Last rows: liquidations 2026-04-27, agg_trades 2026-04-24, mark_prices 2026-05-28.
- Registry demonstrably unreliable in BOTH directions:
  - **False flags:** the 21 open-ended liquidations rows have empirical silences of ~0.1–2s (next liquidation row arrives immediately) while claiming 70–431543s staleness at detect — clock/restart artifacts.
  - **Missed real losses:** the liquidations table has a **40.1-day total blackout (2026-04-27 14:27:26 → 2026-06-06 17:43:52; all of May = 0 rows across all symbols)** plus multi-hour April holes (Apr 24: 12.3h; Apr 27: 7.5h; Apr 23: 6.9h — only the last was a resolved registry row). The registry recorded none of this as resolved evidence.

**Conclusion: NO GAP RECORD ≠ SOURCE_COMPLETE, in any era.** Both METHOD_A and METHOD_B rest on this invalid equivalence (METHOD_A additionally trusts Feb–Mar, an era where the registry did not yet exist — its first row ever is 2026-04-02).

## GOAL C — independent completeness evidence

- **Cross-stream health is INVALID as liquidation-completeness proof:** in May 2026 the shared combined websocket delivered 15.27M ETHUSDT agg_trades rows (29 days) and 260k mark_prices rows (31 days) while the liquidations stream delivered **zero rows** — liquidations-stream-only silent failure for weeks is proven. Same during the Apr 24 12.3h hole (232k agg_trades rows flowed through it).
- **Stream-mode discovery:** Feb–Mar liquidations contain only 2 symbols, April 3 symbols (**per-symbol `@forceOrder` subscriptions**); Jun 6+ contains **733 symbols** (**all-market `!forceOrder@arr`**, current collector). Data resumed 2026-06-06 17:43:52.
- Consequence: for the per-symbol era (Feb 15–Apr 27), natural per-symbol silences of minutes–hours make cadence-based per-window verification impossible → **no positive completeness evidence is achievable for ANY Feb–Apr signal**. For the all-market era (Jun 6+), the all-market inter-arrival cadence is dense and healthy (1.13M rows; 27 holes ≥120s in a month; max 818s) → per-window stream-specific positive verification IS achievable.

## GOAL D — proposed fail-closed contract (`liq-source-quality-contract-v2`)

1. Window overlaps a RESOLVED registry gap → `SOURCE_GAPPED` (frozen precedent; 1 signal).
2. `birth < 2026-06-06 17:43:52` (per-symbol era / blackout) → `SOURCE_COVERAGE_UNRESOLVED` — stream-specific completeness structurally unprovable.
3. All-market era: max all-market liquidation inter-arrival over `[window_start − 1800s, birth]` ≤ **300s** (the old collector's own frozen critical-gap constant; 1800s = frozen MIN_GAP_SEC) → `SOURCE_COMPLETE`; else `SOURCE_COVERAGE_UNRESOLVED`.

Thresholds are pre-existing frozen collector/protocol constants — not fit to any population or outcome. Per the operator's own instruction, the 42 disputed signals **cannot be positively proven complete → UNRESOLVED** under this contract.

## GOAL E — research readiness under each candidate

| Contract | COMPLETE | complete-only cycles (TRAIN/TEST) | MIN_BUCKET_N=20 |
|---|---|---|---|
| METHOD_A | 83 (Feb 24/Mar 55/Apr 4) | 59 (41/18) | **FAIL** (test=18) |
| METHOD_B | 125 (Feb 24/Mar 55/Apr 46) | 87 (60/27) | pass — but absence-of-evidence-based |
| STANDARD-2 (cross-stream health) | 204 | 133 (93/40) | pass — **INVALIDATED** by the May liquidations-only-failure evidence |
| **CONTRACT-V2 (proposed)** | **93** (Jun 90/Jul 3) | **54 (37/17)** | **FAIL** (test=17) |

Both semantically defensible candidates (A and V2) fail MIN_BUCKET_N. The only passing candidates rest on invalid absence-of-evidence reasoning.

**Research-readiness verdict: `BLOCKED_BY_SOURCE_QUALITY_CONTRACT`** (operator must freeze the contract; under CONTRACT-V2 the follow-on verdict would be `GEOMETRY_INFERENTIAL_RESEARCH_BLOCKED_BY_SOURCE_QUALITY`). This does not block canonical storage of correctly flagged rows.

## GOAL F — test-ground-truth reconciliation

- **751/751** = `pytest tests/test_ami_*.py tests/test_buyfade_mutations.py tests/test_buyfade_silexit_mutations.py --collect-only` — 61 files (59 AMI + 2 buyfade mutation suites: 711+24+16). Reproduced exactly: the same set collects **783 today = 751 + 32** new geometry tests (both new files ran green this session).
- **2611** = `pytest tests/ --ignore=tests/legacy_tools --collect-only` — the FULL test tree (S34 v-engine/executor/shadow/dashboard/…), a much broader scope that was never the AMI regression ground truth. Not parametrization — scope.
- **3 errors** = collection errors confined to `tests/legacy_tools/` (an `IndentationError` inside a legacy test file itself + duplicate-basename import mismatches with `tests/execution/`), pre-existing, environment/structural, unrelated to AMI.
- **Proposed frozen canonical command** (not yet adopted — requires one fully-green run first, per operator instruction): the 751-set command above + the 2 geometry test files → expected **783**. TEST_STATUS ground truth remains 751/751 until then.

## GOAL G — document status

`SYSTEM_STATE.md` §86'nın research-readiness satırı düzeltildi → **PROVISIONAL — `BLOCKED_BY_SOURCE_QUALITY_CONTRACT`**; §87 bu reconciliation'ı kaydediyor. Ledger + TEST_STATUS güncellendi.

## INTEGRITY

canonical.sqlite sha256 `c2b0b300…3098f` unchanged (verified before/after); microstructure.db mode=ro only; no migration; no experiment write; no outcome read; protected delta ZERO.

**WAIT_FOR_OPERATOR_APPROVAL** — freeze CONTRACT-V2 (or provide the standard you prefer), and approve/reject the proposed frozen test command.
