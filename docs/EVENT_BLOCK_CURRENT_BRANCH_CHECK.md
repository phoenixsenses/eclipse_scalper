# EVENT_BLOCK CURRENT BRANCH CHECK

Date:
- 2026-03-08

Branch:
- `codex/research/post-merge-sweep`

Objective:
- verify whether the current branch/code state still supports the previously documented ETH event-block candidates on the live `FILTER_SWEEP_PASSIVE_REALISTIC_ETH_TOP8.md` surface

Profiles checked:
- `baseline`
- `event_block_v1`
- `event_block_eth_micro_imb05_v1`
- `event_block_eth_micro_imb085_v1`

Input:
- `reports/FILTER_SWEEP_PASSIVE_REALISTIC_ETH_TOP8.md`
- `data/microstructure.db`

Data-source check:
- the research worktree local DB is **not** the same as the live root repo DB
- research worktree DB:
  - `C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-research\data\microstructure.db`
  - size ~= `45 KB`
- live/root repo DB:
  - `C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper\data\microstructure.db`
  - size ~= `8.89 GB`
- implication:
  - the first current-branch `no_fills` result was not a reliable event-filter decision by itself
  - it was initially confounded by a data-source mismatch / stale-local-DB problem

Root-DB rerun:
- reran the same ETH TOP8 surface against the live/root DB:
  - `C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper\data\microstructure.db`
- explicit baseline result:
  - `reports/ROOTDB_BASELINE_EXPLICIT_ETH_TOP8.json`
  - `reports/ROOTDB_BASELINE_EXPLICIT_ETH_TOP8.md`
- result stayed the same:
  - `count = 0`
  - all 8 pockets skipped
  - dominant failures:
    - `insufficient_fill_rate = 1.0`
    - `attempt_fill_rate = 0.0` on the remaining pockets

Fresh current-surface rebuild:
- instead of relying on the stale historical ETH TOP8 markdown, rebuilt the current 1D ETH passive-realistic surface from the live/root DB:
  - `reports/FILTER_SWEEP_PASSIVE_REALISTIC_ETH_FRESH_TOP8.md`
- result:
  - `rows = 24`
  - `pass = 0`
  - top 8 current ETH pockets were all `NO`
  - even the best current rows had:
    - negative `avg_net`
    - `insuf% = 100%`
- implication:
  - the current 1D ETH passive-realistic surface itself is non-actionable
  - this is stronger evidence than comparing against the stale historical TOP8 file

7D note:
- attempted a broad 7D fresh ETH passive-realistic rebuild from the live/root DB
- the generic sweep timed out before completion
- conclusion:
  - 7D broad evaluation needs a narrower/optimized runner
  - but this does not change the 1D conclusion above

Focused 7D pocket probes:
- to avoid the broad 7D sweep timeout, ran direct 7D pocket validation on two historical TOP8 ETH pockets against the live/root DB:
  - pocket A:
    - `h=120`
    - `imb>=0.30`
    - `int>=8000`
    - `spr<=0.000200`
    - artifact: `reports/FOCUSED_ETH_POCKET_A_7D.json`
  - pocket B:
    - `h=60`
    - `imb>=0.30`
    - `int>=8000`
    - `spr<=0.000200`
    - artifact: `reports/FOCUSED_ETH_POCKET_B_7D.json`
- with the default `min_n=50` both failed due to insufficient fills
- then reran both with `min_n=20`:
  - pocket A still failed
    - artifact: `reports/FOCUSED_ETH_POCKET_A_7D_MIN20.json`
    - pass_count = `0`
  - pocket B partially opened
    - artifact: `reports/FOCUSED_ETH_POCKET_B_7D_MIN20.json`
    - pass_count = `3/6`
    - pass_rate = `0.5`
- implication:
  - the 7D ETH surface is not uniformly dead
  - it is capacity-threshold sensitive
  - at least one narrow ETH 60s pocket can become actionable under a softer capacity requirement

Focused 7D event-filter comparison on the viable 60s pocket:
- pocket used:
  - `ETHUSDT`
  - `h=60`
  - `imb>=0.30`
  - `int>=8000`
  - `spr<=0.000200`
  - `min_n=20`
- baseline:
  - artifact: `reports/FOCUSED_ETH_POCKET_B_7D_MIN20.json`
  - `pass_count = 3/6`
  - `pass_rate = 0.5`
  - `insufficient_fill_rate = 0.5`
- book-proxy-only block:
  - artifact: `reports/FOCUSED_ETH_POCKET_B_7D_BOOKBLOCK.json`
  - `pass_count = 3/6`
  - `pass_rate = 0.5`
  - `insufficient_fill_rate = 0.5`
  - but split-level fills improved materially in the strong split (e.g. `filled_n` rose from `21` to `29`)
- two-lane block (`book_proxy_pressure + volatility_burst`):
  - artifact: `reports/FOCUSED_ETH_POCKET_B_7D_BLOCKV1.json`
  - `pass_count = 0/6`
  - `insufficient_fill_rate = 1.0`
- implication:
  - on the current branch, the lighter single-lane negative filter is still plausible
  - the two-lane block is too aggressive on this viable 7D ETH pocket

Adjacent 7D pocket check (same horizon, slightly wider spread):
- pocket C:
  - `ETHUSDT`
  - `h=60`
  - `imb>=0.30`
  - `int>=8000`
  - `spr<=0.000250`
  - `min_n=20`
- baseline:
  - artifact: `reports/FOCUSED_ETH_POCKET_C_7D_MIN20.json`
  - `pass_count = 1/6`
  - `pass_rate = 0.1667`
- book-proxy-only block:
  - artifact: `reports/FOCUSED_ETH_POCKET_C_7D_BOOKBLOCK.json`
  - `pass_count = 0/6`
  - `pass_rate = 0.0`
- implication:
  - `book_proxy_pressure`-only blocking is not generically helpful across nearby ETH 60s pockets
  - it may still help on the tighter `spr<=0.000200` pocket
  - but it currently fails the adjacent-pocket robustness check

Interpretation update:
- the stale local DB was real and had to be ruled out
- however, even after rerunning on the live/root DB, the current branch still yields a non-tradeable ETH TOP8 passive-realistic surface
- therefore the problem is no longer just data-source mismatch
- the likely remaining causes are current-branch model / surface drift, especially in:
  - `execution/passive_execution_simulator.py`
  - `tools/validate_passive_pocket_forward.py`
  - `tools/rank_passive_pockets_forward.py`

Current-code result:
- all four runs produced `count = 0`
- all eight ETH pockets ended with `failure_reason_top = no_fills`
- effective `attempt_fill_rate = 0.00%`

Artifacts:
- `reports/CURR_BASELINE_ETH_TOP8.json`
- `reports/CURR_V1_ETH_TOP8.json`
- `reports/CURR_IMB05_ETH_TOP8.json`
- `reports/CURR_IMB085_ETH_TOP8.json`

Relaxed probe:
- reran `baseline` and `event_block_eth_micro_imb05_v1` with softer gates:
  - `splits = 2`
  - `min_n = 5`
  - `min_attempt_fill_rate = 0`
  - `max_insufficient_fill_rate = 1`
  - `maker_fee_bps_grid = 0`
- result still unchanged:
  - all pockets `no_fills`
  - no usable common set for filter comparison

Artifacts:
- `reports/CURR_BASELINE_ETH_TOP8_RELAX.json`
- `reports/CURR_IMB05_ETH_TOP8_RELAX.json`

Interpretation:
- this is not currently a filter-ranking problem
- on the current branch, even the live/root DB rerun produces a tradeable coverage failure on the ETH TOP8 passive-realistic surface
- therefore none of the event-block profiles can be meaningfully compared on this snapshot

Practical decision:
- keep the older decision document as historical research evidence
- do not promote any event-block profile from the current branch snapshot
- treat the current branch state as:
  - `surface_non_actionable_on_current_branch`
  - until current-branch execution/validation drift is reconciled
  - and until a fresh actionable surface is re-established
- more precise refinement after focused 7D probes:
  - `1D broad ETH surface` = non-actionable
  - `7D focused ETH pockets` = threshold-sensitive, with at least one viable 60s pocket under softer `min_n`
  - `book_proxy_pressure`-only filter = pocket-sensitive, not yet robust enough for broad rollout

Next sensible step:
1. do not use `C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-research\data\microstructure.db` for promotion decisions
2. treat the root-DB rerun as the authoritative current-branch result:
   - the ETH TOP8 passive-realistic surface is currently non-actionable
3. treat the fresh 1D ETH rebuild as confirming that current ETH passive-realistic pockets are not passing even before event filters
4. for 7D work, stop using broad generic sweep as the only decision source; use targeted pocket probes where needed
5. do not promote `book_proxy_pressure`-only filtering as an ETH-wide 60s rule from current evidence
6. if continuing this line, scope any further test to the tight `spr<=0.000200` pocket family only
7. next debugging target should be current-branch drift / surface weakness in passive execution or candidate generation on the dead 1D surface

Book-proxy-only broad current-branch check:
- broader current-branch runs were also checked with `event_block_book_proxy_v1`
- artifacts:
  - `reports/EVENT_BLOCK_SET_BOOK_PROXY_V1_REAL.json`
  - `reports/EVENT_BLOCK_SET_BOOK_PROXY_V1_REAL_1D.json`
  - `reports/EVENT_BLOCK_SET_BOOK_PROXY_V1_FULL_REAL.json`
  - `reports/RANK_EVENT_FILTER_SET_SUMMARY_BOOK_PROXY_V1_7D.json`
- result:
  - current broad 1D / broad-current surfaces still ranked `0`
  - the useful signal only appeared in the older narrow 7D probe family
  - even there the set summary was:
    - `common_count = 8`
    - `improved_count = 4`
    - `degraded_count = 4`
    - `median_delta_npa_core = -1.019758e-05`
    - `median_filtered_kept_ratio = 94.82%`
- interpretation:
  - `event_block_book_proxy_v1` is too weak to be a rollout candidate on the current branch
  - it preserves coverage well, but quality improvement is not robust
  - on current evidence it should be treated as a diagnostic / pocket-level research knob, not as a production-facing mitigation profile

Current freeze decision:
- do not promote:
  - `event_block_v1`
  - `event_block_book_proxy_v1`
  - `event_block_eth_*`
  from the current branch snapshot
- treat the whole event-block family on this branch as:
  - `research_only_until_surface_recovers`
- preferred next move:
  - stop broad event-filter testing here
  - focus on why the current ETH passive-realistic surface itself is weak

Limit-offset sensitivity check:
- to test whether the weak current surface is mainly caused by passive limit placement, reran focused ETH pockets with:
  - default `limit_offset_mult = 0.5`
  - tighter `limit_offset_mult = 0.25`
- pocket B (`h=60`, `imb>=0.30`, `int>=8000`, `spr<=0.000200`, 7D, `min_n=20`):
  - default rank result:
    - artifact: `reports/LIMIT_OFFSET_POCKET_B_DEFAULT.json`
    - skipped with `insufficient_fill_rate = 0.6`
  - tighter rank result:
    - artifact: `reports/LIMIT_OFFSET_POCKET_B_TIGHT.json`
    - also skipped with `insufficient_fill_rate = 0.6`
  - implication:
    - tighter offset did not rescue this tighter pocket at rank level
- pocket C (`h=60`, `imb>=0.30`, `int>=8000`, `spr<=0.000250`, 7D, `min_n=20`):
  - default rank result:
    - artifact: `reports/LIMIT_OFFSET_POCKET_C_DEFAULT.json`
    - `npa = -1.344515e-04`
    - `pass_core = 10%`
    - `pass_stress = 0%`
    - `afr = 68.63%`
  - tighter rank result:
    - artifact: `reports/LIMIT_OFFSET_POCKET_C_TIGHT.json`
    - `npa = -1.329405e-04`
    - `pass_core = 10%`
    - `pass_stress = 10%`
    - `afr = 69.91%`
  - implication:
    - tighter offset gives only a marginal improvement
    - it does not turn the pocket positive or robust

Execution-side conclusion:
- current ETH surface weakness is not primarily explained by the passive limit offset alone
- `limit_offset_mult` can slightly improve a marginal pocket
- but it is not the main unlock for the current non-actionable broad ETH surface

Execution-shape sensitivity on the viable 7D focused ETH pocket:
- reran the same focused pocket with exact comparable settings:
  - `ETHUSDT`
  - `h=60`
  - `imb>=0.30`
  - `int>=8000`
  - `spr<=0.000200`
  - `lookback=7D`
  - `splits=2`
  - `min_n=20`
  - `seeds=11,22,33`
- baseline passive-realistic:
  - artifact: `reports/ETH_POCKET_B_7D_BASELINE_SPLIT2.json`
  - `rows_total = 3`
  - `pass_count = 1`
  - `pass_rate = 33.33%`
  - `attempt_fill_rate_mean = 79.97%`
  - `filled_avg_net_mean = -2.043730e-04`
- passive wait only (`passive_realistic`, `passive_max_wait_buckets=2`):
  - artifact: `reports/ETH_POCKET_B_7D_WAIT2.json`
  - `rows_total = 3`
  - `pass_count = 0`
  - `pass_rate = 0.00%`
  - `insufficient_fill_rate = 100.00%`
  - implication:
    - waiting longer without changing exit style made fills worse, not better
- passive-then-taker (`exec_model=passive_then_taker`, `passive_max_wait_buckets=2`):
  - artifact: `reports/ETH_POCKET_B_7D_PASSIVE_THEN_TAKER.json`
  - `rows_total = 3`
  - `pass_count = 3`
  - `pass_rate = 100.00%`
  - `insufficient_fill_rate = 0.00%`
  - all three rows:
    - `attempt_fill_rate = 100%`
    - `filled_n = 28`
    - positive `filled_avg_net`
  - implication:
    - for this current-branch pocket, the main bottleneck is not event context or limit offset
    - the main unlock is execution shape: pure passive is too brittle, passive-then-taker rescues the pocket

Updated current-branch diagnosis:
- broad ETH 1D surface is still non-actionable
- event-block profiles are still frozen on this branch
- but one focused 7D ETH pocket now shows a clear next direction:
  - investigate `passive_then_taker`
  - not broader event blocking
