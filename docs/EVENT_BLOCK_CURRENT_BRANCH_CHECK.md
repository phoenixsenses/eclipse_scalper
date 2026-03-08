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
  - the current `no_fills` result is not a reliable event-filter decision by itself
  - it is primarily a data-source mismatch / stale-local-DB problem

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
- on the research worktree local DB, it appears as a current-surface tradeable coverage failure
- but the stronger root cause is that the run used a tiny stale local DB instead of the live root DB snapshot
- therefore none of the event-block profiles can be meaningfully compared on this snapshot

Practical decision:
- keep the older decision document as historical research evidence
- do not promote any event-block profile from the current branch snapshot
- treat the current branch state as:
  - `data_source_invalid_for_decision`
  - until runs are repeated against the live/root DB or the research DB is resynced

Next sensible step:
1. do not use `C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-research\data\microstructure.db` for promotion decisions
2. rerun the same ETH TOP8 checks against:
   - `C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper\data\microstructure.db`
3. only after that:
   - continue event-block comparison
   - or declare the surface genuinely non-actionable
