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

Next sensible step:
1. do not use `C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-research\data\microstructure.db` for promotion decisions
2. treat the root-DB rerun as the authoritative current-branch result:
   - the ETH TOP8 passive-realistic surface is currently non-actionable
3. next debugging target should be current-branch drift in passive execution / pocket validation, not more event-block profile tuning
