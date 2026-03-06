# Tooling Audit

## Summary
- tool_count: 211
- script_count: 5
- run_summary_gap_count: 0
- dev_only_count: 12

## Status Counts
- core: 35
- dev-only: 12
- legacy: 2
- support: 162

## Family Counts
- builder: 12
- dev: 14
- misc: 96
- ops: 13
- reporting: 16
- research: 18
- runner: 10
- runtime: 26
- validation: 6

## Cut Candidates
- tools/data/test_build_canonical_dataset_smoke.py
- tools/data/test_execution_journal_parse_smoke.py
- tools/dev/_inspect_forward_json.py
- tools/dev/_inspect_rank_evals.py
- tools/dev/_inspect_rank_evals2.py
- tools/dev/_inspect_rank_json.py
- tools/dev/_inspect_rank_net2.py
- tools/dev/_inspect_rank_scores2.py
- tools/dev/_print_adv_sweep.py
- tools/dev/_print_fee_sweep.py
- tools/dev/_print_fee_sweep_newmetrics.py
- tools/dev/_summarize_gate_md.py

## Excluded Utilities
- `tools/build_presentation.py` is treated as a presentation/document artifact builder, not a report-contract tool.
- `tools/write_risk_policy_doc.py` is treated as a policy/doc export helper, not a report-contract tool.

## Run Summary Gaps
- none
