# LIQUIDATION REVERSAL E2E

symbol=ETHUSDT rule=high_liq_reversal_regime lookback_min=1440 bucket_sec=5

coverage_windows=1 max_rule_fire_count=12 max_rule_given_liq_rate=50.00%
candidate_count=8
baseline_rank_count=0
v5_rank_count=0
v6_rank_count=1
next_step=inspect_ranked_pockets

## Top Results

- baseline_top=None
- v5_top=None
- v6_top={'symbol': 'ETHUSDT', 'rule': 'high_liq_reversal_regime', 'horizon_sec': 60, 'score': 0.0001, 'score_raw_core': 0.0002, 'npa_core': 3e-05, 'pass_rate_core': 0.5, 'attempt_fill_rate': 0.25, 'failure_reason_top': 'mixed'}

## Artifacts
- coverage_json=reports\test_run_liq_reversal_e2e\LIQ_REVERSAL_E2E_COVERAGE.json
- candidates_json=reports\test_run_liq_reversal_e2e\LIQ_REVERSAL_E2E_CANDIDATES.json
- rank_baseline_json=reports\test_run_liq_reversal_e2e\LIQ_REVERSAL_E2E_RANK_BASELINE.json
- rank_v5_json=reports\test_run_liq_reversal_e2e\LIQ_REVERSAL_E2E_RANK_V5.json
- rank_v6_json=reports\test_run_liq_reversal_e2e\LIQ_REVERSAL_E2E_RANK_V6.json
