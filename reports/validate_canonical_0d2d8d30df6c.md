# CANONICAL_VALIDATION

status=fail
run_id=0d2d8d30df6c
source=reports\test_validate_canonical\fail_nan.csv
violations=1

## Violations
- {'type': 'nan', 'code': 'nan_ratio_above_threshold', 'column': 'mid', 'severity': 'critical', 'nan_ratio': 0.5, 'threshold': 0.25}

## Invariant Summary
- {'duplicate_timestamps_per_symbol': 0, 'backward_time_jumps': 0, 'negative_spread_count': 0, 'negative_volume_count': 0, 'non_positive_mid_count': 0, 'rows': 4, 'violations': 1}
