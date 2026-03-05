# CANONICAL_VALIDATION

status=fail
run_id=97d28bbeea09
source=reports\test_validate_canonical\fail_dup.csv
violations=1

## Violations
- {'type': 'time', 'code': 'duplicate_timestamp_per_symbol', 'severity': 'critical', 'count': 1}

## Invariant Summary
- {'duplicate_timestamps_per_symbol': 1, 'backward_time_jumps': 0, 'negative_spread_count': 0, 'negative_volume_count': 0, 'non_positive_mid_count': 0, 'rows': 3, 'violations': 1}
