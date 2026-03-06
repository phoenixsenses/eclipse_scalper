# CANONICAL_VALIDATION

status=fail
run_id=97d28bbeea09
source=reports\test_validate_canonical\fail_dup.csv
violations=1

## Violations
- {'type': 'time', 'code': 'duplicate_timestamp_per_symbol', 'severity': 'critical', 'count': 1}

## Invariant Summary
- {'timestamp_min_utc': '1970-01-01T00:00:01+00:00', 'timestamp_max_utc': '1970-01-01T00:00:02+00:00', 'blank_symbol_count': 0, 'duplicate_timestamps_per_symbol': 1, 'backward_time_jumps': 0, 'negative_spread_count': 0, 'negative_volume_count': 0, 'non_positive_mid_count': 0, 'rows': 3, 'violations': 1}

## Run Summary
- {'version': 'v1', 'run_type': 'validate_canonical', 'inputs': {'source': 'reports\\test_validate_canonical\\fail_dup.csv', 'nan_threshold': 0.05, 'db': ''}, 'metrics': {'status': 'fail', 'violation_count': 1, 'row_count': 3}, 'artifacts': {'json': 'reports\\validate_canonical_97d28bbeea09.json', 'md': 'reports\\validate_canonical_97d28bbeea09.md'}}
