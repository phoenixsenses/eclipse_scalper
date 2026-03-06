# CANONICAL_VALIDATION

status=fail
run_id=0d2d8d30df6c
source=reports\test_validate_canonical\fail_nan.csv
violations=1

## Violations
- {'type': 'nan', 'code': 'nan_ratio_above_threshold', 'column': 'mid', 'severity': 'critical', 'nan_ratio': 0.5, 'threshold': 0.25}

## Invariant Summary
- {'timestamp_min_utc': '1970-01-01T00:00:01+00:00', 'timestamp_max_utc': '1970-01-01T00:00:04+00:00', 'blank_symbol_count': 0, 'duplicate_timestamps_per_symbol': 0, 'backward_time_jumps': 0, 'negative_spread_count': 0, 'negative_volume_count': 0, 'non_positive_mid_count': 0, 'rows': 4, 'violations': 1}

## Run Summary
- {'version': 'v1', 'run_type': 'validate_canonical', 'inputs': {'source': 'reports\\test_validate_canonical\\fail_nan.csv', 'nan_threshold': 0.25, 'db': ''}, 'metrics': {'status': 'fail', 'violation_count': 1, 'row_count': 4}, 'artifacts': {'json': 'reports\\validate_canonical_0d2d8d30df6c.json', 'md': 'reports\\validate_canonical_0d2d8d30df6c.md'}}
