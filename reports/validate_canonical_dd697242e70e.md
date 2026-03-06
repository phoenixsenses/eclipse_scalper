# CANONICAL_VALIDATION

status=fail
run_id=dd697242e70e
source=reports\test_validate_canonical\fail_blank_symbol.csv
violations=1

## Violations
- {'type': 'schema', 'code': 'blank_symbol', 'severity': 'critical', 'count': 2}

## Invariant Summary
- {'timestamp_min_utc': '1970-01-01T00:00:01+00:00', 'timestamp_max_utc': '1970-01-01T00:00:03+00:00', 'blank_symbol_count': 2, 'duplicate_timestamps_per_symbol': 0, 'backward_time_jumps': 0, 'negative_spread_count': 0, 'negative_volume_count': 0, 'non_positive_mid_count': 0, 'rows': 3, 'violations': 1}

## Run Summary
- {'version': 'v1', 'run_type': 'validate_canonical', 'inputs': {'source': 'reports\\test_validate_canonical\\fail_blank_symbol.csv', 'nan_threshold': 0.05, 'db': ''}, 'metrics': {'status': 'fail', 'violation_count': 1, 'row_count': 3}, 'artifacts': {'json': 'reports\\validate_canonical_dd697242e70e.json', 'md': 'reports\\validate_canonical_dd697242e70e.md'}}
