# CANONICAL_VALIDATION

status=fail
run_id=517afbab5a7d
source=reports\test_validate_canonical\fail_missing_col.csv
violations=1

## Violations
- {'type': 'schema', 'code': 'missing_symbol_col', 'severity': 'critical'}

## Invariant Summary
- {'timestamp_min_utc': '1970-01-01T00:00:01+00:00', 'timestamp_max_utc': '1970-01-01T00:00:02+00:00', 'non_positive_mid_count': 0, 'rows': 2, 'violations': 1}

## Run Summary
- {'version': 'v1', 'run_type': 'validate_canonical', 'inputs': {'source': 'reports\\test_validate_canonical\\fail_missing_col.csv', 'nan_threshold': 0.05, 'db': ''}, 'metrics': {'status': 'fail', 'violation_count': 1, 'row_count': 2}, 'artifacts': {'json': 'reports\\validate_canonical_517afbab5a7d.json', 'md': 'reports\\validate_canonical_517afbab5a7d.md'}}
