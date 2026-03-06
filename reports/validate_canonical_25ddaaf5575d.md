# CANONICAL_VALIDATION

status=fail
run_id=25ddaaf5575d
source=reports\test_validate_canonical\fail_empty.csv
violations=1

## Violations
- {'type': 'schema', 'code': 'empty_dataframe', 'severity': 'critical'}

## Invariant Summary
- {'blank_symbol_count': 0, 'negative_spread_count': 0, 'negative_volume_count': 0, 'non_positive_mid_count': 0, 'rows': 0, 'violations': 1}

## Run Summary
- {'version': 'v1', 'run_type': 'validate_canonical', 'inputs': {'source': 'reports\\test_validate_canonical\\fail_empty.csv', 'nan_threshold': 0.05, 'db': ''}, 'metrics': {'status': 'fail', 'violation_count': 1, 'row_count': 0}, 'artifacts': {'json': 'reports\\validate_canonical_25ddaaf5575d.json', 'md': 'reports\\validate_canonical_25ddaaf5575d.md'}}
