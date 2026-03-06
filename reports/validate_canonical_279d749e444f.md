# CANONICAL_VALIDATION

status=pass
run_id=279d749e444f
source=reports\test_validate_canonical\pass_clean.csv
violations=0

## Violations

## Invariant Summary
- {'timestamp_min_utc': '1970-01-01T00:00:01+00:00', 'timestamp_max_utc': '1970-01-01T00:00:04+00:00', 'blank_symbol_count': 0, 'duplicate_timestamps_per_symbol': 0, 'backward_time_jumps': 0, 'negative_spread_count': 0, 'negative_volume_count': 0, 'non_positive_mid_count': 0, 'rows': 4, 'violations': 0}

## Run Summary
- {'version': 'v1', 'run_type': 'validate_canonical', 'inputs': {'source': 'reports\\test_validate_canonical\\pass_clean.csv', 'nan_threshold': 0.05, 'db': ''}, 'metrics': {'status': 'pass', 'violation_count': 0, 'row_count': 4}, 'artifacts': {'json': 'reports\\validate_canonical_279d749e444f.json', 'md': 'reports\\validate_canonical_279d749e444f.md'}}
