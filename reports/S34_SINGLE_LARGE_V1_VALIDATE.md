# S34 Single Large V1 Validate

- verdict: `VALIDATE_FORWARD`
- detector_rows: `73`

## Results

| group | h | N | WR | mean_bps | median_bps |
|---|---:|---:|---:|---:|---:|
| all | 60 | 73 | 60.27% | 4.07 | 5.62 |
| all | 120 | 73 | 54.79% | 2.00 | 2.19 |
| all | 300 | 73 | 63.01% | 2.48 | 4.26 |
| all | 900 | 73 | 60.27% | 5.55 | 13.47 |
| single_large | 60 | 21 | 85.71% | 8.02 | 12.35 |
| single_large | 120 | 21 | 80.95% | 10.78 | 13.46 |
| single_large | 300 | 21 | 85.71% | 12.26 | 11.56 |
| single_large | 900 | 21 | 80.95% | 17.37 | 19.72 |
| clustered | 60 | 44 | 45.45% | 0.01 | -1.14 |
| clustered | 120 | 44 | 36.36% | -5.47 | -6.08 |
| clustered | 300 | 44 | 45.45% | -7.66 | -1.98 |
| clustered | 900 | 44 | 47.73% | -4.54 | -2.76 |
| other_or_null | 60 | 8 | 75.00% | 15.96 | 13.97 |
| other_or_null | 120 | 8 | 87.50% | 20.01 | 16.47 |
| other_or_null | 300 | 8 | 100.00% | 32.62 | 25.65 |
| other_or_null | 900 | 8 | 75.00% | 30.02 | 14.76 |
| single_large_basis_pos | 60 | 10 | 80.00% | 10.00 | 13.23 |
| single_large_basis_pos | 120 | 10 | 90.00% | 11.97 | 13.50 |
| single_large_basis_pos | 300 | 10 | 100.00% | 15.74 | 11.16 |
| single_large_basis_pos | 900 | 10 | 90.00% | 31.78 | 30.85 |
| single_large_ofi_pos | 60 | 10 | 80.00% | 2.68 | 5.58 |
| single_large_ofi_pos | 120 | 10 | 70.00% | 3.91 | 3.91 |
| single_large_ofi_pos | 300 | 10 | 80.00% | 5.40 | 8.90 |
| single_large_ofi_pos | 900 | 10 | 70.00% | 5.36 | 11.27 |
| single_large_book_partial | 60 | 3 | 66.67% | 0.47 | 1.01 |
| single_large_book_partial | 120 | 3 | 100.00% | 2.58 | 2.44 |
| single_large_book_partial | 300 | 3 | 100.00% | 10.02 | 10.77 |
| single_large_book_partial | 900 | 3 | 66.67% | 4.75 | 12.77 |
| single_large_book_recovered | 60 | 1 | 0.00% | -45.57 | -45.57 |
| single_large_book_recovered | 120 | 1 | 0.00% | -39.40 | -39.40 |
| single_large_book_recovered | 300 | 1 | 0.00% | -33.89 | -33.89 |
| single_large_book_recovered | 900 | 1 | 0.00% | -59.75 | -59.75 |

## Best N>=5

`{'label': 'other_or_null', 'horizon_sec': 300, 'n': 8, 'wr': 100.0, 'mean_bps': 32.61671103358235, 'median_bps': 25.645524312798905}`