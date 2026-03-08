# Research Book Proxy Pressure Lane

## Purpose

`book_proxy_pressure` detects one-sided pressure using current trade-plus-mark data, without claiming real order-book depth.

## Why This Exists

The collector does not currently persist true top-of-book depth.

That means a real `book_imbalance_stress` lane would be misleading today.

This proxy lane instead uses:

- imbalance proxy from derived buckets
- spread
- trade intensity
- short-horizon return stability

## Constraint

This is explicitly a proxy lane.

It should be replaced or recalibrated once true L1 book data exists:

- `best_bid`
- `best_ask`
- `bid_qty`
- `ask_qty`
