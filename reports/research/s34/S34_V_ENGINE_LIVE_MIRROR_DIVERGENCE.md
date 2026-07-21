# S34 V Engine Live/Mirror Divergence

Generated: `2026-06-30T14:00:20.860943+00:00`

Read-only diagnostic. No live state, order logic, config, or executor process is changed.

## Latest Explanation

- reason: `LIVE_BLOCKED_BY_no_fresh_eligible_signal`
- mirror signal: `2026-06-30T13:32:16.371000+00:00`
- mirror status: `DATA_INCOMPLETE` / `NO_EXIT_BOOK`
- mirror decision: `OBSERVE_ONLY_NO_ORDER`
- live active status: `None`
- live blocked by: `no_fresh_eligible_signal`

## Latest Mirror

```json
{
  "decision": "OBSERVE_ONLY_NO_ORDER",
  "entry_price": "1553.48912092884",
  "exit_utc": "",
  "maker_fill_utc": "2026-06-30T13:34:26+00:00",
  "net_bps": "",
  "notes": "paper_shadow_mirror_only_no_order",
  "observation_id": "981f158bf5b81227843cae9c",
  "observation_status": "DATA_INCOMPLETE",
  "signal_utc": "2026-06-30T13:32:16.371000+00:00",
  "sim_status": "NO_EXIT_BOOK"
}
```

## Live Diagnostic

```json
{
  "active": null,
  "active_status": null,
  "last_missed_signal": null,
  "last_signal_scan": {
    "anchors_reconstructed": 1,
    "eligible_fresh_candidates": 0,
    "lookback_sec": 1800,
    "max_book_staleness_sec": 10,
    "reject_counts": {
      "book": 0,
      "mark": 0,
      "prior4h": 0,
      "too_old": 1,
      "vdepth": 0
    },
    "scan_ts_utc": "2026-06-30T14:00:03.027449+00:00",
    "signal_fresh_sec": 60
  },
  "new_entry_blocked_by": "no_fresh_eligible_signal",
  "orders_count": 0,
  "reconciliation": {
    "position_amount": 0.0,
    "s34ve_open_client_ids": [],
    "s34ve_open_order_count": 0,
    "updated_at_utc": "2026-06-30T14:00:03.027088+00:00"
  }
}
```
