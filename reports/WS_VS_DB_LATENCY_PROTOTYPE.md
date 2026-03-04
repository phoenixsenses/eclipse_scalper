# WS vs DB Latency Prototype

- symbol: `ETHUSDT`
- collector_connected: `1`
- collector_progress_lag_sec: `None`
- db_lag_sec: `2.91`
- estimated_ws_bypass_gain_sec: `None`

## Interpretation
- If `db_lag_sec` is consistently > 2s, feature staleness is materially high for microstructure triggers.
- If estimated gain is large, prioritize direct-WS feature pipeline prototype.
