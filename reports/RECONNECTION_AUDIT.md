# Reconnection Audit Report

- Generated UTC: `2026-03-04T11:36:53.669793+00:00`
- Heartbeat path: `logs\collector_heartbeat.json`
- Status: `degraded`
- Connected: `1`
- Backend: `websockets`
- Current backoff sec: `1.00`
- Last message ts: `2026-03-04T11:36:11.246292+00:00`
- WAL size MB: `0.00`

## Findings
- Reconnection logic uses exponential backoff + jitter in collector.
- Collector resets reconnect delay after stable connection window.
- Last collector error: `connection_error:TimeoutError: timed out during opening handshake`

## Recommendations
- Keep `stall_timeout_sec` finite to force reconnect on silent WS stalls.
- Use supervisor with restart cap to avoid crash loops.
- If ISP intermittently blocks Binance WS, configure VPN/SOCKS5 path and validate DNS/TLS.
- Monitor `current_backoff_seconds` and repeated errors via heartbeat + Telegram alerts.
