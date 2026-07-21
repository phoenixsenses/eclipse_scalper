# S34 v8 Stop-Reliability Sizing

Generated: `2026-06-29T07:09:00.382973+00:00`

Mode: `RESEARCH_RISK_ONLY_NO_LIVE_CHANGE`

## 1. Unified Sizing

- p_atomic empirical / upper95: `0.182` / `0.477`
- p_gap-exceed empirical / upper95: `0.0` / `0.143`
- p_stop_fail empirical / conservative: `0.182` / `0.552`
- recommendation: `use conservative_weighted as single operational recommendation unless operator explicitly chooses tail_only`

| Basis | Loss bps | Max notional | Max margin @40x | Oversize vs env |
| --- | ---: | ---: | ---: | ---: |
| `stop_only_unreliable_floor` | 175.7 | $39.8 | $1.0 | 29.9x |
| `empirical_weighted` | 259.0 | $27.0 | $0.7 | 44.0x |
| `conservative_weighted` | 428.6 | $16.3 | $0.4 | 72.9x |
| `tail_only_hard_floor` | 634.0 | $11.0 | $0.3 | 107.8x |

## 2. Catastrophic Atomicity Scan

- status: `NO_CATASTROPHIC_GAP_FOUND`
- filled N: `23`
- worst adverse gap: `-18.5 bps`
- adverse <= -5bps N: `3`
- adverse <= -25bps N: `0`
- SL trigger inside gap N: `0`
- baseline-tail gap-start N: `0`
- read: No hit does not prove zero risk; it bounds observed history only.

## 3. Kill / Drawdown Simulation

- current env notional max DD: `$-17.2` = `49.1%` equity
- current env first trigger: `None`
- conservative-weighted notional max DD: `$-0.2` = `0.7%` equity
- tail probability before 5-trade kill window: `64.6%`
- read: Historical 23-row filled sequence does not trigger early enough to be a primary defense; first-tail risk is before kill.

## 4. Ledger Completeness

- source shadow rows: `11`
- management rows: `11`
- unique source ids: `11`
- duplicates: `0`
- complete against shadow ledger: `True`

## Final Read

The single honest sizing recommendation is conservative_weighted; tail-only remains the hard floor if operator wants maximum survival.
