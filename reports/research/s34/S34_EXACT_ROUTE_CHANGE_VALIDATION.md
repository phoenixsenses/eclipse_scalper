# S34 Exact Route-Change Validation

Generated: `2026-06-26T18:03:47+00:00`

**Methodology**: runner-parity via direct _bucket_events/_paper_trade_from_signal/_evaluate_trade import

**Note**: No risk gates applied. All fillable signals evaluated independently per variant.

---

## Summary Table

| Candidate | Current | Alternative | Exact N | Current median | Alt median | Current cum | Alt cum | Giveback delta | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| ETH_1M_SELL | TP80/SL40/BE40 (current) | TP60/SL40/BE30 | 76 | -14.4 | -11.5 | -195.8 | -94.7 | -1 pp | **SWITCH_RECOMMENDED** |
| SOL_100K_SELL | TP60/SL30/BE40 (current) | TP40/SL40/BE40 | 85 | -38.4 | -17.4 | -877.5 | -953.6 | +4 pp | **ALT_MARGINALLY_BETTER** |

---

## ETH_1M_SELL — ETHUSDT SELL >= $1,000,000

Signals: 114  |  No-fill: 36 (32%)  |  Fillable: 78

| Variant | N | Median | Mean | Cum | Top3-rem | WR | Exit mix | Hold sec | GivebackN | Giveback% | H1 med | H2 med | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **TP80/SL40/BE40 (current)** | 76 | -14.4 | -2.6 | -195.8 | -537.7 | 28% | TP=19 BE=18 SL=29 TIME=10 | 1299.5 | 18 | 24% | -15.5 | -10.2 | CURRENT |
| TP60/SL40/BE40 | 76 | -12.0 | -2.6 | -199.6 | -422.0 | 37% | TP=26 BE=11 SL=29 TIME=10 | 1220.5 | 18 | 24% | -15.1 | +0.5 | INCONCLUSIVE_NO_MEANINGFUL_DIFFERENCE |
| TP70/SL40/BE40 | 76 | -13.8 | -3.4 | -261.3 | -555.6 | 30% | TP=21 BE=16 SL=29 TIME=10 | 1254.4 | 20 | 26% | -15.1 | -10.2 | ALT_MARGINALLY_BETTER |
| TP60/SL40/BE30 | 75 | -11.5 | -1.3 | -94.7 | -317.1 | 37% | TP=26 BE=16 SL=27 TIME=6 | 1071.3 | 17 | 23% | -14.5 | +0.5 | SWITCH_RECOMMENDED |

*H1 = first 50% of closed trades chronologically. H2 = second 50%.*
*Giveback = MFE >= 50% of TP but net_bps < 0. ⚠ = N < 30 (preliminary).*

---

## SOL_100K_SELL — SOLUSDT SELL >= $100,000

Signals: 106  |  No-fill: 20 (19%)  |  Fillable: 86

| Variant | N | Median | Mean | Cum | Top3-rem | WR | Exit mix | Hold sec | GivebackN | Giveback% | H1 med | H2 med | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **TP60/SL30/BE40 (current)** | 85 | -38.4 | -10.3 | -877.5 | -1129.5 | 28% | TP=23 BE=9 SL=47 TIME=6 | 923.8 | 12 | 14% | -28.0 | -38.4 | CURRENT |
| TP40/SL30/BE40 | 85 | -38.4 | -10.5 | -891.4 | -1084.4 | 39% | TP=32 SL=47 TIME=6 | 840.4 | 14 | 16% | -28.0 | -38.4 | INCONCLUSIVE_NO_MEANINGFUL_DIFFERENCE |
| TP40/SL40/BE40 | 85 | -17.4 | -11.2 | -953.6 | -1146.7 | 45% | TP=36 SL=40 TIME=9 | 1125.4 | 15 | 18% | -9.2 | -46.3 | ALT_MARGINALLY_BETTER |
| TP50/SL30/BE40 | 85 | -38.4 | -8.4 | -713.9 | -951.3 | 36% | TP=30 BE=2 SL=47 TIME=6 | 870.9 | 9 | 11% | -28.0 | -38.4 | KEEP_CURRENT |

*H1 = first 50% of closed trades chronologically. H2 = second 50%.*
*Giveback = MFE >= 50% of TP but net_bps < 0. ⚠ = N < 30 (preliminary).*

---

