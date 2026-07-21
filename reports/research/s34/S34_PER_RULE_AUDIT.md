# S34 Active Rule Bucket Refinement Audit

Generated: `2026-06-27T10:50:05.265229+00:00`

Per-rule validation of pooled findings. **No runner/config changes.**
Tests: day_trend >4%, UTC 20-24 weakness, cascade size, single-liq dominance.

Pooled findings to validate:
1. day_trend > 4% hurts BUY routes
2. UTC 20-24 underperforms
3. Cascade < 200K is bad
4. ETH_BUY_50K is a drag (diagnosis only — confirm not in active allow list)

## ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30

**N=19  Median=+52.3  WR=79%  Cum=+743.3  Top3R=+856.5**

### Day Trend Bins

| Trend | N | Median | Cum | Top3R | WR | Note |
| --- | --- | --- | --- | --- | --- | --- |
| <0% | 0 |  |  |  |  |  |
| 0–1% | 10 | 52.6 | 427.3 | 482.3 | 80% |  |
| 1–2% | 4 | 54.4 | 218.0 | 59.3 | 100% | (thin) |
| 2–4% | 5 | 50.6 | 98.0 | 111.7 | 60% |  |
| >=4% | 0 |  |  |  |  |  |

### UTC Session Bins

| Session (UTC) | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| 00-04 | 4 | 23.4 | 57.3 | 50% | (thin) |
| 04-08 | 3 | 51.7 | 111.6 | 67% | (thin) |
| 08-12 | 2 | 69.2 | 138.3 | 100% | (thin) |
| 12-16 | 8 | 52.4 | 333.2 | 88% |  |
| 16-20 | 2 | 51.4 | 102.8 | 100% | (thin) |
| 20-24 | 0 |  |  |  |  |

### Cascade Notional Bins

| Cascade | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| <100K | 0 |  |  |  |  |
| 100K-200K | 0 |  |  |  |  |
| 200K-500K | 0 |  |  |  |  |
| 500K-1000K | 5 | 53.5 | 241.5 | 80% |  |
| >1000K | 14 | 52.0 | 501.7 | 79% |  |

### Max Single Liq Share Bins

| Single Share | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| <50% | 5 | 52.3 | 241.5 | 80% |  |
| 50–80% | 9 | 50.6 | 312.0 | 78% |  |
| >=80% | 5 | 52.5 | 189.8 | 80% |  |

### Candidate Filter Tests

| Filter | N (kept) | N (removed) | Median | Cum | Top3R | WR | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| max_day_trend <= 4% | 19 | 0 | 52.3 | 743.3 | 856.5 | 79% | N<20 — too thin |
| max_day_trend <= 3% | 18 | 1 | 52.1 | 690.9 | 804.1 | 78% | N<20 — too thin |
| exclude UTC 20-24 | 19 | 0 | 52.3 | 743.3 | 856.5 | 79% | N<20 — too thin |
| min_liq_count >= 3 | 19 | 0 | 52.3 | 743.3 | 856.5 | 79% | N<20 — too thin |
| min_liq_count >= 5 | 19 | 0 | 52.3 | 743.3 | 856.5 | 79% | N<20 — too thin |
| max_single_share <= 80% | 14 | 5 | 52.0 | 553.5 | 627.4 | 79% | N<20 — too thin |
| cascade >= 200K | 19 | 0 | 52.3 | 743.3 | 856.5 | 79% | N<20 — too thin |
| max_trend<=4% + UTC<20 | 19 | 0 | 52.3 | 743.3 | 856.5 | 79% | N<20 — too thin |

## SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30

**N=24  Median=+48.5  WR=62%  Cum=+640.4  Top3R=+821.8**

### Day Trend Bins

| Trend | N | Median | Cum | Top3R | WR | Note |
| --- | --- | --- | --- | --- | --- | --- |
| <0% | 6 | 21.6 | 134.7 | 216.6 | 50% |  |
| 0–1% | 3 | 59.7 | 209.1 | 209.1 | 100% | (thin) |
| 1–2% | 4 | 18.1 | 56.7 | 73.5 | 50% | (thin) |
| 2–4% | 3 | 48.6 | 168.0 | 168.0 | 100% | (thin) |
| >=4% | 8 | 19.9 | 71.8 | 206.8 | 50% |  |

### UTC Session Bins

| Session (UTC) | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| 00-04 | 2 | 54.1 | 108.3 | 100% | (thin) |
| 04-08 | 6 | 50.9 | 283.9 | 83% |  |
| 08-12 | 0 |  |  |  |  |
| 12-16 | 6 | -34.7 | -7.4 | 33% |  |
| 16-20 | 3 | 50.6 | 56.5 | 67% | (thin) |
| 20-24 | 7 | 48.3 | 199.2 | 57% |  |

### Cascade Notional Bins

| Cascade | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| <100K | 0 |  |  |  |  |
| 100K-200K | 0 |  |  |  |  |
| 200K-500K | 14 | 53.9 | 747.7 | 86% |  |
| 500K-1000K | 8 | -15.2 | -115.7 | 25% |  |
| >1000K | 2 | 4.2 | 8.4 | 50% | (thin) |

### Max Single Liq Share Bins

| Single Share | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| <50% | 6 | 67.4 | 347.5 | 83% |  |
| 50–80% | 8 | 47.8 | 239.4 | 62% |  |
| >=80% | 10 | 14.0 | 53.5 | 50% |  |

### Candidate Filter Tests

| Filter | N (kept) | N (removed) | Median | Cum | Top3R | WR | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| max_day_trend <= 4% | 16 | 8 | 48.5 | 568.6 | 698.4 | 69% | N<20 — too thin |
| max_day_trend <= 3% | 15 | 9 | 48.6 | 526.6 | 656.4 | 67% | N<20 — too thin |
| exclude UTC 20-24 | 17 | 7 | 48.6 | 441.2 | 622.6 | 65% | N<20 — too thin |
| min_liq_count >= 3 | 22 | 2 | 48.5 | 636.1 | 815.0 | 64% | ok |
| min_liq_count >= 5 | 18 | 6 | 48.5 | 646.8 | 740.1 | 67% | N<20 — too thin |
| max_single_share <= 80% | 14 | 10 | 52.8 | 586.9 | 671.4 | 71% | N<20 — too thin |
| cascade >= 200K | 24 | 0 | 48.5 | 640.4 | 821.8 | 62% | ok |
| max_trend<=4% + UTC<20 | 12 | 12 | 51.7 | 464.3 | 591.0 | 75% | N<20 — too thin |

## ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30

**N=24  Median=+41.1  WR=58%  Cum=+453.6  Top3R=+619.2**

### Day Trend Bins

| Trend | N | Median | Cum | Top3R | WR | Note |
| --- | --- | --- | --- | --- | --- | --- |
| <0% | 0 |  |  |  |  |  |
| 0–1% | 0 |  |  |  |  |  |
| 1–2% | 8 | 12.3 | 100.5 | 156.7 | 50% |  |
| 2–4% | 8 | 53.8 | 216.4 | 331.4 | 62% |  |
| >=4% | 8 | 52.2 | 136.7 | 292.1 | 62% |  |

### UTC Session Bins

| Session (UTC) | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| 00-04 | 0 |  |  |  |  |
| 04-08 | 1 | -17.3 | -17.3 | 0% | (thin) |
| 08-12 | 2 | 20.8 | 41.6 | 50% | (thin) |
| 12-16 | 12 | 54.1 | 504.6 | 83% |  |
| 16-20 | 4 | -8.2 | -3.8 | 50% | (thin) |
| 20-24 | 5 | -18.7 | -71.5 | 20% |  |

### Cascade Notional Bins

| Cascade | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| <100K | 0 |  |  |  |  |
| 100K-200K | 0 |  |  |  |  |
| 200K-500K | 12 | 12.5 | 196.0 | 50% |  |
| 500K-1000K | 6 | 41.1 | 136.9 | 67% |  |
| >1000K | 6 | 49.8 | 120.8 | 67% |  |

### Max Single Liq Share Bins

| Single Share | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| <50% | 8 | 44.4 | 203.6 | 62% |  |
| 50–80% | 11 | 47.7 | 183.1 | 64% |  |
| >=80% | 5 | -17.3 | 66.9 | 40% |  |

### Candidate Filter Tests

| Filter | N (kept) | N (removed) | Median | Cum | Top3R | WR | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| max_day_trend <= 4% | 16 | 8 | 32.8 | 317.0 | 446.0 | 56% | N<20 — too thin |
| max_day_trend <= 3% | 11 | 13 | -6.5 | 45.2 | 174.2 | 46% | N<20 — too thin |
| exclude UTC 20-24 | 19 | 5 | 49.9 | 525.1 | 684.4 | 68% | N<20 — too thin |
| min_liq_count >= 3 | 23 | 1 | 47.7 | 504.5 | 670.2 | 61% | improvement from ≤2 removes |
| min_liq_count >= 5 | 21 | 3 | 49.7 | 578.6 | 735.2 | 67% | ok |
| max_single_share <= 80% | 19 | 5 | 47.7 | 386.7 | 552.3 | 63% | N<20 — too thin |
| cascade >= 200K | 24 | 0 | 41.1 | 453.6 | 619.2 | 58% | ok |
| max_trend<=4% + UTC<20 | 11 | 13 | 49.9 | 388.5 | 463.6 | 73% | N<20 — too thin |

## BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30

**N=6  Median=+38.6  WR=67%  Cum=+214.0  Top3R=+212.7**

### Day Trend Bins

| Trend | N | Median | Cum | Top3R | WR | Note |
| --- | --- | --- | --- | --- | --- | --- |
| <0% | 2 | 78.9 | 157.7 | 157.7 | 100% | (thin) |
| 0–1% | 2 | 7.1 | 14.2 | 14.2 | 50% | (thin) |
| 1–2% | 1 | -12.9 | -12.9 | -12.9 | 0% | (thin) |
| 2–4% | 1 | 55.0 | 55.0 | 55.0 | 100% | (thin) |
| >=4% | 0 |  |  |  |  |  |

### UTC Session Bins

| Session (UTC) | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| 00-04 | 1 | -12.9 | -12.9 | 0% | (thin) |
| 04-08 | 1 | 22.1 | 22.1 | 100% | (thin) |
| 08-12 | 1 | -8.0 | -8.0 | 0% | (thin) |
| 12-16 | 2 | 78.2 | 156.4 | 100% | (thin) |
| 16-20 | 1 | 56.3 | 56.3 | 100% | (thin) |
| 20-24 | 0 |  |  |  |  |

### Cascade Notional Bins

| Cascade | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| <100K | 0 |  |  |  |  |
| 100K-200K | 0 |  |  |  |  |
| 200K-500K | 0 |  |  |  |  |
| 500K-1000K | 0 |  |  |  |  |
| >1000K | 6 | 38.6 | 214.0 | 67% |  |

### Max Single Liq Share Bins

| Single Share | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| <50% | 6 | 38.6 | 214.0 | 67% |  |
| 50–80% | 0 |  |  |  |  |
| >=80% | 0 |  |  |  |  |

### Candidate Filter Tests

| Filter | N (kept) | N (removed) | Median | Cum | Top3R | WR | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| max_day_trend <= 4% | 6 | 0 | 38.6 | 214.0 | 212.7 | 67% | N<20 — too thin |
| max_day_trend <= 3% | 6 | 0 | 38.6 | 214.0 | 212.7 | 67% | N<20 — too thin |
| exclude UTC 20-24 | 6 | 0 | 38.6 | 214.0 | 212.7 | 67% | N<20 — too thin |
| min_liq_count >= 3 | 6 | 0 | 38.6 | 214.0 | 212.7 | 67% | N<20 — too thin |
| min_liq_count >= 5 | 6 | 0 | 38.6 | 214.0 | 212.7 | 67% | N<20 — too thin |
| max_single_share <= 80% | 6 | 0 | 38.6 | 214.0 | 212.7 | 67% | N<20 — too thin |
| cascade >= 200K | 6 | 0 | 38.6 | 214.0 | 212.7 | 67% | N<20 — too thin |
| max_trend<=4% + UTC<20 | 6 | 0 | 38.6 | 214.0 | 212.7 | 67% | N<20 — too thin |

## ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30

**N=25  Median=-38.9  WR=28%  Cum=-80.9  Top3R=+84.9**

### Day Trend Bins

| Trend | N | Median | Cum | Top3R | WR | Note |
| --- | --- | --- | --- | --- | --- | --- |
| <0% | 0 |  |  |  |  |  |
| 0–1% | 0 |  |  |  |  |  |
| 1–2% | 10 | 0.8 | 232.4 | 364.2 | 50% |  |
| 2–4% | 9 | -45.8 | -168.1 | -9.7 | 11% |  |
| >=4% | 6 | -49.8 | -145.1 | 19.1 | 17% |  |

### UTC Session Bins

| Session (UTC) | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| 00-04 | 0 |  |  |  |  |
| 04-08 | 1 | -7.7 | -7.7 | 0% | (thin) |
| 08-12 | 1 | -11.8 | -11.8 | 0% | (thin) |
| 12-16 | 11 | -38.9 | 40.8 | 27% |  |
| 16-20 | 5 | -47.8 | -69.1 | 20% |  |
| 20-24 | 7 | -43.5 | -33.1 | 43% |  |

### Cascade Notional Bins

| Cascade | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| <100K | 8 | -26.2 | -88.4 | 25% |  |
| 100K-200K | 9 | -47.8 | -357.3 | 0% |  |
| 200K-500K | 3 | 31.1 | 88.8 | 67% | (thin) |
| 500K-1000K | 1 | 99.7 | 99.7 | 100% | (thin) |
| >1000K | 4 | 51.1 | 176.4 | 50% | (thin) |

### Max Single Liq Share Bins

| Single Share | N | Median | Cum | WR | Note |
| --- | --- | --- | --- | --- | --- |
| <50% | 5 | -11.8 | -17.0 | 20% |  |
| 50–80% | 11 | -38.9 | -48.3 | 36% |  |
| >=80% | 9 | -43.5 | -15.5 | 22% |  |

### Candidate Filter Tests

| Filter | N (kept) | N (removed) | Median | Cum | Top3R | WR | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| max_day_trend <= 4% | 19 | 6 | -10.9 | 64.3 | 222.7 | 32% | N<20 — too thin |
| max_day_trend <= 3% | 17 | 8 | -9.8 | 123.2 | 281.6 | 35% | N<20 — too thin |
| exclude UTC 20-24 | 18 | 7 | -25.4 | -47.8 | 118.0 | 22% | N<20 — too thin |
| min_liq_count >= 3 | 20 | 5 | -11.3 | 117.2 | 283.0 | 35% | ok |
| min_liq_count >= 5 | 16 | 9 | -11.3 | -1.0 | 157.5 | 31% | N<20 — too thin |
| max_single_share <= 80% | 16 | 9 | -25.4 | -65.3 | 100.5 | 31% | N<20 — too thin |
| cascade >= 200K | 8 | 17 | 65.4 | 364.9 | 484.2 | 62% | N<20 — too thin |
| max_trend<=4% + UTC<20 | 13 | 12 | -10.9 | 45.5 | 203.9 | 23% | N<20 — too thin |

---
## Summary — Which Pooled Findings Survive Per-Rule?

| Finding | ETH 500K | SOL 200K | ETH 200K | BTC 1M | ETH 50K | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| day_trend >4% bad | N/A (0 trades) | Minor drag (WR 50% vs 62%) | No — WR=62% same as avg | N/A (thin) | Yes — WR=17% | **Pooled effect = ETH 50K artifact. Does NOT hold for active rules.** |
| UTC 20-24 weak | N/A (0 trades) | Opposite! WR=57% cum=+199 | Confirmed WR=20% cum=-72 | N/A (0 trades) | All sessions bad | **Mixed. ETH 200K weak. SOL 200K opposite. Not universal.** |
| cascade <200K bad | N/A (threshold ≥500K) | N/A (threshold ≥200K) | N/A (threshold ≥200K) | N/A (threshold ≥1M) | Confirmed WR=0-25% | **Finding = ETH 50K only. Active rules already above threshold.** |
| SOL cascade 500K-1M bad | — | Yes: WR=25% cum=-116 vs 200K-500K WR=86% | — | — | — | **SOL-specific, worth monitoring. N=8 borderline.** |
| Single-liq dominance | Flat (~79% all bins) | >=80% share → WR=50% vs <50% WR=83% | >=80% slightly worse | All <50% | All high-share bad | **SOL 200K signal: spread cascade >> spike.** |

## Key Per-Rule Findings

### ETH_BUY_LIQ_LONG_500K_DAYTREND0 — N=19, median=+52.3, WR=79%
- **No >4% trend trades at all** — already naturally filtered (min_day_trend_bps=0 + regime gate)
- **No 20-24 UTC trades** — natural absence in the data
- **All cascade ≥500K** by design; WR consistent across 500K-1M and >1M (~79-80%)
- Single share distribution flat — no spike concern
- **No filter is applicable: all candidates hit N<20 threshold**
- Conclusion: rule is working well as-is. No change warranted.

### SOL_BUY_LIQ_LONG_200K — N=24, median=+48.5, WR=62%
- **UTC 20-24 is NOT weak here** (N=7, WR=57%, cum=+199). Pooled finding was from ETH 50K.
- **12-16 UTC is actually the worst session** (N=6, WR=33%, cum=-7). Surprising — needs more data.
- **Cascade 200K-500K = best** (N=14, WR=86%, cum=+748). Cascade 500K-1M = bad (N=8, WR=25%, cum=-116).
- Single-liq dominance (≥80% share): WR=50% vs multi-liq WR=83%. Spread cascade >> spike.
- `min_liq_count >= 3` is the only filter staying above N=20 (N=22, WR=64%). Marginal improvement.
- Conclusion: SOL 200K-500K cascade with multi-liq structure is the sweet spot. Worthy of shadow exploratory testing if N grows.

### ETH_BUY_LIQ_LONG_200K — N=24, median=+41.1, WR=58%
- **day_trend >4% is fine** here (WR=62%, same as avg). Pooled harm = ETH 50K.
- **UTC 20-24 is genuinely weak** (N=5, WR=20%, cum=-72). Only confirmed cross-rule finding.
- **12-16 UTC is best** (N=12, WR=83%, cum=+505) — strongest single session across all rules.
- `min_liq_count >= 5` is the only filter above N=20 threshold (N=21, WR 58→67%, cum +454→+579).
- Conclusion: UTC 20-24 weakness is real for ETH 200K. min_liq_count>=5 shows improvement but N=21 is barely above threshold — needs more data before acting.

### ETH_BUY_LIQ_LONG_50K — N=25, median=-38.9, WR=28% [DIAGNOSIS ONLY]
- Not in active allow list — confirmed no live impact.
- Root cause: threshold too low (fires on tiny cascades). 83% of trades have cascade <200K.
- cascade <200K: WR=0-25%, cascade ≥200K: WR=62-100% (but N=8, thin).
- Pooled findings (day_trend >4% bad, cascade <200K bad, UTC 20-24 bad) are **all artifacts of this rule**.
- Recommendation: exclude from any live-forward dashboards / monitoring panels to avoid confusion.

## No Live Rule Change Recommended

Evidence assessment:
- ETH_500K: too few trades (N=19) for any filter. Currently performing well (WR=79%).
- SOL_200K: cascade 500K-1M weakness (N=8) borderline — monitor as data grows.
- ETH_200K: UTC 20-24 (N=5) and min_liq_count>=5 (N=21) suggestive but not conclusive.
- BTC_1M: N=6, all evidence too thin.

**No live rule change recommended.** Revisit when each active rule reaches N≥50.
