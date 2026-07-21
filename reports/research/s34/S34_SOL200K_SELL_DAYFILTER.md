# SOL 200K SELL Bear-Day Filter Sweep

Generated: 2026-06-26T10:24:27.359695+00:00

**Motivation:** Regime analysis found SOL 200K SELL is strongly bear-day dependent.
Bull days: N=6, median=-10.1 bps, WR=33%. Bear days: N=40, median=+50.7, WR=70%.
This sweep finds the optimal day_trend gate and checks if it's stable.

Lookback: 120 days. Total closed trades (no filter): 46. No-fill: 11 (19%).

---

## 1. Day Trend Gate Sweep

| Gate | N | % kept | Median | Mean | WR | Top3-Rmv | Pos Days | 1H Median | 2H Median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no filter | 46 | 100% | +50.3 | +29.8 | 65% | +1070 | 14/18 | +48.0 | +51.0 |
| trend < 0 | 40 | 87% | +50.7 | +32.3 | 70% | +1016 | 12/14 | +48.2 | +51.2 |
| trend < -25 | 38 | 83% | +50.7 | +29.7 | 68% | +898 | 10/12 | +48.0 | +51.0 |
| trend < -50 | 35 | 76% | +51.0 | +32.5 | 71% | +908 | 10/11 | +51.0 | +50.9 |
| trend < -100 | 31 | 67% | +51.0 | +33.7 | 71% | +830 | 10/11 | +51.5 | +50.9 |
| trend < -150 | 22 | 48% | +51.1 | +30.6 | 68% | +478 | 7/9 | +51.5 | +50.8 |

---

## 2. Combined Gate: Bear Day + Liq Imbalance

Base filter: `day_trend < 0`. Adding liq imbalance gate on top.

| Gate | N | % of all | Median | Mean | WR | Top3-Rmv | 1H Median | 2H Median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no liq filter | 40 | 87% | +50.7 | +32.3 | 70% | +1016 | +48.2 | +51.2 |
| sell-heavy < -0.05 | 36 | 78% | +50.4 | +30.9 | 69% | +881 | +48.2 | +51.0 |
| sell-heavy < -0.10 | 36 | 78% | +50.4 | +30.9 | 69% | +881 | +48.2 | +51.0 |
| sell-heavy < -0.25 | 32 | 70% | +50.4 | +32.0 | 72% | +795 | +44.1 | +51.2 |

---

## 3. Bear Day Range Split

Within bear-day trades only — does range matter?

| Slice | N | Median | Mean | WR | Top3-Rmv |
|---|---:|---:|---:|---:|---:|
| Bear, range < 250 | 13 | +49.9 | +35.2 | 77% | +271 |
| Bear, range 250-500 | 21 | +50.8 | +29.3 | 62% | +338 |
| Bear, range >= 500 | 6 | +50.2 | +36.8 | 83% | +59 |

---

## 4. Conclusion

**Best stable gate: `trend < -150`**

- N=22 (48% of all signals kept)
- Median=+51.1  Mean=+30.6  WR=68%
- Half-split: 1H=+51.5  2H=+50.8  (both positive)
- Top3-removed cum=+478
- Positive days: 7/9

**Gate assessment:**

- Enough N for exploratory (N>=30): NO — N too small for pre-reg criteria
- Median improvement vs no filter: +0.8 bps

**Gate improves quality but N too small for independent pre-registration. Watchlist. More lookback data needed before committing.**

_Read-only. No runner, config, or pre-reg changes made._
