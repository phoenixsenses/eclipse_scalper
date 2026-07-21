# S34 Pre-Cascade Dip-Recovery Pattern (2-3 dusus+cikis hipotezi)

- Tarih: 2026-07-06T15:51:37.765361+00:00
- Universe: ETHUSDT SELL 200K anchors, bucket=300s, min_gap=900s
- Outcome: LONG 4h hold net (FEE=5bps)
- Events: 625 (4.66 ay), TRAIN 60%
- Feature: T0 oncesi tamamlanmis dip->recovery dongusu (zigzag, retrace>=60%); T0'a giren tamamlanmamis dusus sayilmaz.

## Secilen config (TRAIN'de secildi): `tf15_n32_a60`

TRAIN: n23=69 avg23=39.1 rest=-9.3 diff=48.4

## TEST sonucu

| bucket | N | WR% | avg net | total | worst | mc_p |
|---|---|---|---|---|---|---|
| 0 | 132 | 59.1 | 21.6 | 2852.0 | -464.5 | 0.035 |
| 1 | 84 | 52.4 | -20.6 | -1734.0 | -512.1 | 0.844 |
| 2 | 23 | 65.2 | 48.8 | 1122.0 | -313.6 | 0.085 |
| 3 | 9 | 44.4 | -20.0 | -180.0 | -208.8 | 0.695 |
| 4+ | 2 | 50.0 | -48.7 | -97.0 | -145.9 | None |

**2-3 vs rest (TEST):** diff=24.7 bps, perm_p=0.203 (n23=32, rest=218)

### Hold duyarliligi (TEST, secilen config)

| hold | 2-3: N/WR/avg/total | rest: N/WR/avg/total |
|---|---|---|
| l1h | 32/53.1/-4.0/-128.0 | 218/54.1/-1.6/-343.0 |
| l4h | 32/59.4/29.4/941.0 | 218/56.4/4.7/1021.0 |
| l6h | 32/62.5/62.8/2009.0 | 218/57.3/5.2/1142.0 |

### Core gate altinda (not bull, not EU, silence) — TEST

n=66, 2-3 vs rest diff=11.1 perm_p=0.411 (n23=11 rest=55)


### TRAIN grid ozeti (min bucket N saglayanlar)

| config | N | n23 | avg23 | rest | diff |
|---|---|---|---|---|---|
| tf1_n60_a10 | 355 | 134 | -5.2 | 3.1 | -8.3 |
| tf1_n60_a20 | 355 | 100 | -1.1 | 0.3 | -1.4 |
| tf1_n60_a35 | 355 | 49 | 8.6 | -1.4 | 10.0 |
| tf1_n120_a10 | 355 | 108 | -6.9 | 2.9 | -9.8 |
| tf1_n120_a20 | 355 | 123 | 0.0 | -0.1 | 0.1 |
| tf1_n120_a35 | 355 | 97 | 31.2 | -11.8 | 43.0 |
| tf1_n120_a60 | 355 | 33 | 12.1 | -1.3 | 13.4 |
| tf3_n40_a10 | 356 | 145 | -7.8 | 6.0 | -13.8 |
| tf3_n40_a20 | 356 | 122 | 7.5 | -3.3 | 10.8 |
| tf3_n40_a35 | 356 | 63 | 11.6 | -2.0 | 13.6 |
| tf3_n40_a60 | 356 | 18 | -3.1 | 0.6 | -3.7 |
| tf3_n80_a10 | 356 | 85 | -15.5 | 5.4 | -20.9 |
| tf3_n80_a20 | 356 | 132 | -5.6 | 3.9 | -9.5 |
| tf3_n80_a35 | 356 | 126 | 8.0 | -3.8 | 11.8 |
| tf3_n80_a60 | 356 | 54 | -5.8 | 1.5 | -7.3 |
| tf5_n48_a10 | 353 | 114 | -22.8 | 11.3 | -34.1 |
| tf5_n48_a20 | 353 | 148 | 3.6 | -2.1 | 5.7 |
| tf5_n48_a35 | 353 | 105 | 21.1 | -8.6 | 29.7 |
| tf5_n48_a60 | 353 | 40 | 26.0 | -3.0 | 29.0 |
| tf5_n96_a10 | 353 | 61 | -14.5 | 3.4 | -17.9 |
| tf5_n96_a20 | 353 | 96 | -20.3 | 8.0 | -28.3 |
| tf5_n96_a35 | 353 | 135 | -8.9 | 6.0 | -14.9 |
| tf5_n96_a60 | 353 | 118 | 14.7 | -7.0 | 21.7 |
| tf15_n32_a10 | 358 | 140 | -18.1 | 11.6 | -29.7 |
| tf15_n32_a20 | 358 | 173 | -17.8 | 16.7 | -34.5 |
| tf15_n32_a35 | 358 | 141 | -11.1 | 7.3 | -18.4 |
| tf15_n32_a60 | 358 | 69 | 39.1 | -9.3 | 48.4 |
| tf15_n64_a10 | 358 | 59 | 5.2 | -1.0 | 6.2 |
| tf15_n64_a20 | 358 | 96 | 9.5 | -3.4 | 12.9 |
| tf15_n64_a35 | 358 | 135 | -9.2 | 5.6 | -14.8 |
| tf15_n64_a60 | 358 | 145 | 0.7 | -0.4 | 1.1 |

> Knowledge Object notu: kanit seviyesi = tek-universe TRAIN/TEST + perm testi; kapsam = ETH SELL 200K cascade LONG reversion; curutme kosulu = TEST perm_p > 0.05 veya diff isareti TRAIN/TEST arasi tutarsiz.
