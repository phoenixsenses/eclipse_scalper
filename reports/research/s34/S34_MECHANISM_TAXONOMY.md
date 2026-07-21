# S34 Mechanism Taxonomy (Faz 2)

> event=418 gated=97. Esikler ungated-TRAIN. 2026-07-02

## Gated altkume tek-ayiricilar

- **bk_pull** (hi): favN=58 WR=70.7% avg=+57.4 (anti -12.8) D=+70.2 mc=0.0
- **fl_pre10_avg_sz** (lo): favN=70 WR=67.1% avg=+48.5 (anti -21.0) D=+69.5 mc=0.0
- **fund_vel_1h** (lo): favN=37 WR=64.9% avg=+64.4 (anti +7.4) D=+57.0 mc=0.002
- **bk_refill** (hi): favN=53 WR=56.6% avg=+3.9 (anti +59.6) D=-55.7 mc=0.43
- **fund_rate** (lo): favN=44 WR=63.6% avg=+58.2 (anti +5.0) D=+53.2 mc=0.002
- **fl_pre10_ofi** (hi): favN=55 WR=69.1% avg=+42.1 (anti +12.3) D=+29.8 mc=0.008
- **px_rv** (hi): favN=54 WR=63.0% avg=+36.1 (anti +20.5) D=+15.6 mc=0.036
- **liq_two_sided_1h** (hi): favN=48 WR=58.3% avg=+33.1 (anti +25.3) D=+7.8 mc=0.042

## Mekanizma kompoziti

- **ungated_m3**: N=329 WR=56.5% avg=+7.9 TOT=2586.0 worst=-621.4 mc=0.19
- **ungated_m3_TEST**: N=107 WR=62.6% avg=+16.5 TOT=1769.0 worst=-621.4 mc=0.21
- **ungated_m4**: N=220 WR=61.4% avg=+29.1 TOT=6396.0 worst=-443.2 mc=0.0
- **ungated_m4_TEST**: N=76 WR=67.1% avg=+49.9 TOT=3792.0 worst=-443.2 mc=0.002
- **ungated_m5**: N=125 WR=63.2% avg=+40.6 TOT=5075.0 worst=-443.2 mc=0.004
- **ungated_m5_TEST**: N=53 WR=67.9% avg=+55.5 TOT=2943.0 worst=-443.2 mc=0.012
- **gated_m3**: N=79 WR=63.3% avg=+31.4 TOT=2480.0 worst=-347.5 mc=0.014
- **gated_m3_TEST**: N=28 WR=75.0% avg=+64.2 TOT=1798.0 worst=-150.8 mc=0.01
- **gated_m4**: N=60 WR=70.0% avg=+49.9 TOT=2992.0 worst=-206.3 mc=0.0
- **gated_m4_TEST**: N=23 WR=82.6% avg=+88.0 TOT=2024.0 worst=-150.8 mc=0.002
- **gated_m5**: N=36 WR=75.0% avg=+75.8 TOT=2728.0 worst=-206.3 mc=0.0
- **gated_m5_TEST**: N=16 WR=81.2% avg=+98.6 TOT=1577.0 worst=-150.8 mc=0.014

## Funding 2x2 (seviye x velocity)

- **rate_lo_vel_lo**: N=117 WR=55.6% avg=+31.9 mc=0.004
- **rate_lo_vel_hi**: N=84 WR=61.9% avg=+25.1 mc=0.038
- **rate_hi_vel_lo**: N=109 WR=48.6% avg=-22.4 mc=0.858
- **rate_hi_vel_hi**: N=108 WR=49.1% avg=-43.4 mc=0.998

---
*Script: tools/s34_mechanism_taxonomy.py*