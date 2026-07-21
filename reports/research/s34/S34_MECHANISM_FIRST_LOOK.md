# S34 Mechanism First Look

> event=418 kontrol=418. 2026-07-02

## A — continuation vs reversal ayiricilar (TEST)

- **fund_rate** fav=lo: TEST N=55 WR=69.1% avg=+64.9 (anti -49.9, D=+114.8) cont% 56.3→43.6
- **fl_post5_impact** fav=lo: TEST N=43 WR=44.2% avg=-46.5 (anti +24.4, D=-70.9) cont% 51.8→48.8
- **px_rv** fav=hi: TEST N=77 WR=59.7% avg=+26.6 (anti -41.2, D=+67.8) cont% 57.1→46.8
- **fl_pre10_avg_sz** fav=lo: TEST N=96 WR=63.5% avg=+16.1 (anti -50.6, D=+66.7) cont% 53.3→50.0
- **fund_vel_1h** fav=lo: TEST N=80 WR=61.2% avg=+23.9 (anti -40.9, D=+64.8) cont% 52.2→50.0
- **fl_pre10_ofi** fav=hi: TEST N=67 WR=68.7% avg=+28.8 (anti -32.2, D=+61.0) cont% 54.2→47.8
- **liq_two_sided_1h** fav=hi: TEST N=72 WR=58.3% avg=+24.5 (anti -32.1, D=+56.6) cont% 48.1→52.8
- **fund_vel_8h** fav=lo: TEST N=56 WR=60.7% avg=+28.0 (anti -22.0, D=+50.0) cont% 52.9→48.2
- **fl_post1_avg_sz** fav=lo: TEST N=88 WR=61.4% avg=+14.9 (anti -33.7, D=+48.6) cont% 57.9→47.7
- **bk_refill** fav=hi: TEST N=73 WR=60.3% avg=+13.2 (anti -17.6, D=+30.8) cont% 58.5→45.2
- **liq_btc_sync** fav=hi: TEST N=82 WR=56.1% avg=-8.8 (anti +17.0, D=-25.8) cont% 52.3→50.0
- **bk_pull** fav=hi: TEST N=78 WR=60.3% avg=+9.9 (anti -15.5, D=+25.4) cont% 60.4→44.9
- **bk_pre10_imb** fav=lo: TEST N=66 WR=56.1% avg=-10.5 (anti +12.0, D=-22.5) cont% 50.0→51.5
- **bk_pre1_imb** fav=lo: TEST N=58 WR=60.3% avg=+12.3 (anti -10.1, D=+22.4) cont% 54.4→46.6
- **fl_post1_impact** fav=hi: TEST N=57 WR=64.9% avg=+11.1 (anti -8.8, D=+19.9) cont% 59.4→40.4
- **px_ret_1h** fav=lo: TEST N=65 WR=64.6% avg=+9.5 (anti -9.7, D=+19.2) cont% 52.5→49.2
- **fl_pre10_impact** fav=hi: TEST N=77 WR=57.1% avg=+7.4 (anti -11.1, D=+18.5) cont% 55.1→48.1
- **bk_post1_spread_max** fav=lo: TEST N=66 WR=56.1% avg=-7.2 (anti +8.3, D=-15.5) cont% 51.7→50.0
- **fl_post1_ofi** fav=hi: TEST N=61 WR=52.5% avg=+7.7 (anti -6.8, D=+14.5) cont% 60.0→41.0
- **bk_pre10_imb_slope** fav=lo: TEST N=59 WR=57.6% avg=-4.9 (anti +4.8, D=-9.7) cont% 49.3→52.5
- **bk_pre10_spread** fav=hi: TEST N=126 WR=57.1% avg=+0.2 (anti +0.0, D=+0.2) cont% None→50.8

## B — event-vs-kontrol ayrimi (pre-cascade)

- **liq_btc_sync**: sep=1.0 (med_ev=54228.22980000001, med_ctl=0.0)
- **liq_two_sided_1h**: sep=1.0 (med_ev=80907.14211, med_ctl=0.0)
- **px_rv**: sep=0.909 (med_ev=0.0021854183535057626, med_ctl=0.0009705440510023222)
- **basis_spot_bps**: sep=0.789 (med_ev=-15.239559841571419, med_ctl=-4.412188168395209)
- **fl_pre10_ofi**: sep=0.675 (med_ev=-0.14284823011390294, med_ctl=-0.0009664563487935644)
- **px_ret_1h**: sep=0.67 (med_ev=-42.14313664668544, med_ctl=2.2675112336072787)
- **bk_pre10_spread**: sep=0.656 (med_ev=6.333966764844879e-06, med_ctl=4.999853997426931e-06)
- **basis_spot_slope**: sep=0.604 (med_ev=-9.293604274525244, med_ctl=-0.39357209437470164)
- **fund_rate**: sep=0.545 (med_ev=2.96e-06, med_ctl=3.062e-05)
- **bk_pre1_imb**: sep=0.469 (med_ev=-0.10896560246579556, med_ctl=-0.0027622138544109317)
- **bk_pull**: sep=0.445 (med_ev=8.113667020483247e-06, med_ctl=7.170485529422127e-06)
- **fl_pre10_impact**: sep=0.411 (med_ev=0.46907607929176376, med_ctl=0.3014162396980847)

## C — giris gecikme egrisi

- **grid_2s**: N=418 WR=54.3% avg=-2.2 mc=0.6
- **grid_5s**: N=418 WR=54.3% avg=-1.9 mc=0.584
- **grid_10s**: N=418 WR=53.8% avg=-1.4 mc=0.57
- **grid_30s**: N=418 WR=51.9% avg=-1.1 mc=0.562
- **grid_1m**: N=418 WR=52.6% avg=-0.3 mc=0.524
- **grid_5m**: N=417 WR=53.7% avg=-1.0 mc=0.552
- **grid_15m**: N=417 WR=50.1% avg=-3.5 mc=0.67
- **grid_refillhi_10s**: N=219 WR=58.4% avg=+11.2 mc=0.134
- **grid_refillhi_1m**: N=219 WR=57.5% avg=+10.4 mc=0.144
- **grid_refillhi_5m**: N=218 WR=56.0% avg=+6.2 mc=0.28

---
*Script: tools/s34_mechanism_first_look.py*