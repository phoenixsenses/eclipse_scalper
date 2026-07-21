# S34 Frequency Expansion Tests

Generated: `2026-07-01T03:22:03.827417+00:00`

Baseline sync<200K: N=100 | WR=58.0% | avg=+26.5 bps | /mo=23.0

## [B1_btc3d_lt0]
N=9 | WR=100.0% | avg=+163.7 bps | /mo=2.3
vs baseline (N=34, WR=70.6%, +76.2 bps)
Karar: PROMISING

## [B2_btc4h_only_no_btc7d]
N=17 | WR=82.4% | avg=+94.7 bps | /mo=4.3
vs baseline (N=34, WR=70.6%, +76.2 bps)
Karar: PROMISING

## [B3_asia_sync_btc7d]
N=12 | WR=75.0% | avg=+93.8 bps | /mo=3.0
vs baseline (N=34, WR=70.6%, +76.2 bps)
Karar: PROMISING

## [B4_score5_no_btc7d]
N=0 | WR=NA | avg=NA bps | /mo=0.0
vs baseline (N=34, WR=70.6%, +76.2 bps)
Karar: WORSE

## [B5_eth_sell_150k_added_no_200k_bucket]
N=143 | WR=55.2% | avg=+7.0 bps | /mo=32.8
vs baseline (N=34, WR=70.6%, +76.2 bps)
Karar: WORSE

## [B6_multiple_anchor_n2h]
- n2h_0_1: N=34 | WR=67.6% | avg=+41.2 bps | /mo=7.8
- n2h_2_4: N=49 | WR=49.0% | avg=+1.4 bps | /mo=11.5
- n2h_5_plus: N=17 | WR=64.7% | avg=+69.7 bps | /mo=4.0

## [B7_btc1h_split]
- btc1h_gt0: N=16 | WR=43.8% | avg=-19.3 bps | /mo=3.9
- btc1h_lt0: N=84 | WR=60.7% | avg=+35.3 bps | /mo=19.4

## [B8_short_btc1m_longer_delay]
- btc1m_delay10: N=10 | WR=90.0% | avg=+133.2 bps | /mo=2.7
- btc1m_delay15: N=7 | WR=85.7% | avg=+89.3 bps | /mo=7.0
- current_btc2m_delay5: N=7 | WR=85.7% | avg=+99.0 bps | /mo=7.0

## [B9_funding_rate]
- source: `mark_prices`
- negative: N=59 | WR=55.9% | avg=+38.4 bps | /mo=13.5
- non_negative: N=41 | WR=61.0% | avg=+9.4 bps | /mo=9.8

## [B10_btc7d_or_score5]
N=9 | WR=100.0% | avg=+170.6 bps | /mo=2.3
vs baseline (N=34, WR=70.6%, +76.2 bps)
Karar: PROMISING
