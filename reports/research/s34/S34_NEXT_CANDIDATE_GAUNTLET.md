# S34 Next Candidate Gauntlet

Generated: `2026-07-01T03:36:43.028541+00:00`

Research-only. No live executor, env, order logic, leverage, or sizing was changed.

## Baselines
- sync<200K baseline: N=100 | WR=58.0% | avg=+26.5 bps | /mo=23.0
- current LONG gate: N=9 | WR=100.0% | avg=+170.6 bps | /mo=2.3

## Candidate Results
| Candidate | Family | N | WR | Avg | Sum | T3R | /mo | WF +sum | WF +T3R | Worst fold | MC p | Basic | Note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| C_score_relax_short1m10 | COMBINED | 25 | 92.0% | +138.3 | +3457.8 | +2384.8 | 6.2 | 5/5 | 4/5 | +329.4 | 0.018 | True | L_base_score1_added + S_btc1m_delay10 |
| C_btc7d500_short1m10 | COMBINED | 24 | 87.5% | +127.5 | +3059.4 | +2015.2 | 6.1 | 5/5 | 3/5 | +236.5 | 0.240 | True | L_btc7d_lt_500 + S_btc1m_delay10 |
| C_no_btc7d_short1m10 | COMBINED | 31 | 80.6% | +98.5 | +3053.0 | +2008.8 | 7.9 | 5/5 | 3/5 | +53.2 | 0.248 | True | L_no_btc7d + S_btc1m_delay10 |
| C_freq_balanced_btc4h_short1m10 | COMBINED | 27 | 85.2% | +108.9 | +2941.3 | +1897.1 | 6.9 | 5/5 | 3/5 | +214.5 | 0.375 | True | L_btc4h_lt0_no_btc7d + S_btc1m_delay10 |
| C_current_live_long_short | COMBINED | 16 | 93.8% | +139.3 | +2229.0 | +1184.8 | 4.1 | 4/5 | 3/5 | -58.4 | 0.977 | True | L_current_live_gate + S_current_btc2m_delay5 |
| L_base_score1_added | LONG | 15 | 93.3% | +141.7 | +2125.7 | +1082.6 | 3.7 | 5/5 | 5/5 | +140.8 | 0.109 | True | Relax long_score to include base_score1. |
| L_btc7d_lt_500 | LONG | 14 | 85.7% | +123.4 | +1727.3 | +782.1 | 3.6 | 5/5 | 5/5 | +79.5 | 0.636 | True | Relax btc7d to +500 bps. |
| L_btc4h_lt0_no_btc7d | LONG | 17 | 82.4% | +94.7 | +1609.2 | +751.0 | 4.3 | 4/5 | 3/5 | -0.8 | 0.798 | True | Use btc4h<0 only. |
| L_wed_only_block_btc7d0 | LONG | 11 | 100.0% | +152.8 | +1680.4 | +735.2 | 2.5 | 5/5 | 5/5 | +233.6 | 0.709 | True | Remove Monday block; keep Wednesday block. |
| S_btc1m_delay5 | SHORT | 12 | 83.3% | +123.3 | +1479.9 | +558.8 | 3.3 | 5/5 | 5/5 | +126.7 | 0.236 | True | SHORT BTC>=1000000, delay>=5m, hold=2h. |
| S_btc1m_delay10 | SHORT | 10 | 90.0% | +133.2 | +1332.1 | +523.6 | 2.7 | 4/5 | 4/5 | -11.2 | 0.440 | True | SHORT BTC>=1000000, delay>=10m, hold=2h. |
| SEQ_silence_no_btc1m | STATE_SEQUENCE_LONG4H | 166 | 60.8% | +37.9 | +6290.4 | +4947.3 | 38.1 | 5/5 | 4/5 | +108.3 | 0.001 | False | Silence path with no BTC confirm. |
| L_no_btc7d | LONG | 21 | 76.2% | +82.0 | +1721.0 | +775.8 | 5.3 | 4/5 | 2/5 | -51.4 | 0.649 | False | Remove btc7d regime gate. |
| L_notional_300_500_btc7d0 | LONG | 3 | 100.0% | +211.1 | +633.4 | +633.4 | 0.8 | 3/5 | 3/5 | +0.0 | 1.000 | False | Notional sweet spot with current btc7d gate. |
| L_current_live_gate | LONG | 9 | 100.0% | +170.6 | +1535.8 | +590.6 | 2.3 | 5/5 | 5/5 | +105.1 | 0.870 | False | Current conservative LONG gate. |
| L_btc3d_lt0 | LONG | 9 | 100.0% | +163.7 | +1472.9 | +527.7 | 2.3 | 5/5 | 5/5 | +105.1 | 0.916 | False | Use btc3d<0 instead of btc7d. |
| S_btc1m_delay15 | SHORT | 7 | 85.7% | +89.3 | +625.0 | +101.5 | 7.0 | 4/5 | 4/5 | -11.2 | 0.999 | False | SHORT BTC>=1000000, delay>=15m, hold=2h. |
| SEQ_silence_with_btc1m | STATE_SEQUENCE_LONG4H | 1 | 100.0% | +51.7 | +51.7 | +51.7 | 1.0 | 1/5 | 1/5 | +0.0 | 0.997 | False | Silence path but BTC confirm appears. |
| S_btc2m_delay10 | SHORT | 5 | 100.0% | +153.4 | +767.0 | +15.4 | 5.0 | 4/5 | 4/5 | +0.0 | 0.988 | False | SHORT BTC>=2000000, delay>=10m, hold=2h. |
| S_current_hold_90m | SHORT_HOLD | 7 | 85.7% | +71.5 | +500.8 | -37.0 | 7.0 | 4/5 | 4/5 | -131.1 | 1.000 | False | Current SHORT confirm with hold=90m. |
| S_current_btc2m_delay5 | SHORT | 7 | 85.7% | +99.0 | +693.2 | -58.4 | 7.0 | 4/5 | 4/5 | -169.6 | 0.994 | False | SHORT BTC>=2000000, delay>=5m, hold=2h. |
| S_current_hold_120m | SHORT_HOLD | 7 | 85.7% | +99.0 | +693.2 | -58.4 | 7.0 | 4/5 | 4/5 | -169.6 | 0.965 | False | Current SHORT confirm with hold=120m. |
| L_notional_300_500 | LONG | 8 | 62.5% | +69.8 | +558.2 | -79.0 | 2.2 | 3/5 | 3/5 | -197.9 | 1.000 | False | Anchor running_notional sweet spot; no btc7d gate. |
| S_current_hold_180m | SHORT_HOLD | 7 | 71.4% | +75.6 | +529.3 | -102.0 | 7.0 | 4/5 | 4/5 | -118.8 | 1.000 | False | Current SHORT confirm with hold=180m. |
| S_current_hold_150m | SHORT_HOLD | 7 | 57.1% | +74.4 | +520.8 | -197.0 | 7.0 | 4/5 | 4/5 | -206.3 | 1.000 | False | Current SHORT confirm with hold=150m. |
| SEQ_noisy_no_btc1m | STATE_SEQUENCE_LONG4H | 205 | 57.1% | -1.3 | -270.7 | -1341.7 | 46.9 | 2/5 | 2/5 | -1905.5 | 1.000 | False | Noisy/follow-on without BTC confirm. |
| SEQ_noisy_with_btc1m | STATE_SEQUENCE_LONG4H | 48 | 22.9% | -129.4 | -6212.7 | -6615.0 | 11.6 | 0/5 | 0/5 | -1900.3 | 1.000 | False | Noisy/follow-on plus BTC confirm. |

## Interpretation
- `Basic=True` only means the candidate cleared simple in-sample robustness thresholds; MC p and fold stability still matter.
- `MC p` is max-stat multiple-comparison permutation inside each family, so it is stricter than a single-cell shuffle.
- Low-N candidates are hypotheses, not live promotions.