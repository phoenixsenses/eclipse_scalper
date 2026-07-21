# S34 State Machine Deep-Dive Tests

Generated: `2026-07-01T04:12:40.463384+00:00`

Research-only. No live executor, env, order logic, leverage, sizing, or runtime state was changed.

## Ideas Tested
1. Added-only value: each relaxed candidate's incremental trades vs current live combo.
2. 70/30 chronological holdout: calibration versus latest holdout behavior.
3. Month stability: whether a candidate is one-regime/month dependent.
4. Fee sensitivity: robustness if total costs rise from 5 bps to 8/10/15 bps.
5. Tail and drawdown: worst loss, -100 bps tail count, max drawdown, losing streaks.
6. No-overlap execution: one-position-at-a-time simulation, because live cannot hold infinite overlapping trades.
7. Candidate overlap map: whether new candidates add independent events or mostly relabel current trades.
8. State navigation: silence/noisy + BTC confirm as OK/DANGER context, not an entry alpha.
9. Current-vs-relaxed delta: whether frequency comes from genuinely positive added trades.
10. Live-readiness score: reject/paper/research-only classification from the above checks.

## Candidate Summary
| Candidate | N | WR | Avg | T3R | Added N | Added Avg | Holdout Sum | Holdout T3R | Worst | TailN | NoOverlap N | Readiness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C_current_live_long_short | 16 | 93.8% | +139.3 | +1184.8 | 0 |  | +358.9 | -154.3 | -169.6 | 1 | 13 | RESEARCH_ONLY_LOW_N |
| C_score_relax_short1m10 | 25 | 92.0% | +138.3 | +2384.8 | 12 | +97.5 | +1223.3 | +331.8 | -49.2 | 0 | 20 | PAPER_CANDIDATE |
| C_no_btc7d_short1m10 | 31 | 80.6% | +98.5 | +2008.8 | 18 | +42.5 | +1472.4 | +580.9 | -197.9 | 1 | 25 | PAPER_SHADOW_ONLY_TAIL |
| C_freq_balanced_btc4h_short1m10 | 27 | 85.2% | +108.9 | +1897.1 | 16 | +62.0 | +1395.7 | +504.2 | -33.8 | 0 | 22 | PAPER_CANDIDATE |
| C_btc7d500_short1m10 | 24 | 87.5% | +127.5 | +2015.2 | 11 | +70.2 | +1309.9 | +418.4 | -25.5 | 0 | 20 | PAPER_CANDIDATE |
| L_base_score1_added | 15 | 93.3% | +141.7 | +1082.6 | 6 | +98.3 | +738.2 | +128.0 | -49.2 | 0 | 14 | RESEARCH_ONLY_LOW_N |
| L_no_btc7d | 21 | 76.2% | +82.0 | +775.8 | 12 | +15.4 | +806.0 | +195.3 | -197.9 | 1 | 19 | REJECT_FOLD_T3R |
| S_btc1m_delay10 | 10 | 90.0% | +133.2 | +523.6 | 6 | +96.7 | +658.9 | +658.9 | -11.2 | 0 | 6 | RESEARCH_ONLY_LOW_N |
| S_current_btc2m_delay5 | 7 | 85.7% | +99.0 | -58.4 | 0 |  | -154.3 | -154.3 | -169.6 | 1 | 4 | RESEARCH_ONLY_LOW_N |
| SEQ_silence_no_btc1m | 166 | 60.8% | +37.9 | +4947.3 | 157 | +30.3 | +3222.9 | +1908.6 | -452.3 | 20 | 121 | PAPER_SHADOW_ONLY_TAIL |
| SEQ_noisy_with_btc1m | 48 | 22.9% | -129.4 | -6615.0 | 48 | -129.4 | -2602.0 | -2917.7 | -511.1 | 23 | 29 | REJECT_T3R |

## Navigation Tests
- baseline_sync200: N=100 | WR=58.0% | avg=+26.5 bps | /mo=23.0 | T3R=+1609.3
- danger_noisy_with_btc1m_long4h: N=48 | WR=22.9% | avg=-129.4 bps | /mo=11.6 | T3R=-6615.0
- safe_silence_no_btc1m_long4h: N=166 | WR=60.8% | avg=+37.9 bps | /mo=38.1 | T3R=+4947.3
- baseline_excluding_danger_noisy_btc1m: N=100 | WR=58.0% | avg=+26.5 bps | /mo=23.0 | T3R=+1609.3

## Key Interpretation
- Best raw candidate remains C_score_relax_short1m10, but it is still PAPER/SHADOW level because N is modest and it has a -100 bps tail.
- C_no_btc7d_short1m10 gets closest to 8 trades/month, but its added trades are materially weaker than the stricter score-relax candidate.
- Noisy+BTC-confirm is a strong DANGER navigation state for LONG, while silence without BTC confirm is the cleanest broad state label.
- No-overlap matters: frequency candidates must be evaluated as executable portfolios, not independent rows.
- No live changes were made; next step is forward-shadowing the leading candidates before promotion.