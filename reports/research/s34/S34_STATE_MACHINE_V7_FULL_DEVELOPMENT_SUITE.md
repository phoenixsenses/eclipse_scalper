# S34 State Machine V7 Full Development Suite

- generated_at_utc: `2026-06-30T19:01:52.355483+00:00`
- research_only: `true`
- live_changes: `none`
- primary_config: `btc1000_dow_score3`
- primary_hold: `N=30 WR=83.3% sum=3471.4 mean=115.7 med=106.8 T3R=2411.8 maxL=-52.0 DD=52.0`

## Questions Tested

1. Early momentum threshold grid.
2. Profit-lock exit grid.
3. Arm-specific horizon grid.
4. Score/confidence monotonicity.
5. BTC threshold dose-response.
6. BTC/ETH divergence navigation.
7. Regime pause/kill simulation.
8. Session-specific management.
9. Volatility context.
10. Shadow-candidate shortlist.

## Shortlist By Holdout T3R

- confidence_sized: `N=30 WR=83.3% sum=4419.4 mean=147.3 med=93.6 T3R=2852.9 maxL=-60.2 DD=60.2` caution=candidate
- profit_lock_trig100_lock50: `N=30 WR=86.7% sum=3700.6 mean=123.4 med=106.8 T3R=2640.9 maxL=-52.0 DD=52.0` caution=candidate
- primary_live_baseline: `N=30 WR=83.3% sum=3471.4 mean=115.7 med=106.8 T3R=2411.8 maxL=-52.0 DD=52.0` caution=candidate
- btc750_shadow: `N=32 WR=78.1% sum=3359.0 mean=105.0 med=72.8 T3R=2299.4 maxL=-52.0 DD=70.5` caution=candidate
- early_5m_fav20: `N=21 WR=90.5% sum=3282.4 mean=156.3 med=138.9 T3R=2222.7 maxL=-40.2 DD=40.2` caution=candidate
- score4_shadow: `N=17 WR=88.2% sum=2493.2 mean=146.7 med=137.9 T3R=1448.9 maxL=-40.2 DD=40.2` caution=small_N
- early_5m_fav20_clean: `N=12 WR=91.7% sum=2242.8 mean=186.9 med=208.0 T3R=1238.6 maxL=-40.2 DD=40.2` caution=small_N
- low_vol: `N=16 WR=87.5% sum=2181.7 mean=136.4 med=122.7 T3R=1222.8 maxL=-51.4 DD=51.4` caution=small_N
- eth_weaker_than_btc: `N=8 WR=100.0% sum=1380.2 mean=172.5 med=157.1 T3R=465.1 maxL=6.1 DD=0.0` caution=small_N

## Selected Results

- early 5m fav>=20: `N=21 WR=90.5% sum=3282.4 mean=156.3 med=138.9 T3R=2222.7 maxL=-40.2 DD=40.2`
- early 5m fav>=20 clean: `N=12 WR=91.7% sum=2242.8 mean=186.9 med=208.0 T3R=1238.6 maxL=-40.2 DD=40.2`
- profit lock 100/50: `N=30 WR=86.7% sum=3700.6 mean=123.4 med=106.8 T3R=2640.9 maxL=-52.0 DD=52.0`
- BTC 1000K: `N=30 WR=83.3% sum=3471.4 mean=115.7 med=106.8 T3R=2411.8 maxL=-52.0 DD=52.0`
- BTC 750K: `N=32 WR=78.1% sum=3359.0 mean=105.0 med=72.8 T3R=2299.4 maxL=-52.0 DD=70.5`
- low vol: `N=16 WR=87.5% sum=2181.7 mean=136.4 med=122.7 T3R=1222.8 maxL=-51.4 DD=51.4`
- ETH weaker than BTC: `N=8 WR=100.0% sum=1380.2 mean=172.5 med=157.1 T3R=465.1 maxL=6.1 DD=0.0`

## Full JSON

- `D:\eclipse_scalper\reports\research\s34\S34_STATE_MACHINE_V7_FULL_DEVELOPMENT_SUITE.json`
