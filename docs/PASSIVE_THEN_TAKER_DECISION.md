# PASSIVE_THEN_TAKER Decision

Date:
- 2026-03-09

Branch:
- `codex/research/pocket-promotion-checklist`

Question:
- does `passive_then_taker` rescue the current weak ETH 60s passive-realistic surface better than pure passive execution?

Scope:
- symbol:
  - `ETHUSDT`
- rule:
  - `micro_edge_v3_passive_alpha`
- lookback:
  - `7D`
- current evidence:
  - focused ETH 60s pocket family only

Decision Contract
-----------------

Primary metrics:
- `pass_count`
- `filled_avg_net_mean`

Secondary metrics:
- `attempt_fill_rate_mean`
- `insufficient_fill_rate`
- `filled_n_mean`

Success criteria:
- `pass_count` strictly improves over baseline
- `filled_avg_net_mean` improves over baseline
- `attempt_fill_rate_mean` materially improves or stays near `100%`

Failure / freeze criteria:
- no improvement versus baseline
- or fillability improves but `filled_avg_net_mean` stays structurally negative
- or benefit appears only in one isolated pocket and disappears in adjacent pockets

Outcome labels:
- `experimental_on`
- `observe_only`
- `keep_baseline`

## Evidence

### Pocket B
- scope:
  - `h=60`
  - `imb>=0.30`
  - `int>=8000`
  - `spr<=0.000200`
  - `splits=2`
  - `min_n=20`

Baseline:
- artifact:
  - `reports/ETH_POCKET_B_7D_BASELINE_SPLIT2.json`
- result:
  - `pass_count = 1/3`
  - `filled_avg_net_mean = -2.043730e-04`
  - `attempt_fill_rate_mean = 79.97%`

Passive-then-taker:
- artifact:
  - `reports/ETH_POCKET_B_7D_PASSIVE_THEN_TAKER.json`
- result:
  - `pass_count = 3/3`
  - `filled_avg_net_mean = +9.015886e-05`
  - `attempt_fill_rate_mean = 100%`

Interpretation:
- strong improvement
- this is a real rescue, not just a marginal fill tweak

### Pocket C
- scope:
  - `h=60`
  - `imb>=0.30`
  - `int>=8000`
  - `spr<=0.000250`
  - `splits=2`
  - `min_n=20`

Passive-then-taker:
- artifact:
  - `reports/ETH_POCKET_C_7D_PASSIVE_THEN_TAKER.json`
- result:
  - `pass_count = 3/3`
  - `filled_avg_net_mean = +3.791728e-04`
  - `attempt_fill_rate_mean = 100%`

Interpretation:
- confirms the effect on an adjacent pocket
- stronger than pocket B

### Soft ETH pocket
- scope:
  - `h=60`
  - `imb>=0.40`
  - `int>=2500`
  - `spr<=0.000300`
  - `splits=2`
  - `min_n=20`

Baseline:
- artifact:
  - `reports/ETH_POCKET_SOFT_7D_BASELINE.json`
- result:
  - `pass_count = 0/3`
  - `filled_avg_net_mean = -2.775605e-04`
  - `attempt_fill_rate_mean = 55.91%`

Passive-then-taker:
- artifact:
  - `reports/ETH_POCKET_SOFT_7D_PASSIVE_THEN_TAKER.json`
- result:
  - `pass_count = 2/3`
  - `filled_avg_net_mean = +4.934115e-06`
  - `attempt_fill_rate_mean = 100%`

Interpretation:
- benefit persists outside the tight high-intensity subfamily
- edge is weaker, but still directionally positive versus baseline

### Mid ETH pocket
- scope:
  - `h=60`
  - `imb>=0.50`
  - `int>=3500`
  - `spr<=0.000300`
  - `splits=2`
  - `min_n=20`

Baseline:
- artifact:
  - `reports/ETH_POCKET_MID_7D_BASELINE.json`
- result:
  - `pass_count = 0/3`
  - `filled_avg_net_mean = -4.311250e-04`
  - `attempt_fill_rate_mean = 57.76%`

Passive-then-taker:
- artifact:
  - `reports/ETH_POCKET_MID_7D_PASSIVE_THEN_TAKER.json`
- result:
  - `pass_count = 0/3`
  - `filled_avg_net_mean = -4.930262e-05`
  - `attempt_fill_rate_mean = 100%`

Interpretation:
- fillability improves materially
- loss magnitude shrinks sharply
- but this pocket still does not pass
- this is not promotable as a passing pocket

### Tight-mid ETH pocket
- scope:
  - `h=60`
  - `imb>=0.50`
  - `int>=5000`
  - `spr<=0.000250`
  - `splits=2`
  - `min_n=20`

Baseline:
- artifact:
  - `reports/ETH_POCKET_TIGHTMID_7D_BASELINE.json`
- result:
  - `pass_count = 0/3`
  - `filled_avg_net_mean = -2.028754e-04`
  - `attempt_fill_rate_mean = 60.73%`

Passive-then-taker:
- artifact:
  - `reports/ETH_POCKET_TIGHTMID_7D_PASSIVE_THEN_TAKER.json`
- result:
  - `pass_count = 3/3`
  - `filled_avg_net_mean = +2.131802e-04`
  - `attempt_fill_rate_mean = 100%`

Interpretation:
- this confirms the rescue is strongest on the tighter, higher-intensity subfamily
- this is a real flip from non-passing to fully passing

## Decision

Current status:
- `passive_then_taker = experimental_on`

Why:
- it fully rescues multiple tighter ETH 60s pockets
- it improves the soft ETH pocket enough to become partially passing
- it improves fillability consistently toward `100%`
- but it does not rescue every softer-mid pocket in the family

What this does **not** mean:
- not a global execution default
- not yet validated for BTC
- not yet validated for long-window broad surfaces

## Recommended rollout scope

Safe experimental scope:
- `ETHUSDT`
- `micro_edge_v3_passive_alpha`
- `h=60`
- tighter pocket family first:
  - stronger imbalance and/or stronger intensity
  - tighter spread caps

Preferred order:
1. keep baseline ranking as the reference
2. test `passive_then_taker` as an experimental execution profile on the tighter ETH 60s pocket family
3. keep softer-mid pockets in `observe_only` until they produce passing results

## Freeze conditions

Freeze this line if:
- tighter ETH 60s family mapping turns mixed/negative
- BTC replication turns clearly negative
- current `7D` effect disappears on the next refreshed window

## Next step

Run a small ETH 60s family map:
- vary:
  - `min_imbalance`
  - `min_trade_intensity`
  - `max_spread`
- compare:
  - `passive_realistic`
  - `passive_then_taker`
- then promote only if the median family result stays positive

Current map artifact:
- `docs/PASSIVE_THEN_TAKER_ETH60_FAMILY_MAP.md`

Map read:
- tighter spread / higher-intensity pockets are promotable experimental candidates
- softer-mid pockets remain `observe_only` or `do_not_promote`
