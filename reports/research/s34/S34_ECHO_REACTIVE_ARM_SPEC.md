# Echo Reactive/Stop Arm Spec (forward, pre-declared grids)

_2026-07-20T10:31:19.341188+00:00 · READ-ONLY · causal N=118 · tails=14_

> PRE-DECLARED grids; TRADEOFF curves, NOT a chosen threshold. Forward locks the choice (OD-028/029). No edge claim. §163: stop caps magnitude not edge, cut whipsaws.

Baseline hold-4h: N=118 WR=69.5 mean=+41.2 sum=+4856.5 worst=-338.9 tail=14

## A) Mechanical stop grid (hold 4h)

| stop bps | N | WR | mean | sum | worst | tail_n | tail-only mean | tail-only worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 118 | 69.5 | +41.2 | +4856.5 | -338.9 | 14 | -186.8 | -338.9 |
| 100 | 118 | 61.9 | +30.5 | +3602.1 | -105.0 | 36 | -105.0 | -105.0 |
| 120 | 118 | 64.4 | +34.0 | +4006.1 | -125.0 | 30 | -125.0 | -125.0 |
| 150 | 118 | 67.8 | +36.6 | +4319.6 | -155.0 | 22 | -152.9 | -155.0 |
| 200 | 118 | 69.5 | +39.1 | +4609.5 | -205.0 | 16 | -179.2 | -205.0 |
| 250 | 118 | 69.5 | +39.2 | +4626.6 | -255.0 | 15 | -188.8 | -255.0 |

## B) Reactive trigger tradeoff (exit@T+k if be_ratio>=theta, else hold 4h)

| k | theta | n_cut | tails_cut | tail-catch | winners_whipsawed | reactive_sum | Δ vs hold |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.5 | 67 | 11 | 0.786 | 43 | +2293.7 | -2562.8 |
| 4 | 1.0 | 36 | 6 | 0.429 | 23 | +3558.2 | -1298.3 |
| 4 | 1.5 | 24 | 5 | 0.357 | 13 | +4161.1 | -695.4 |
| 4 | 2.0 | 21 | 5 | 0.357 | 11 | +4460.8 | -395.7 |
| 4 | 3.0 | 14 | 3 | 0.214 | 8 | +4284.0 | -572.6 |
| 5 | 0.5 | 67 | 11 | 0.786 | 43 | +2379.6 | -2476.9 |
| 5 | 1.0 | 37 | 7 | 0.5 | 23 | +3679.5 | -1177.0 |
| 5 | 1.5 | 25 | 6 | 0.429 | 13 | +4159.4 | -697.1 |
| 5 | 2.0 | 21 | 5 | 0.357 | 11 | +4381.6 | -474.9 |
| 5 | 3.0 | 14 | 3 | 0.214 | 8 | +4225.7 | -630.8 |
| 6 | 0.5 | 70 | 12 | 0.857 | 45 | +2358.5 | -2498.0 |
| 6 | 1.0 | 39 | 9 | 0.643 | 23 | +3956.3 | -900.2 |
| 6 | 1.5 | 26 | 7 | 0.5 | 13 | +4362.1 | -494.4 |
| 6 | 2.0 | 22 | 6 | 0.429 | 11 | +4588.2 | -268.3 |
| 6 | 3.0 | 15 | 4 | 0.286 | 8 | +4483.7 | -372.8 |
| 7 | 0.5 | 71 | 13 | 0.929 | 45 | +2293.9 | -2562.6 |
| 7 | 1.0 | 40 | 10 | 0.714 | 23 | +3841.0 | -1015.5 |
| 7 | 1.5 | 26 | 7 | 0.5 | 13 | +4206.4 | -650.1 |
| 7 | 2.0 | 22 | 6 | 0.429 | 11 | +4462.3 | -394.2 |
| 7 | 3.0 | 15 | 4 | 0.286 | 8 | +4348.2 | -508.3 |

_Δ vs hold > 0 means the reactive cut improved total net IN-SAMPLE (fragile, forward-only)._

## C) Multi-feature info-curve (AUC vs tail; sign-oriented so higher=more tail)

| k(min) | be_ratio | eth_own_pnl(benchmark) | btc_ret | new_btc_sell_cnt | new_btc_sell_$ | new_eth_sell_cnt |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.502 | 0.828 | 0.780 | 0.580 | 0.703 | 0.600 |
| 2 | 0.525 | 0.782 | 0.788 | 0.657 | 0.736 | 0.637 |
| 3 | 0.586 | 0.773 | 0.788 | 0.753 | 0.791 | 0.692 |
| 4 | 0.630 | 0.739 | 0.757 | 0.752 | 0.801 | 0.678 |
| 5 | 0.653 | 0.782 | 0.788 | 0.744 | 0.812 | 0.680 |
| 6 | 0.714 | 0.815 | 0.811 | 0.778 | 0.850 | 0.708 |
| 7 | 0.742 | 0.834 | 0.843 | 0.829 | 0.866 | 0.761 |
| 8 | 0.730 | 0.842 | 0.845 | 0.822 | 0.851 | 0.755 |
| 10 | 0.729 | 0.856 | 0.871 | 0.817 | 0.843 | 0.753 |

_If eth_own_pnl (watching your own position) matches/beats be_ratio, a plain price-stop already captures it and the external BTC-flush signal adds nothing._

## D) Late-flush deep dive (BTC in [ts,ts+10m])

| group | n | tail rate | mean net 4h | BTC ret@10m | new BTC sell cnt | new BTC sell $M | new ETH sell cnt |
|---|---:|---:|---:|---:|---:|---:|---:|
| LATE-FLUSH | 11 | 0.545 | -73.0 | -17.0 | 6 | 1.704 | 5 |
| STAYS-LOW | 48 | 0.042 | 58.5 | 6.6 | 0.0 | 0.088 | 1.0 |

## Read (forward spec, not adoption)
- A: pick the stop that caps tail worst without gutting mean — but §163 says it's a magnitude
  cap, not edge; forward confirms. B: any theta with Δ>0 AND low winners_whipsawed is a forward
  reactive candidate; the whole table is the tradeoff surface, forward locks (k,theta).
- C: compare be_ratio vs eth_own_pnl per k — external signal only earns an arm if it beats the
  self-position price-stop benchmark. D: names the mechanism (BTC liq buildup) behind late-flush.
