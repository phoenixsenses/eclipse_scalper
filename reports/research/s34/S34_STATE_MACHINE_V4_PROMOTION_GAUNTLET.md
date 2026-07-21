# S34 State Machine V4 Promotion Gauntlet

- generated_at_utc: `2026-06-30T17:21:50.240057+00:00`
- research_only: `true`

## Decision Read

- leading_config: `btc1000_dow_score3`
- leading_hold: `{'n': 30, 'wr': 0.833, 'sum': 3471.4, 'mean': 115.7, 'median': 106.8, 't3r': 2411.8, 'max_loss': -52.0, 'max_win': 370.0, 'max_dd_bps': 52.0}`
- blocker_1: `vdepth/live-feature rebuild is not proven 100%; current parity covers b4h/sync/n2h/book availability.`
- blocker_2: `timestamp/action realtime shadow parity is still only ID-level; live process uniqueness must be audited separately.`

## Final Config Duel

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| btc750_dow_score3 | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| btc1000_dow_score3 | 30 | 83.3% | 3471.4 | 115.7 | 106.8 | 2411.8 | -52.0 | 52.0 |
| btc750_score4 | 21 | 81.0% | 2977.9 | 141.8 | 138.9 | 1933.5 | -42.3 | 42.3 |
| btc750_dow_score4 | 17 | 82.4% | 2355.6 | 138.6 | 137.9 | 1311.3 | -40.2 | 40.2 |
| btc750_dow_score3_no_noisy | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| btc750_dow_score3_sl100 | 32 | 75.0% | 3074.0 | 96.1 | 52.3 | 2014.4 | -105.0 | 175.5 |
| btc750_dow_score3_sl150 | 32 | 78.1% | 3244.2 | 101.4 | 72.8 | 2184.6 | -155.0 | 155.0 |
| btc750_dow_score3_book_required | 30 | 70.0% | 3344.1 | 111.5 | 92.1 | 2276.0 | -53.7 | 73.0 |
| btc750_dow_score3_exclude_april_diag | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |

## April / Regime Killer Tests

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| exclude_april_diag | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| exclude_sat | 30 | 80.0% | 3341.6 | 111.4 | 91.6 | 2281.9 | -52.0 | 70.5 |
| exclude_tue_sat | 23 | 78.3% | 3061.1 | 133.1 | 137.9 | 2001.4 | -52.0 | 52.0 |
| score4_plus | 17 | 82.4% | 2355.6 | 138.6 | 137.9 | 1311.3 | -40.2 | 40.2 |
| btc1000_only | 30 | 83.3% | 3471.4 | 115.7 | 106.8 | 2411.8 | -52.0 | 52.0 |
| sl100 | 32 | 75.0% | 3074.0 | 96.1 | 52.3 | 2014.4 | -105.0 | 175.5 |
| sl150 | 32 | 78.1% | 3244.2 | 101.4 | 72.8 | 2184.6 | -155.0 | 155.0 |

## April Feature Cards

- april: `{'n': 13, 'summary': {'n': 13, 'wr': 0.308, 'sum': -173.7, 'mean': -13.4, 'median': -26.1, 't3r': -516.9, 'max_loss': -121.2, 'max_win': 158.0, 'max_dd_bps': 516.9}, 'avg_score': 3.69, 'avg_b4h': -17.3, 'avg_sync_k': 552980.1, 'avg_n2h': 5.1, 'sessions': {'ASIA': 2, 'EUROPE': 1, 'OFF': 2, 'US': 8}, 'states': {'SILENCE': 9, 'NEITHER': 4}}`
- non_april: `{'n': 64, 'summary': {'n': 64, 'wr': 0.734, 'sum': 5095.4, 'mean': 79.6, 'median': 57.8, 't3r': 4011.1, 'max_loss': -318.3, 'max_win': 370.0, 'max_dd_bps': 318.3}, 'avg_score': 3.91, 'avg_b4h': -111.6, 'avg_sync_k': 898173.8, 'avg_n2h': 11.3, 'sessions': {'ASIA': 18, 'EUROPE': 3, 'OFF': 3, 'US': 40}, 'states': {'SILENCE': 36, 'NEITHER': 28}}`

## Live Feature Rebuild Parity

- `{'n': 450, 'b4h_diff_median_bps': 0.6, 'b4h_diff_p95_bps': 3.3, 'sync_match_rate': 1.0, 'n2h_match_rate': 1.0, 'dow_match_rate': 1.0, 'book_available_rate_10s': 0.453, 'vdepth_note': 'vdepth_bps is not directly rebuilt here; live executor must either recompute from running anchor marks or reject if unavailable. This remains a live-feature blocker.'}`

## Rolling Kill Tests

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| roll3_sum_le_-150_pause_month | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| roll5_sum_le_-200_pause_month | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| roll5_sum_le_-300_pause_month | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| roll10_sum_le_-400_pause_month | 0 |  | 0.0 | None | None | 0.0 | None | 0.0 |

## Latency Stress

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| delay_0s | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| delay_5s | 32 | 71.9% | 3354.3 | 104.8 | 72.9 | 2303.2 | -61.9 | 63.0 |
| delay_15s | 32 | 71.9% | 3296.1 | 103.0 | 69.7 | 2265.2 | -65.1 | 65.1 |
| delay_30s | 32 | 81.2% | 3402.2 | 106.3 | 69.0 | 2359.5 | -59.1 | 59.1 |
| delay_60s | 32 | 78.1% | 3311.4 | 103.5 | 71.8 | 2242.2 | -62.4 | 62.4 |

## Slippage Stress

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slip_0bps | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| slip_5bps | 32 | 75.0% | 3199.0 | 100.0 | 67.8 | 2154.4 | -57.0 | 85.5 |
| slip_10bps | 32 | 68.8% | 3039.0 | 95.0 | 62.8 | 2009.4 | -62.0 | 100.5 |
| slip_20bps | 32 | 59.4% | 2719.0 | 85.0 | 52.8 | 1719.4 | -72.0 | 130.5 |
| slip_30bps | 32 | 59.4% | 2399.0 | 75.0 | 42.8 | 1429.4 | -82.0 | 160.5 |

## Risk Sizing

- `{'worst_bps': -318.3, 'equity_usdt': 35.0, 'leverage': 40.0, 'risk_1pct': {'risk_usdt': 0.35, 'max_notional_usdt': 11.0, 'margin_usdt_at_40x': 0.2749}, 'risk_2pct': {'risk_usdt': 0.7, 'max_notional_usdt': 21.99, 'margin_usdt_at_40x': 0.5498}, 'risk_5pct': {'risk_usdt': 1.75, 'max_notional_usdt': 54.98, 'margin_usdt_at_40x': 1.3745}, 'risk_10pct': {'risk_usdt': 3.5, 'max_notional_usdt': 109.96, 'margin_usdt_at_40x': 2.7489}}`

## Shadow / Readiness

- shadow_timestamp_parity: `{'exists': True, 'ledger_backfill_closes': 503, 'expected_ids': 503, 'matching_ids': 503, 'missing_expected_ids': 0, 'extra_ledger_ids': 0, 'parity_ratio': 1.0, 'note': 'ID-level parity only; P&L parity differs because backfill uses NAV labels while this suite recomputes mark/book outcomes.'}`
- readiness_readout: `{'live_state_exists': True, 'live_pid_file_exists': True, 'live_pid_file': '8680', 'live_active': None, 'live_status_rule': 'S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID', 'live_status_mode': 'LIVE', 'realtime_shadow_state_exists': True, 'realtime_shadow_open_positions': 0, 'shadow_ledger_exists': True, 'note': 'Read-only readiness snapshot. It does not prove process uniqueness; process cleanup/audit must be separate before live promotion.'}`

## Read

The state-machine candidate remains statistically strong. The remaining live blockers are operational: live feature parity for vdepth/book coverage, timestamp/action shadow parity beyond ID parity, and separate duplicate-process safety audit.
