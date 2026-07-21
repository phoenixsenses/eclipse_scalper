# S34 State Machine V6 Development Ideas

- generated_at_utc: `2026-06-30T18:55:21.264187+00:00`
- research_only: `true`
- primary_hold: `{'n': 30, 'wr': 0.833, 'sum': 3471.4, 'mean': 115.7, 'median': 106.8, 't3r': 2411.8, 'max_loss': -52.0, 'max_win': 370.0, 'max_dd_bps': 52.0}`

## Executive Read

- No live changes made.
- Best immediate development lead: early momentum observer + arm-specific exit research.
- Best shadow-only expansion leads: score4 and BTC750; neither should replace the live rule yet.

## Key Results

- early_5m_fav_ge20: `{'n': 21, 'wr': 0.905, 'sum': 3282.4, 'mean': 156.3, 'median': 138.9, 't3r': 2222.7, 'max_loss': -40.2, 'max_win': 370.0, 'max_dd_bps': 40.2}`
- early_5m_fav_lt20: `{'n': 9, 'wr': 0.667, 'sum': 189.0, 'mean': 21.0, 'median': 16.5, 't3r': -63.1, 'max_loss': -52.0, 'max_win': 106.5, 'max_dd_bps': 64.0}`
- score4_shadow_hold: `{'n': 17, 'wr': 0.882, 'sum': 2493.2, 'mean': 146.7, 'median': 137.9, 't3r': 1448.9, 'max_loss': -40.2, 'max_win': 370.0, 'max_dd_bps': 40.2}`
- btc750_shadow_hold: `{'n': 32, 'wr': 0.781, 'sum': 3359.0, 'mean': 105.0, 'median': 72.8, 't3r': 2299.4, 'max_loss': -52.0, 'max_win': 370.0, 'max_dd_bps': 70.5}`
- confidence_sized_hold: `{'n': 30, 'wr': 0.833, 'sum': 4419.4, 'mean': 147.3, 'median': 93.6, 't3r': 2852.9, 'max_loss': -60.2, 'max_win': 555.0, 'max_dd_bps': 60.2}`

## Report JSON

- `D:\eclipse_scalper\reports\research\s34\S34_STATE_MACHINE_V6_DEVELOPMENT_IDEAS.json`
