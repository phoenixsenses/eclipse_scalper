# Lane Shadow Specs - 2026-06-02

All specs are `SHADOW_ONLY`. They must not place orders or change live routing.

## ETH_BUY250K_SHORT_900_UTC14

| field | value |
| --- | --- |
| symbol | `ETHUSDT` |
| event | forced liquidation |
| liquidation side | `BUY` |
| liquidation notional | `>= 250000` |
| lane | UTC hour `14` |
| direction | `SHORT` |
| horizon | `900s` |
| current evidence | 34 events, 76.47% WR, +43.28 bps mean |

Use this as a narrow shadow alpha candidate. It is high value but time-lane specific.

## ETH_BUY500K_SHORT_900_SESSION_US

| field | value |
| --- | --- |
| symbol | `ETHUSDT` |
| event | forced liquidation |
| liquidation side | `BUY` |
| liquidation notional | `>= 500000` |
| lane | UTC hour `14 <= hour < 21` |
| direction | `SHORT` |
| horizon | `900s` |
| current evidence | 62 events, 72.58% WR, +25.86 bps mean |

Use this as the broader ETH forced-flow shadow candidate. It is less sharp than UTC14 but more robust.

## SOL_BUY50K_SHORT_900_FUNDING_NEGATIVE

| field | value |
| --- | --- |
| symbol | `SOLUSDT` |
| event | forced liquidation |
| liquidation side | `BUY` |
| liquidation notional | `>= 50000` |
| lane | latest funding rate `< 0` |
| direction | `SHORT` |
| horizon | `900s` |
| current evidence | 20 events, 85.00% WR, +31.83 bps mean |

Use this as a refinement on `SOL_BUY_LIQ_SHORT_V1`. It should be tracked separately from the broader SOL candidate.

## S34_SHORT_900_SESSION_US

| field | value |
| --- | --- |
| source | `detector_signals` |
| symbol | `ETHUSDT` |
| event | S34 detector signal |
| lane | UTC hour `14 <= hour < 21` |
| direction | `SHORT` |
| horizon | `900s` |
| current evidence | 25 events, 72.00% WR, +29.01 bps mean |

Use this as a time-lane quality filter for S34.

## S34_SHORT_900_BASIS_POSITIVE

| field | value |
| --- | --- |
| source | `detector_signals` |
| symbol | `ETHUSDT` |
| event | S34 detector signal |
| lane | `basis_at_entry > 0` |
| direction | `SHORT` |
| horizon | `900s` |
| current evidence | 31 events, 80.65% WR, +27.38 bps mean |

Use this as a quality filter for S34. It reinforces the prior basis-positive branch.

## Logging Requirements

Every shadow signal should record:

- `signal_family`
- `status = SHADOW_ONLY`
- `ts_ms`
- `symbol`
- `direction`
- `horizon_sec`
- trigger source fields, including liquidation notional or detector signal id
- lane fields that caused inclusion
- mark price at trigger
- forward returns at `60s`, `120s`, `300s`, and `900s`
- fee-stressed net estimates at 2, 4, 8, and 10 bps round trip
- overlap flags for ETH/BTC/SOL forced-flow within 60s

## Promotion Gates

Do not promote beyond shadow unless forward-only data satisfies:

- at least 100 new events for the specific lane
- at least 5 chronological folds
- at least 4/5 positive folds after 8 bps round-trip cost
- positive performance both with and without cross-asset forced-flow overlap
- no mature fold worse than -5 bps mean once it has at least 20 events

