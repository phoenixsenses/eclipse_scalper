# S34 V02 H4 Shadow Protocol

Status: `SHADOW_OBSERVATION_ONLY`

Protocol under observation:
`S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID`

This protocol does not authorize live order changes. It records how the current
V02 mirror would have behaved under H2/H3/H4 management buckets and companion
observers.

## Buckets

| Bucket | Meaning |
| --- | --- |
| `H2_CURRENT` | Existing two-hour mirror result. |
| `H3_SHADOW` | Same fill, three-hour time exit. |
| `H4_SHADOW` | Same fill, four-hour time exit. |
| `H4_CROSS_NO_DUMP_SHADOW` | Hold H4 only when BTC 30m > -40 bps and SOL 30m > -80 bps; otherwise H2. |

## Observers

- Cross-no-dump: classifies whether BTC/SOL avoided a post-fill dump.
- Catastrophic stop: reports SL100/125/150/175/200 touch behavior and proxy PnL.
- State machine v2: labels the path from anchor to fill, pain, rebound, cross-state, and runner outcome.
- Queue/fill realism: top-of-book proxy only. True queue position still requires tick/order-book replay.
- Live/shadow parity: read-only check that mirror configuration matches the armed live rule identity.

## Promotion Gate

This protocol can only become a paper-candidate after fresh forward data:

- at least 30 closed forward fills,
- at least 30 calendar days,
- forward sum > 0,
- forward top-3-winner-removed sum > 0,
- no single winner carries the sample,
- live/shadow parity remains `PASS`,
- explicit operator approval before any live order logic changes.

Until then, H4 is a navigation/management hypothesis only.
