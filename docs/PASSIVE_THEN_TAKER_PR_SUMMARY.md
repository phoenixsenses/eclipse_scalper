# PASSIVE_THEN_TAKER PR Summary

Date:
- 2026-03-09

Branch:
- `codex/research/pocket-promotion-checklist`

Summary:
- `passive_then_taker` remains `experimental_on`
- promotion case is now narrower and more honest
- this is not an ETH 60s-wide fix
- this is an ETH 60s tighter-subfamily experimental candidate

Scope supported by evidence:
- `ETHUSDT`
- `micro_edge_v3_passive_alpha`
- `h=60`
- tighter pockets first

Core result:
- execution shape effect is real
- the effect is strongest where spread is tighter and/or intensity is higher
- the effect does not generalize evenly across the full ETH 60s family

Pocket read:
- `Pocket B`: real rescue, `1/3 -> 3/3`, net flips from `-2.044 bps` to `+0.902 bps`
- `Pocket C`: strongest seen pocket, `3/3`, net `+3.792 bps`
- `Tight-mid`: real flip, `0/3 -> 3/3`, net `-2.029 bps -> +2.132 bps`
- `Soft`: partial rescue only, `0/3 -> 2/3`, net barely positive at `+0.049 bps`
- `Mid`: fillability improves to `100%`, but pocket still fails `0/3` and stays negative

Decision:
- promote only as a narrow ETH 60s tighter-pocket experiment
- keep softer-mid pockets out of promotion language
- do not frame this as a broad ETH execution default

Recommended PR language:
- `passive_then_taker improves ETH 60s execution quality inside a tighter pocket subfamily`
- `evidence supports experimental rollout for tighter ETH pockets, not family-wide promotion`

Not supported yet:
- BTC replication
- broad-surface ETH claims
- default execution replacement

References:
- `docs/PASSIVE_THEN_TAKER_DECISION.md`
- `docs/PASSIVE_THEN_TAKER_ETH60_FAMILY_MAP.md`
