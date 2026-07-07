# FAM_BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1

**Family:** `FAM_BOOK_SPREAD_DYNAMICS` · **Child:** `H-BOOK-SPREAD-CHANGE-BPS-W300-V1`
**Freeze version:** `BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1`
**Status:** Immutable, outcome-blind row-accounting and lineage freeze. Binds the operator-approved definition to one exact ordered reproducible anchor population and one exact set of selected source quotes, before any canonical migration or outcome-linked work.
**Depends on:** operator ruling `FAM_BOOK_SPREAD_DYNAMICS_PRIMARY_DEFINITION_V1`; rehearsal commit `6a449a64`; readiness commit `f115b9c1`.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Family / child identity and operator-approved definition

- **Family:** `FAM_BOOK_SPREAD_DYNAMICS` · **Child working ID:** `H-BOOK-SPREAD-CHANGE-BPS-W300-V1`
- **Formula version:** `BOOK_SPREAD_CHANGE_BPS_W300_V1` · **Specification hash:** `ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212`
- **Formula:** `spread_change_bps_w300 = spread_bps(t0) − spread_bps(t0 − 300s)`; `mid=(ask+bid)/2`; `spread_bps=1e4·(ask−bid)/mid`. Sign: + expansion, − compression, 0 unchanged. Units: bps of spread change.
- **Source:** exact `book_ticker` L1 best bid/ask (`BINANCE_USDM_PERP`, ETHUSDT, `PERPETUAL_FUTURES`, USDT). **W300 only.** Additive difference only.

## Immutable scope (frozen for V1)

Frozen immutable: family ID, child ID, formula version, W300 window, additive spread-bps difference, source table/fields, endpoint selection rules, `id`-DESC tie-break, 5-minute staleness, locked/crossed/zero-negative policies, source-quality precedence (`UNAVAILABLE > ZERO_NEG > CROSSED > LOCKED > STALE`), the ordered 324-anchor universe, the exact 196-row eligible set, the 128-row excluded set, quality classes, exclusion reasons, selected source quote IDs, feature values, cycle membership, the 97 cycle representatives, the specification hash, all manifest hashes, and the root hash.

## Ordered manifest identities and full hashes

| Manifest | Count | sha256 |
|---|---|---|
| Ordered anchor manifest | 324 | `a77a8daf2a8d198d775436674a20a9bd5328dc071e2883938b7c331c17c534bb` |
| Exact feature manifest | 196 | `b1eb902f5b3d1ea0f19b4b60d0ad999907a042b228adf506bbe09800a81e155b` |
| Exclusion manifest | 128 | `0694e43300710e1204c1b23643d9eacb9f10188c21aa0ceda572c28229cc8449` |
| Cycle-membership manifest | 196 | `e692ff1c8ce37b54a3349a501a38bd44f24865e75a51accc81c7e97399d29e18` |
| Representative manifest | 97 | `edadf5972cbbdddb0efa1db8234473ee089972f504d3bfbfafbae508238db246` |
| Accepted rehearsal content hash | — | `5e9ee58cd9c260c2877b05ed803dbf51767ecedc579bdc90c37b5391a867bcbb` |
| Accepted rehearsal row-manifest hash | — | `8e8e23ff8af6dfd1c11199f963698d4a148583fd2b9c979dffa7f4e4fdec72f2` |
| Specification hash | — | `ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212` |

**Ordering policy:** `signal_birth_ts ASC, anchor_id ASC` (immutable anchor fields only). **Serialization policy:** per-field `repr()` (full float precision); fields joined U+001F; records joined U+001E; sha256. Committed detail manifest: `reports/research/s34/S34_BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1_MANIFEST.json` (sha256 `0a65c45ffba906414c7a484e3f966e2405017eaea8990aded429dc35ed142c89`).

## Root hash

**`BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1_ROOT` = `33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31`**

Definition: sha256 over the sorted `name=full-hash` pairs of the 8-component set (5 ordered manifests + rehearsal content-hash + rehearsal row-manifest-hash + specification-hash), joined by U+001E. Commits to scientific identity, specification identity, anchor membership, selected source quote IDs, numerical values, quality classes, exclusions, cycle membership, cycle representatives, serialization policy, and ordering policy. No truncated hash is used as an identity field; the `33c4f4be` prefix is a human-readable summary only.

## Accounting (reconciles exactly)

- `324 = 196 EXACT_RECONSTRUCTABLE + 22 STALE_SOURCE + 106 UNAVAILABLE_BEFORE_COLLECTION`
- Non-exact `128 = 22 + 106`; 0 crossed/locked/zero/gapped/repaired/proxy/mismatch/duplicate.
- 196 exact anchors → 97 independent cycles → 97 representatives (0 duplicate, 0 missing, 0 cycles with >1).
- Every exact anchor has exactly one feature value; no excluded anchor has one.

Proven by two fresh independent replays, each byte-identical to the other and to the accepted rehearsal evidence.

## Amendment policy

Any later change to any frozen field requires: (1) a new version, (2) a new explicit operator authorization, (3) a new freeze artifact, (4) a full explanation of why the prior version changed, and (5) **no silent mutation of V1**.

## Future-repair policy

Post-freeze source repairs must **not** rewrite V1 retroactively. Any repaired future version must remain separately versioned and must disclose its relationship to V1.

## Prohibited mutations (V1)

No alternative window (W60/W600/W1800/W3600), no alternative transform (level/ratio/log-ratio/z-score/quantile/bin/threshold/sign-only/clipped/winsorized/smoothed/path-stat/persistence/reversion/nonlinear/interaction), no source-quality-policy change, no cycle-representative-rule change, no outcome linkage.

## Next controlled gate

`BATCH-BOOK-SPREAD-DYNAMICS-CANONICAL-MIGRATION-V1` — **not begun automatically.** It must reuse this root hash, preserve the exact 324/196/128/97 accounting, remain outcome-blind, create no experiment/nullifier/gate-receipt, perform no preregistration/TEST access, use a dedicated migration ID assigned only inside that authorized gate, and prove idempotency + canonical immutability.
