# Phase 03B — E-DER V1 → `eclipse.alpha.trade_candidate` contract mapping

**Scope:** E-DER V1 only. Mapping and adapter. **The live path is not wired.**
**Frozen decisions this implements:** hard source-side seal · fail-closed suppression with local
ledger unchanged · producer-owned `arm_version` · editable install, no `sys.path` mutation ·
P2 absolute (no backfill).

---

## 1. The source of truth

`tools/e_der_v1_forward_shadow.py` — role `e_der_v1_forward_runner`, launched as
`python -m tools.e_der_v1_forward_shadow`, mode `FROZEN_V1_PAPER_SHADOW_NO_ORDERS`.

Two functions matter and they are cleanly separated:

| Function | When | Produces |
|---|---|---|
| `make_event(row, state, universe)` | **T0**, at detection | the `DETECTED` event — the only thing the adapter may see |
| `mature(state, cache, now)` | T+31m entry, T+240m boundary | `ENTRY` / `CLOSE` with realised returns — **the adapter must never see this** |

---

## 2. The hazard this mapping exists to prevent

`run_cycle` does exactly this:

```python
event = make_event(row, state, universe)
if event["event_id"] not in state["pending"] and event["event_id"] not in state["closed"]:
    state["pending"][event["event_id"]] = event    # ← stored BY REFERENCE
    append_jsonl(LEDGER, event)
mature(state, cache, now)                          # ← mutates that same dict in place
```

`mature()` calls `event.update({... "gross_return_bps": gross, "net_return_bps": gross-10.0 ...})`
**on the same object** the adapter would be holding.

Worse, `make_event` already returns the outcome keys, present and `None`:

```python
"entry_open": None, "boundary_open": None,
"gross_return_bps": None, "net_return_bps": None,
```

**Consequence:** any adapter that keeps a reference, queues an event, or publishes lazily will
publish realised returns from a sealed arm. This is not theoretical — it is one `await` away.

**Therefore the adapter is defined as a pure, synchronous, snapshot-taking function.** It copies
the fields it needs at call time, returns a validated immutable `TradeCandidate`, and holds no
reference to the caller's dict. It also refuses any event that is not a fresh `DETECTED`.

---

## 3. Field mapping

`TradeCandidate` is frozen: `candidate_id, arm, arm_version, anchor_id, symbol, direction,
horizon_minutes, context` — and `extra="forbid"`.

| TradeCandidate | Source | Notes |
|---|---|---|
| `candidate_id` | `event["event_id"]` — `E:{symbol}:{anchor_ts}` | Already unique and stable; no new id minted |
| `arm` | `"E-DER-V1"` (manifest constant) | The frozen arm's public name |
| `arm_version` | `PROTOCOL` = `E_DER_V1_PROSPECTIVE_FORWARD_2026_08_21` | **Existing authoritative constant** in the runner. Not Git SHA, not branch, not filename, not Master Center state |
| `anchor_id` | `str(event["anchor_ts"])` | The anchor instant, ms epoch |
| `symbol` | `event["symbol"]` | |
| `direction` | `LONG` (manifest constant) | The event carries **no** direction field. V1 is a rebound after down-pressure, so direction is a declared property of the frozen arm — asserted in the manifest, never inferred per event |
| `horizon_minutes` | `(boundary_ms − entry_ms) / 60000` | Computed from the event's own frozen timing. Both timestamps are known at T0 |
| `context` | strict whitelist, below | `dict[str, str]`, so every value is stringified |

### `context` whitelist — T0-only, additions require review

`q_parent` · `q_echo` · `prior_stress_count` · `multiscale_vote_sum` · `parent_id` ·
`parent_ts_ms` · `echo_id` · `cascade_id` · `product_cohort` · `session_state` ·
`universe_version` · `protocol` · `classification` · `paper_only` · `base_ms` · `entry_ms` ·
`boundary_ms` · `integration_contract`

The whitelist is **closed**: any key not on it is dropped, and the adapter fails if a forbidden key
appears at all.

### Forbidden — hard failure, not filtering

`entry_open` · `boundary_open` · `gross_return_bps` · `net_return_bps` · `updated_at_utc` ·
any `status` other than `AWAITING_ENTRY` · any `event` other than `DETECTED`.

Also deliberately excluded, though harmless: `code_sha` (Git SHA — explicitly barred from identity),
`contract_sha`, `cost_bps`, `data_quality_status`, `real_order_sent`, `listing_age_days`,
`created_at_utc`. Minimal is safer; each addition is a review.

> **Publication boundary:** the frozen entry/boundary offsets are legitimate internal detail and are
> used here to compute a horizon. They are covered by W1 and **must never appear on `web/`.**

---

## 4. Fail-closed and P2

* **Health.** Publication is attempted only when the Alpha agent and every declared dependency are
  healthy (P11, strict, no diagnostic exemption). When not, the adapter's caller suppresses the
  publish. **The local ledger path is untouched either way** — `append_jsonl(LEDGER, event)` and
  `mature()` continue exactly as today. Research behaviour does not depend on the bus.
* **P2.** Only events detected after the integration boundary timestamp are eligible. The adapter
  refuses an event whose `anchor_ts` precedes the boundary recorded in the manifest, so a restart
  that replays state cannot backfill onto the bus.

---

## 5. Files

**Added** (new package — nothing existing is modified):

```
integrations/eclipse_alpha/__init__.py
integrations/eclipse_alpha/manifest.py          immutable integration manifest
integrations/eclipse_alpha/candidate_adapter.py the pure mapping function
tests/test_eclipse_alpha_adapter.py             acceptance tests
.gitignore                                       one line: .venv/
```

**Read but never modified:** `tools/e_der_v1_forward_shadow.py`, `tools/e_der_v1_frozen.py`.

**Explicitly not touched:** `start_eclipse.ps1`, every role, every ledger, `execution/`, `risk/`,
`brain/`, `.env`, `tools/s34_state_machine_live_executor.py`.

**Dependency:** `eclipse-shared` editable-installed into `D:\eclipse_scalper\.venv`, resolving to
`D:\eclipse_platform\eclipse-shared`. No code copied, no `sys.path` mutation, no global install.
The 27 live roles still run on the system interpreter and are unaffected — **repointing them is a
separate, gated decision that touches `start_eclipse.ps1`.**

---

## 6. Acceptance criteria → tests

| # | Criterion | Test |
|---|---|---|
| 1 | No existing Scalper file modified | reviewed in the diff |
| 2 | No role behaviour changes | nothing imports the adapter yet |
| 3 | Produces a schema-valid `TradeCandidate`; a stray `size` is rejected | `test_maps_a_detected_event`, `test_sizing_fields_are_impossible` |
| 4 | No code path reads an outcome field | `test_refuses_a_matured_event`, `test_refuses_when_outcome_fields_are_populated`, `test_adapter_source_never_names_an_outcome_field` |
| 5 | No transport, socket, broker or persistence | `test_adapter_imports_nothing_forbidden` |
| 6 | Does not import `eclipse_master_center` | same test |
| 7 | Snapshot isolation — later mutation cannot reach the candidate | `test_mutating_the_source_event_afterwards_cannot_change_the_candidate` |
| 8 | P2 — pre-boundary anchors refused | `test_refuses_an_anchor_before_the_integration_boundary` |

---

## 7. Not done in 03B

No wiring. Nothing calls the adapter. No transport, no NATS, no persistence, no Execution, no
Data/Observability carve-out, no A2/V3.
