# REPRODUCIBILITY

A result that cannot be reproduced is not evidence. Everything below is a contract the
repository holds itself to, not a description of good intentions.

---

## 1. Determinism is a tested contract

The research plane must satisfy: **same inputs plus same seed produce identical outputs.
Always.**

The rules that make that true:

| Rule | What it forbids |
|---|---|
| Seeded randomness only | any random call whose seed is not traceable to an explicit parameter |
| Stable event identity | ids derived from a fresh UUID, a process id, or wall-clock time |
| No wall-clock dependence | `now()` or `time()` anywhere in scoring or simulation logic — time comes from the data |
| Stable sort and aggregation | relying on dict insertion order, or any grouping whose tie-break is undefined |
| Seed echo in outputs | a report that does not state the seed it was produced with |
| No seeded-to-unseeded path | a seeded function calling an unseeded one that affects its output |

These are enforced as invariants (`DAT-03`, `VAL-02`) with named test files, not as a
style guide. The detection procedure is blunt on purpose: run the same command twice with
the same seed and diff. Any difference is a violation.

## 2. No lookahead, as a tested contract

A signal at time `t` may use only data with index ≤ `t` (`DAT-01`).

The violations that actually happen, and are checked for:

- a centred rolling window, which reads the future by construction
- a label computed from a later bar, then used as a feature earlier
- forward-filling before a feature is computed, which pushes future data backwards
- a negative shift on a feature column

The related timing contract (`DAT-02`) is that `signal index < entry index < exit index`,
and that the same entry convention holds across feature computation and backtest. Two
components using different conventions produce numbers that look comparable and are not.

## 3. Cost units are a tested contract

Basis points and ratios are different things and the conversion is applied exactly once
(`DAT-04`). Applying it twice is an order-of-magnitude error that can invert a conclusion
while leaving every intermediate number looking plausible.

Separately and honestly: the repository's **contradiction register records that the fee
constant is not uniform across all active research code paths**, and that results
computed on different bases are therefore not cross-comparable until the real tier is
confirmed in writing. That is tracked as an open contradiction rather than quietly
normalised — which is the point of having a register.

## 4. True forward splits are a tested contract

Validation uses a future slice with no overlap leakage from discovery (`VAL-01`), and
the validator is checked against synthetic collapse cases — inputs constructed so that a
broken splitter would pass and a correct one must fail.

## 5. Frozen artifacts and content hashes

Research governance artifacts are inventoried with byte counts and SHA-256 digests:

- `docs/research/RESEARCH_GOVERNANCE_MANIFEST_SHA256.csv` — the governance corpus
- `docs/research/audits/data_feasibility_v1/AUDIT_ARTIFACT_MANIFEST_SHA256.csv` — one
  audit's evidence set

A manifest is only worth something if it is rebuilt as the **last** action of a change,
after every artifact it covers has stopped moving. Rebuilding it mid-change produces a
manifest that certifies a state that never existed.

## 6. Study fingerprints

A study's identity is a fingerprint over **every component of its specification**. If any
component changes after an epoch has begun, that is a **new version and a new epoch** —
not an amendment.

The property being bought is not discipline; it is structure. A silent amendment is
impossible to make, because changing any component changes the fingerprint, and the
changed fingerprint is visible without anyone having to remember to look.

## 7. Preregistration and the burned sample

A rule is frozen before its outcome is opened. A window used to develop a rule is spent
and cannot later validate it. When a frozen object changes materially, the forward count
restarts at zero rather than continuing.

The consequence is deliberately expensive: improving a frozen rule costs you all the
forward evidence you had accumulated for the old one. Which is the correct price, and is
why the freeze is meaningful.

## 8. Append-only correction

- Superseded statements are **marked as superseded, not erased**.
- Prior canonical versions are retained under a history directory.
- The errata ledger is **append-only** — a correction never edits the source it
  corrects.
- Every material change carries a dated changelog entry and a decision-log entry.

A record that edits away its own errors cannot be audited, and stops being a record.

## 9. Canonical identity, stable across renumbering

Sections and studies are addressed by a stable identity — study, lane and UUID — rather
than by a display number. Display aliases may duplicate or drift; the canonical reference
does not, and historical numbering is never rewritten to tidy it.

This came out of a real defect: a namespace audit found dozens of duplicate section
aliases spread across hundreds of numbers, systemic and historical rather than a recent
accident. The fix was identity, not renumbering.

## 10. Reproducing something yourself

Deterministic, offline, and safe to run — no exchange credentials, no live connection,
no database required:

```bash
pip install -r requirements.txt

# the full suite
pytest -q

# determinism and no-lookahead contracts
pytest -q tests/test_micro_edge_alignment.py
pytest -q tests/test_passive_realistic_sim.py

# cost-unit correctness
pytest -q tests/test_exec_cost_models.py

# forward-split integrity and ranking reproducibility
pytest -q tests/test_validate_micro_edge_forward.py
pytest -q tests/test_rank_passive_pockets_forward.py

# execution invariants
pytest -q tests/test_order_router_idempotency.py
pytest -q tests/test_order_router_intent_lifecycle.py
pytest -q tests/test_paper_mode_no_live_orders.py

# chaos scenarios, as CI runs them
pytest -q tests/legacy_tools/test_execution_chaos_scenarios.py
```

Run at most a couple of test files per invocation; the suite is large and some
environments need `--basetemp` pointed at a writable scratch directory.

## 11. What reproducibility does not buy

Stated plainly, because the section above is the kind that gets over-read:

- A reproducible result can be reproducibly wrong. Determinism protects the *chain* from
  the analyst, not the *conclusion* from the market.
- A frozen contract prevents post-hoc editing. It does not make an underpowered study
  powered.
- A hash proves an artifact did not change. It says nothing about whether the artifact
  was right when it was hashed.

These properties are necessary. None of them is sufficient, and Eclipse does not treat
any of them as evidence of an edge.
