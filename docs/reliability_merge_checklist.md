# Reliability Merge Checklist

Use this checklist before merging execution-reliability changes into `main`.

## Required Status Checks

- `Chaos Required - ack-after-fill-recovery` is `pass`
- `Chaos Required - cancel-unknown-idempotent` is `pass`
- `Chaos Required - replace-race-single-exposure` is `pass`
- `Execution Invariants and Gate` is `pass`
- `PR Reliability Comment` is `pass` or `skipping` (if not pull-request context)

## Required Review Gate

- PR has at least one approving review from a reviewer with write access.
- Confirm in CLI:

```powershell
& "C:\Program Files\GitHub CLI\gh.exe" pr view 1 --repo phoenixsenses/eclipse_scalper --json reviewDecision,mergeStateStatus
```

Expected:
- `reviewDecision` is `APPROVED`
- `mergeStateStatus` is not blocked by required review

## Merge Command

```powershell
& "C:\Program Files\GitHub CLI\gh.exe" pr merge 1 --repo phoenixsenses/eclipse_scalper --squash --delete-branch
```

## Post-Merge Verification

- Pull latest `main`.
- Run invariant subset locally:

```powershell
pytest -q tools/test_state_machine_unit.py
pytest -q tools/test_belief_controller_unit.py
pytest -q tools/test_order_router_unit.py
pytest -q tools/test_order_router_replace_unit.py
pytest -q tools/test_order_router_classifier_unit.py
pytest -q tools/test_order_router_idempotency_unit.py
```

## Rollback (if urgent regression appears)

- Revert the merge commit on `main`:

```powershell
git revert <merge_commit_sha>
git push
```

- Keep branch protection on; do not bypass required checks for follow-up fixes.
