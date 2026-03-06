# Tooling Phase Plan

## Phase 1
- Generate and maintain a tooling manifest.
- Mark tools as `core`, `support`, `legacy`, or `dev-only`.
- Move unreferenced debug helpers under `tools/dev/`.
Status: completed.

## Phase 2
- Expand `run_summary` to remaining report producers with JSON output.
- Add those outputs to `report_check` fixtures.
- Close `run_summary` gaps reported by `tools.tooling_audit`.
Status: completed.

## Phase 3
- Split `tools/` into clearer subdomains:
  - `validation`
  - `reports`
  - `runs`
  - `ops`
  - `dev`
- Keep compatibility wrappers only where external invocation is likely.
Status: partially completed.
Note: the repository now uses practical sub-grouping via `tools/dev`, audit classification, and legacy test relocation under `tests/legacy_tools/`. A full directory split is no longer blocking.

## Phase 4
- Remove or archive `legacy`/`dev-only` scripts with no references.
- Merge overlapping report or inspect utilities where output contracts are redundant.
Status: in progress.
Note: `tools/dev` candidates remain listed in `docs/TOOLING_AUDIT.md` for optional deletion/pruning.

## Phase 5
- Wire `report_check` into local release workflow and CI smoke.
- Keep docs aligned with the active tool surface and contracts.
Status: mostly completed.
Note: local smoke includes `report_check` and `tooling_audit`; CI wiring can be treated as a follow-up if stricter release gating is needed.
