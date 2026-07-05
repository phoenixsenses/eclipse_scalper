# AMI Current Decisions Memory Capsule — 2026-07-05

This file preserves the current project decisions discussed during the AMI category-completeness and production-roadmap conversation.

## Research Priority

- Continue research integrity, source semantics, feature/event identity and mechanism discovery first.
- Do not switch immediately into full production rebuild.
- Corrected LONG baselines are complete and stable, but are not alpha or live rules.
- Funding design is frozen but blocked by sample size.
- Birth-truncated geometry is known-at-safe but blocked inferentially by liquidation source quality.

## Vitalik-Inspired Broad-Search Principle

- Treat “document” as any knowledge-bearing artifact.
- Search code, tests, config, migrations, Git history, issues/PRs, incident reports, runtime journals, dashboards, runbooks and official external evidence.
- Require question-specific category-completeness audits.
- Require AI search receipts for high-impact audits.
- Absence from one artifact category does not prove absence from the project.

## Production Architecture Decision

- Existing risk, recovery, OMS, monitoring and paper/shadow/live components already exist in partial form.
- Do not blindly preserve them.
- Do not blindly delete and rewrite them.
- When production work begins, first classify every component:
  KEEP / EXTEND / REFACTOR / REPLACE / DELETE / DUPLICATED / UNSAFE / MISSING.
- Prefer selective rebuild, shadow parity and staged cutover.
- Production and research remain separate permission domains.

## Multi-Lane / Threshold Decision

- Multiple threshold/lane signals remain useful for research.
- Signals in the same independent cycle are not independent evidence.
- Live execution must use explicit cycle-level order rights and deduplication.

## Session and Memory Rule

- Canonical repository files are authoritative.
- Chat memory is not an authoritative methodology store.
- Important definitions must be frozen in code, tests, ADRs and contracts.
- New Claude sessions must read SYSTEM_STATE.md, IMPLEMENTATION_PROGRESS_LEDGER.md and TEST_STATUS_LATEST.md.
