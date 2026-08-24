# Eclipse — documentation

Start at the top and stop wherever you have what you came for.

---

## The four that matter

| | Document | What it answers |
|---|---|---|
| 1 | [`RESEARCH_METHOD.md`](RESEARCH_METHOD.md) | how Eclipse decides that something is true — the ladder, and the rules that are not negotiable |
| 2 | [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) | the six planes, and which of them are in this repository |
| 3 | [`../INVARIANTS.md`](../INVARIANTS.md) | the hard contracts: for each, how it breaks, how to detect it, how it is enforced |
| 4 | [`PUBLIC_REPOSITORY_PROVENANCE.md`](PUBLIC_REPOSITORY_PROVENANCE.md) | what this repository is, and what is deliberately absent from it |

## Going deeper

| Document | What it answers |
|---|---|
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | the determinism and freezing contracts — and what they do *not* buy |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | where things stand, what is being worked on, what is closed |
| [`HISTORICAL_RESEARCH_CONTEXT.md`](HISTORICAL_RESEARCH_CONTEXT.md) | the earlier generation of this work, and what survived it |

## Running and operating

| Document | What it answers |
|---|---|
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | subsystem map and data/control flow |
| [`../EXECUTION_CONTRACTS.md`](../EXECUTION_CONTRACTS.md) | execution-layer contracts |
| [`../OPS_RUNBOOK.md`](../OPS_RUNBOOK.md) | operating procedures |
| [`../ENV_REFERENCE.md`](../ENV_REFERENCE.md) | environment variables, under their real code names rather than their documented ones |
| [`../PAPER_TRADING_ARCHITECTURE.md`](../PAPER_TRADING_ARCHITECTURE.md) | how paper mode is kept from reaching an exchange |
| [`../MICROSTRUCTURE_DATA_CONTRACT.md`](../MICROSTRUCTURE_DATA_CONTRACT.md) | what the data promises |
| [`../DEBUG_OPERATIONS.md`](../DEBUG_OPERATIONS.md) | debugging surfaces |

## Contributing and reporting

| Document | What it answers |
|---|---|
| [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) | what a contribution has to satisfy here, which is more than usual |
| [`../../SECURITY.md`](../../SECURITY.md) | how to report a vulnerability, and what this repository is not |

---

## Two things worth knowing before you read further

**This repository is one half of Eclipse.** The engineering framework is here; the
research estate — measurement lanes, frozen rules, outcome ledgers, the governance
subsystem — is private. Where a document names a boundary, that boundary is real, and the
code on this side already handles it.
[`PUBLIC_REPOSITORY_PROVENANCE.md`](PUBLIC_REPOSITORY_PROVENANCE.md) is the full account.

**Nothing here is a performance claim.** No route in Eclipse is claimed to be validated,
no component is claimed to be running or healthy, and no document in this set contains a
threshold, a formula or a measured result. That is enforced mechanically rather than by
care — see [`../maintenance/`](../maintenance/) if you would rather check the enforcement
than take it on trust.
