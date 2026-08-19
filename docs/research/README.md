# Eclipse Scalper / E-DER Research Governance

This directory is the durable research operating system for Eclipse Scalper / E-DER.

## Canonical loading order

1. Repository-root `AGENTS.md`
2. `ECLIPSE_RESEARCH_BIBLE.md`
3. `DECISION_LOG.md`
4. `HYPOTHESIS_LEDGER.md`
5. latest applicable audit
6. active measurement/research contract
7. experiment-specific preregistration and report

The Bible is the canonical research context. The Decision Log preserves why choices changed. The Hypothesis Ledger preserves scientific state. Audits establish factual data feasibility. Contracts freeze measurement and research choices before analysis. Experiment reports contain results.

No result silently redefines the Bible. A materially verified change must retain the old version under `history/`, increment the Bible version, add a dated changelog and Decision Log entry, and mark superseded statements rather than erasing their existence.

## Current authority and stage gate

- Factual data boundary: `audits/data_feasibility_v1/DATA_FEASIBILITY_AUDIT_V1.md`
- Active historical specification: `contracts/E_DER_MEASUREMENT_SENSITIVITY_CONTRACT_V1.md`
- Forward data specification: `contracts/E_DER_PROSPECTIVE_MICROSTRUCTURE_COLLECTOR_CONTRACT_V1.md`
- Current stage: independent review and freeze of Measurement Sensitivity Contract V1 and Prospective Collector Contract V1
- Dynamic exit: `STOP / NOT AUTHORIZED`

Checkpoint commit for Bible V0.1 and Data Feasibility Audit V1: `2198eaa3`.

## Directory roles

- `history/`: immutable prior canonical versions
- `audits/`: evidence-backed feasibility and stage-gate reports
- `contracts/`: frozen or proposed specifications written before analysis/implementation
- experiment artifacts remain in their experiment-specific repository locations
