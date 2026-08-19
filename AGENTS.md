# Eclipse Scalper Research Instructions

For any Eclipse Scalper / E-DER research task, load context in this order:

1. `docs/research/ECLIPSE_RESEARCH_BIBLE.md`
2. `docs/research/README.md`
3. `docs/research/DECISION_LOG.md`
4. `docs/research/HYPOTHESIS_LEDGER.md`
5. the latest applicable audit or stage-gate report
6. the active measurement/research contract

Treat frozen rules and anti-data-mining constraints as binding. Never change the frozen E-DER population or rules, entry/exit timing, research labels, or threshold definitions without explicit authorization and a dated Decision Log entry. Do not execute beyond the current stage gate without authorization.

Always distinguish `VERIFIED FROM CODE/DATA`, `INFERRED`, and `UNKNOWN / NOT RECOVERABLE`; separately distinguish `SUPPORTED`, `NOT SUPPORTED`, and `NOT IDENTIFIABLE`. Documentation language must never imply stronger semantics than the data identifies. A robust anomaly does not identify a unique mechanism when competing mechanisms remain observationally equivalent.

Never select features, models, or thresholds using E-DER return unless an explicitly authorized future confirmatory protocol permits it. Do not silently rewrite historical conclusions. Preserve superseded decisions and mark them `SUPERSEDED`.

Be proactive: identify the next scientifically justified question, required data, falsification condition, measurement risk, and data-mining risk. After a materially verified finding, propose or make only authorized updates to the Decision Log, Hypothesis Ledger, and canonical Bible; preserve the prior Bible version first.
