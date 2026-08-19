# E-DER Decision Log

**Policy:** append-only. Never delete a failed or superseded decision. A later decision links to the older item with `supersedes` / `superseded-by`.

## Historical decisions preserved from Research Bible V0.1

| ID | Date | Status | Decision | Evidence | Consequence | Supersession |
|---|---|---|---|---|---|---|
| D1 | pre-2026-08-19 | ACTIVE | Renewed SELL liquidation is not automatically deterioration. | Frozen Channel A ordering failed. | No thesis-state or exit inference from renewed SELL alone. | — |
| D2 | pre-2026-08-19 | ACTIVE | Raw liquidation amount is not the primary mechanism variable. | Quantity alone cannot identify effectiveness or mechanism. | Separate pressure from response. | Refined by D18 |
| D3 | pre-2026-08-19 | ACTIVE | Impact compression is not proof of absorption. | Competing mechanisms are observationally equivalent. | Use mechanism-compatible, not causal, language. | — |
| D4 | pre-2026-08-19 | ACTIVE | Visible book state is not total executable/latent liquidity. | Hidden liquidity is not displayed. | Displayed LOB evidence cannot identify hidden liquidity. | — |
| D5 | pre-2026-08-19 | ACTIVE | Flow Surprise and Impact Surprise are separate innovations. | Different estimands and null models. | Never collapse them by default. | — |
| D6 | pre-2026-08-19 | ACTIVE | Impact Surprise is a conditional response anomaly, not automatically alpha or mechanism proof. | Residuals are model-dependent. | No trading or causal promotion without further evidence. | — |
| D7 | pre-2026-08-19 | ACTIVE | Measurement sensitivity remains separate from residual magnitude. | Reliability and magnitude answer different questions. | No residual × quality composite. | — |
| D8 | pre-2026-08-19 | ACTIVE | Modern rich data is a validation laboratory, not a historical correction oracle. | Historical transportability is unproven. | No automatic backward correction. | — |
| D9 | pre-2026-08-19 | ACTIVE | Event-count/run-length/Hawkes measures are secondary until event semantics are proven. | forceOrder is throttled/snapshot-like. | Avoid overinterpreting message process. | — |
| D10 | pre-2026-08-19 | SUPERSEDED | Fixed-clock signed-notional aggressive flow was the preferred Flow Surprise primitive. | Conceptual pre-audit design. | Historical use is no longer permitted for frozen 25. | Superseded by D15–D16 |
| D11 | pre-2026-08-19 | ACTIVE | Prefer a minimum adequate null to a complex predictive model. | Anti-overfit research policy. | Complexity requires scientific justification. | — |
| D12 | pre-2026-08-19 | ACTIVE | Historical 25-event mechanism analysis is post-hoc; forward sample is confirmatory. | Hypothesis followed observation. | Historical results remain exploratory. | — |
| D13 | pre-2026-08-19 | ACTIVE | Use NOT IDENTIFIABLE when data cannot separate mechanisms. | Identification discipline. | Do not convert missing evidence into a negative finding. | — |
| D14 | pre-2026-08-19 | COMPLETED | Data Feasibility Audit V1 is the next stage. | Bible V0.1 stage gate. | Audit completed without modeling. | Superseded by D23 |

## Audit V1 decisions — 2026-08-19

| ID | Date | Status | Decision | Evidence | Consequence | Supersession |
|---|---|---|---|---|---|---|
| D15 | 2026-08-19 | ACTIVE | Historical E-DER mechanism work is limited to liquidation-pressure-proxy plus OHLCV response analysis. | Audit V1 §§1, 7, 11, 14–15. | Track A may test conditional OHLCV response anomalies only. | Supersedes historical part of D10 |
| D16 | 2026-08-19 | ACTIVE | Historical event-symbol aggressive-trade and LOB mechanisms are unsupported. | Retained aggTrade/bookTicker archives cover BTC/ETH/SOL, none of the 12 event symbols. | Historical aggressive-flow, quote-mid, OFI, MLOFI and displayed-book resilience are STOP. | — |
| D17 | 2026-08-19 | ACTIVE | Exact historical replenishment, hidden liquidity and executed-liquidation claims are not identifiable. | No sequence-valid L2; fill fields/raw payload absent; hidden orders unobserved. | Use `NOT IDENTIFIABLE`, not `NOT SUPPORTED`. | — |
| D18 | 2026-08-19 | ACTIVE | `q_parent` and `q_echo` are normalized observed forced-liquidation pressure proxies, not executed-liquidation shares. | Audit V1 §7.4 code provenance. | All future documentation uses the weaker semantic. | Refines D2 |
| D19 | 2026-08-19 | ACTIVE | forceOrder E-time and T-time are materially distinct clocks; clock assignment is a measurement-sensitivity dimension. | 1,722,630/1,722,645 keeper rows have E != T; delta 1..10,825 ms; no receive time. | E/T variants are predeclared and never selected by outcome. | — |
| D20 | 2026-08-19 | RESOLVED | Gross versus earlier net E-DER discrepancy is exactly 10 bps/event cost treatment. | Same 25 IDs and OPEN boundaries; every row differs by 10.0 bps. | Preserve both gross and hypothetical 10-bps-cost net contracts. | — |
| D21 | 2026-08-19 | ACTIVE | Existing `fl_*_ofi`, `bk_pull`, and `bk_refill` names receive no stronger semantics than provenance supports. | Audit V1 feature-provenance table. | Use semantic aliases in documentation; do not silently rename stored columns. | — |
| D22 | 2026-08-19 | ACTIVE | Historical and forward E-DER research use separate proxy and rich-microstructure tracks. | Audit V1 feasibility boundary. | Track A is historical proxy research; Track B is prospective confirmation. | — |
| D23 | 2026-08-19 | ACTIVE STAGE GATE | The next authorized object is Measurement Sensitivity Contract V1 plus Prospective Collector Contract V1, not alpha/model/exit research. | Audit V1 §17 and governance authorization. | Contracts require independent review/freeze before any historical proxy diagnostic; dynamic exit remains STOP. | Supersedes D14 |

## Repository checkpoints

| ID | Date | Status | Decision | Evidence | Consequence | Supersession |
|---|---|---|---|---|---|---|
| G1 | 2026-08-19 | COMPLETE | Preserve Bible V0.1 and Data Feasibility Audit V1 in Git. | Commit `2198eaa3`; archived Bible SHA-256 `6BA44FEB5942018AD413ECF22DC9C361E17AA459D3F84061A9106E73B475182D`. | Establishes immutable governance baseline. | — |
