# AMI Phase Validation (real data)

> 2026-07-02 19:38 UTC  ami v0.1.0

- **5. governed knowledge objects** — 9 KO, 12 failures
- **1. observation health** — {"mark_prices": "HEALTHY", "liquidations": "HEALTHY", "book_ticker": "HEALTHY", "agg_trades": "HEALTHY", "open_interest": "HEALTHY", "spot_prices": "HEALTHY", "vol_state": "STALE"}
- **2. multi-TF states** — 11 states, conflict={'alignment_score': 0.6, 'conflict_score': 0.4, 'dominant': 'UP', 'by_tf': {'1m': 'DOWN', '5m': 'UP', '15m': 'DOWN', '1h': 'FLAT', '4h': 'UP', '1D': 'UP', '1W': 'FLAT'}}
- **2b. structure transitions** — bars=361 phases=[('RANGE', 102), ('EXPANSION', 81), ('MATURE_TREND', 44), ('BREAKDOWN', 35)]
- **3. direction probabilities** — {"LONG": 0.58, "SHORT": 0.17, "NO_TRADE": 0.25}
- **9. decision trace** — D:1783021127790 result=SHADOW_ONLY unc=MODERATE
- **10. permission boundary** — OPEN_LONG on holdout-KO -> SHADOW_ONLY
- **4. trade lifecycle** — trades=120 mfe50={'A_continue': 59, 'B_breakeven': 0, 'C_negative': 13, 'D_time_pos': 17}
- **6. research marketplace** — ['Q-MECHCOMP-FORWARD-001', 'Q-META-VALIDATION-001', 'Q-EXPIRATION-RISK-001', 'Q-SIZING-EVIDENCE-001', 'Q-MFE-GIVEBACK-001']
- **7. preregistration** — E-MECHCOMP-FWD-001 frozen=5bc8fd31f48c12b2
- **8. contradiction/gates** — premature promotion blocked: yes (ConstitutionViolation)

Artifacts: `data/ami/knowledge.sqlite`, `data/ami/research.sqlite`, `data/ami/decisions.jsonl`, `data/ami/last_bundle.json`

*Runner: `python -m ami.run_phase_checks`*