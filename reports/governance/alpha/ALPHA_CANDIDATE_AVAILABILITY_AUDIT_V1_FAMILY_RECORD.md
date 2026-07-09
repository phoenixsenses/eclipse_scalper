# Alpha Candidate Availability Audit -- Family Record (auto-generated)

- family_id: `FAM_ETH_BUY_LIQ_CONTINUATION`
- family_disposition: `REJECT_ENTRY_PREDICATE_LOOKAHEAD`
- promotion_disposition: `REJECT_SPURIOUS_OR_DATA_PATH_SUSPECT`
- canonical_alpha_gate_eligible: `False`

| candidate_id | predicate_features | disposition | reason |
| --- | --- | --- | --- |
| `CAND_ETH_BUY_CONT_500K_DAYTREND_D0_TP40_SL50_BE20` | cluster_notional, day_trend_bps | `REJECT_ENTRY_PREDICATE_LOOKAHEAD` | rule 8: event_ts_ms == cluster_start_ts_ms but predicate uses completed-cluster aggregate(s) ['cluster_notional'], which finalize only at cluster_end_ts_ms |
| `CAND_ETH_BUY_CONT_1M_DAYTREND_D0_TP40_SL50_BE20` | cluster_notional, day_trend_bps | `REJECT_ENTRY_PREDICATE_LOOKAHEAD` | rule 8: event_ts_ms == cluster_start_ts_ms but predicate uses completed-cluster aggregate(s) ['cluster_notional'], which finalize only at cluster_end_ts_ms |
| `CAND_ETH_BUY_CONT_500K_GEOM_COUNT22_D0_TP40_SL50_BE20` | cluster_notional, cluster_liq_count | `REJECT_ENTRY_PREDICATE_LOOKAHEAD` | rule 8: event_ts_ms == cluster_start_ts_ms but predicate uses completed-cluster aggregate(s) ['cluster_liq_count', 'cluster_notional'], which finalize only at cluster_end_ts_ms |
| `CAND_ETH_BUY_CONT_500K_CASCADE_P15_109K_D0_TP40_SL50_BE20` | cluster_notional, prior15_buy_liq_notional | `REJECT_ENTRY_PREDICATE_LOOKAHEAD` | rule 8: event_ts_ms == cluster_start_ts_ms but predicate uses completed-cluster aggregate(s) ['cluster_notional'], which finalize only at cluster_end_ts_ms |
| `CAND_ETH_BUY_CONT_500K_DAYTREND_GEOM_CASCADE_D0_TP40_SL50_BE20` | cluster_notional, day_trend_bps, cluster_liq_count, prior15_buy_liq_notional | `REJECT_ENTRY_PREDICATE_LOOKAHEAD` | rule 8: event_ts_ms == cluster_start_ts_ms but predicate uses completed-cluster aggregate(s) ['cluster_liq_count', 'cluster_notional'], which finalize only at cluster_end_ts_ms |
