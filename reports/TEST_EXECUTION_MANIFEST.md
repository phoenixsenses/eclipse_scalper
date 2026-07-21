# ECLIPSE TEST EXECUTION MANIFEST

Generated 2026-07-17. Ordering = dependency depth, foundation first.
Guardrail: max 2 files per pytest call, --basetemp to scratchpad,
-p no:cacheprovider, no parallel processes.


## T0  Isolation guard        (real-DB write protection must hold first)
files=1  tests~5

  1. tests/test_test_isolation_safety.py

## T1  Contracts & schemas    (data shape everything else assumes)
files=6  tests~169

  2. tests/contracts/test_event_schema.py tests/test_ami_lifecycle_canonical_schema.py
  3. tests/test_ami_lifecycle_path_schema.py tests/test_ami_warehouse_schema.py
  4. tests/test_micro_edge_smoke_schema.py tests/test_report_schema_validator.py

## T2  Pure units             (no DB, no network)
files=334  tests~1927

  5. tests/test_activate_history_contains_probe_stats.py tests/test_activate_online_artifacts_integrated_validate.py
  6. tests/test_alpha_cooldown.py tests/test_alpha_cost_decomposition.py
  7. tests/test_alpha_coverage_report.py tests/test_alpha_discovery_tests.py
  8. tests/test_alpha_dsl_eval.py tests/test_alpha_ensemble_regime.py
  9. tests/test_alpha_eval_reality_cap.py tests/test_alpha_filter_sweep.py
 10. tests/test_alpha_gate.py tests/test_alpha_gate_both_regime.py
 11. tests/test_alpha_gate_last_good_fallback.py tests/test_alpha_gate_stability.py
 12. tests/test_alpha_overlap_dedup.py tests/test_alpha_selection_stability.py
 13. tests/test_alpha_selectivity_calibration.py tests/test_alpha_walkforward_determinism.py
 14. tests/test_analyze_cost_breakdown.py tests/test_analyze_fill_timing.py
 15. tests/test_analyze_micro_edge_debug.py tests/test_analyze_micro_edge_regimes.py
 16. tests/test_blocking_fixes.py tests/test_book_proxy_pressure_alerts.py
 17. tests/test_book_proxy_pressure_state.py tests/test_bootstrap_auth_sanitize.py
 18. tests/test_bootstrap_dotenv_autoload.py tests/test_bootstrap_exchange_init_retry.py
 19. tests/test_bootstrap_startup_manifest.py tests/test_bot_core.py
 20. tests/test_brain_persistence_deadlock.py tests/test_buy_fade_duplicate_close_idempotency.py
 21. tests/test_buyfade_mutations.py tests/test_buyfade_silexit_mutations.py
 22. tests/test_calibrate_capacity_thresholds.py tests/test_canonical_health_gate_integration.py
 23. tests/test_chart_generator.py tests/test_check_event_lanes.py
 24. tests/test_collection_health.py tests/test_collection_watchdog.py
 25. tests/test_collector_checkpoint_interval.py tests/test_collector_simulation.py
 26. tests/test_collector_supervisor_cleanup.py tests/test_compare_rank_runs.py
 27. tests/test_compare_runs.py tests/test_compare_scratch_live_vs_backtest.py
 28. tests/test_control_actions_incidents.py tests/test_costs_config.py
 29. tests/test_daemon_calibration_suspect_alert.py tests/test_daily_execution_calibration.py
 30. tests/test_daily_report.py tests/test_data_layer_pid_states.py
 31. tests/test_data_layer_probe_helpers.py tests/test_db_introspect.py
 32. tests/test_db_maintenance.py tests/test_ensemble_regime_experts.py
 33. tests/test_entry_decision_trace.py tests/test_entry_loop_adaptive_guard.py
 34. tests/test_entry_loop_alpha_gate_integration.py tests/test_entry_loop_full_micro.py
 35. tests/test_entry_loop_full_shutdown_logs_reason.py tests/test_entry_loop_gate_integration.py
 36. tests/test_entry_loop_pocket_scheduler_integration.py tests/test_entry_loop_regime_gate.py
 37. tests/test_entry_micro_fallback.py tests/test_env_profile.py
 38. tests/test_eval_execution_model_integration.py tests/test_eval_run_outputs.py
 39. tests/test_eval_run_with_fills.py tests/test_eval_run_with_micro_edge_strategy.py
 40. tests/test_evaluate_canary_expansion_gate.py tests/test_evaluate_event_conditioned_filter.py
 41. tests/test_evaluate_event_conditioned_forward.py tests/test_evaluate_event_conditioned_forward_grid.py
 42. tests/test_event_lane_consolidation.py tests/test_event_lane_gate.py
 43. tests/test_event_lane_overlap.py tests/test_event_lane_persistence_policy.py
 44. tests/test_event_lane_suppression_policy.py tests/test_event_merged_banner_policy.py
 45. tests/test_event_watchboard_effective.py tests/test_event_watchboard_snapshot_append.py
 46. tests/test_event_watchboard_trend.py tests/test_event_watchboard_trend_from_history.py
 47. tests/test_exchanges_binance.py tests/test_exec_cost_models.py
 48. tests/test_exec_hazard_model.py tests/test_exec_queue_sim.py
 49. tests/test_exec_sim_adverse_samples.py tests/test_exec_sim_determinism.py
 50. tests/test_exec_sim_oracle_reduces_skips.py tests/test_exec_sim_skipped.py
 51. tests/test_exec_sim_spread_model.py tests/test_execution_coupled_math.py
 52. tests/test_execution_diagnostics.py tests/test_execution_e2e_pipeline.py
 53. tests/test_execution_quality_audit.py tests/test_execution_runtime_helpers.py
 54. tests/test_exit_scratch_integration.py tests/test_export_s34_visualization_json_reader_migration_parity.py
 55. tests/test_feature_distribution_analysis.py tests/test_fee_model.py
 56. tests/test_fill_toxicity_state.py tests/test_forward_return_labels.py
 57. tests/test_freeze_runtime_profile.py tests/test_funding_rate_analysis.py
 58. tests/test_funding_rate_analysis_reader_migration_parity.py tests/test_generalization_score.py
 59. tests/test_generate_liq_reversal_candidates.py tests/test_generator_coverage_guarantee.py
 60. tests/test_grid_strategy_override.py tests/test_guardrails_directional_sanity.py
 61. tests/test_guardrails_probe_trigger_sanity.py tests/test_health_check.py
 62. tests/test_health_check_stale.py tests/test_impact_law.py
 63. tests/test_incident_bundle.py tests/test_ingestion_check.py
 64. tests/test_latency_profiler.py tests/test_latency_stress_state.py
 65. tests/test_limit_offset_mult.py tests/test_liquidation_alert_state.py
 66. tests/test_liquidation_rule_coverage.py tests/test_liquidation_silence_canary_monitor.py
 67. tests/test_liquidation_silence_policy.py tests/test_live_alerts_drift.py
 68. tests/test_live_daemon_incremental.py tests/test_live_daemon_runtime_hooks.py
 69. tests/test_live_fill_drift_root_cause.py tests/test_live_metrics_alerts.py
 70. tests/test_live_model_loading.py tests/test_live_registry_guardrails.py
 71. tests/test_main_paper_safety.py tests/test_micro_builder_from_minimal_db.py
 72. tests/test_micro_diag.py tests/test_micro_edge_alignment.py
 73. tests/test_micro_edge_backtest_metrics.py tests/test_micro_edge_backtest_signs.py
 74. tests/test_micro_edge_debug_cap_behavior.py tests/test_micro_edge_debug_split.py
 75. tests/test_micro_edge_feature_filter.py tests/test_micro_edge_features.py
 76. tests/test_micro_edge_fixture_pipeline.py tests/test_micro_edge_jsonl.py
 77. tests/test_micro_edge_min_rule_filter.py tests/test_micro_edge_pocket_strategy.py
 78. tests/test_micro_edge_signal_v2.py tests/test_micro_edge_smoke_reader_migration_parity.py
 79. tests/test_micro_features.py tests/test_micro_features_compute.py
 80. tests/test_micro_features_smoke.py tests/test_micro_provider_binding.py
 81. tests/test_micro_signal.py tests/test_micro_signal_integration.py
 82. tests/test_microprice_spread_ofi_math.py tests/test_microstructure_rest_fallback.py
 83. tests/test_microstructure_sample_fixture.py tests/test_multi_symbol_reports.py
 84. tests/test_native_ws_health_policy.py tests/test_notification_integration.py
 85. tests/test_notifications_extended.py tests/test_ops_smoke.py
 86. tests/test_optimize_fill_timeout.py tests/test_order_placement.py
 87. tests/test_order_router_idempotency.py tests/test_order_router_intent_lifecycle.py
 88. tests/test_paper_mode_no_live_orders.py tests/test_paper_trade_summary.py
 89. tests/test_paper_trader_gate.py tests/test_papertrade_unified_engine.py
 90. tests/test_parity_check.py tests/test_passive_adverse_mult.py
 91. tests/test_passive_realistic_sim.py tests/test_passive_scratch_rule.py
 92. tests/test_performance_monitor.py tests/test_physics_signal_math.py
 93. tests/test_pid_registry_identity.py tests/test_pocket_promotion_checklist.py
 94. tests/test_pocket_scheduler.py tests/test_post_rollout_audit.py
 95. tests/test_preflight_check.py tests/test_prepare_release_tag.py
 96. tests/test_price_oracle.py tests/test_probe_require_diary_flag.py
 97. tests/test_propagator_synthetic.py tests/test_prototype_ws_vs_db_latency.py
 98. tests/test_push_status.py tests/test_reader_mapping.py
 99. tests/test_reconcile_paper_vs_backtest.py tests/test_reconnection_audit.py
100. tests/test_regime.py tests/test_regime_alignment.py
101. tests/test_regime_determinism.py tests/test_regime_kernel_filtering.py
102. tests/test_regime_risk.py tests/test_regime_sizer.py
103. tests/test_regime_slice_edge.py tests/test_replay_parity_report.py
104. tests/test_replay_slice.py tests/test_replay_strategy_determinism.py
105. tests/test_report_check.py tests/test_research_ami_mfe50_experiment_reader_migration_parity.py
106. tests/test_research_eth_provision_realism_lookup_migration_parity.py tests/test_research_event_operator_brief.py
107. tests/test_research_fitness_report.py tests/test_research_funding_nonoverlap_lookup_migration_parity.py
108. tests/test_research_nonpredictive_carry_provision_lookup_migration_parity.py tests/test_research_s34_100k_notmon_check_reader_migration_parity.py
109. tests/test_research_s34_500k_daytrend_route_sweep_lookup_migration_parity.py tests/test_research_s34_500k_daytrend_route_sweep_reader_migration_parity.py
110. tests/test_research_s34_btc_microtrend_eth_quality_lookup_migration_parity.py tests/test_research_s34_btc_microtrend_sweep_lookup_migration_parity.py
111. tests/test_research_s34_buy_reversal_short_reader_migration_parity.py tests/test_research_s34_consensus_composite_lookup_migration_parity.py
112. tests/test_research_s34_counter_regime_realfill_lookup_migration_parity.py tests/test_research_s34_day_context_scan_lookup_migration_parity.py
113. tests/test_research_s34_early_confirmation_scan_lookup_migration_parity.py tests/test_research_s34_early_confirmation_scan_reader_migration_parity.py
114. tests/test_research_s34_eth_preliq_control_lookup_migration_parity.py tests/test_research_s34_eth_preliq_executable_lookup_migration_parity.py
115. tests/test_research_s34_exact_route_change_validation_reader_migration_parity.py tests/test_research_s34_hold_sweep_reader_migration_parity.py
116. tests/test_research_s34_micro_entry_scalp_reader_migration_parity.py tests/test_research_s34_orderflow_lead_reader_migration_parity.py
117. tests/test_research_s34_post_tp_continuation_reader_migration_parity.py tests/test_research_s34_prediction_image_lookup_migration_parity.py
118. tests/test_research_s34_preliq_detector_lookup_migration_parity.py tests/test_research_s34_real_fill_parity_lookup_migration_parity.py
119. tests/test_research_s34_sell_liq_bounce_reader_migration_parity.py tests/test_research_s34_sell_path_quality_reader_migration_parity.py
120. tests/test_research_s34_sell_regime_analysis_reader_migration_parity.py tests/test_research_s34_sell_reversal_filter_reader_migration_parity.py
121. tests/test_research_s34_sell_reversal_quality_reader_migration_parity.py tests/test_research_s34_session_analysis_reader_migration_parity.py
122. tests/test_research_s34_sol200k_sell_dayfilter_reader_migration_parity.py tests/test_research_s34_source_quality_reconciliation_import_safe.py
123. tests/test_research_s34_source_quality_reconciliation_reader_migration_parity.py tests/test_research_s34_symbol_compare_reader_migration_parity.py
124. tests/test_research_s34_trailing_oos_realfill_lookup_migration_parity.py tests/test_research_s34_trailing_oos_realfill_reader_migration_parity.py
125. tests/test_research_s34_v6_management_system_lookup_migration_parity.py tests/test_research_s34_wave_absorption_lookup_migration_parity.py
126. tests/test_return_shock_alerts.py tests/test_return_shock_state.py
127. tests/test_review_event_lane_gate_shadow.py tests/test_risk_attribution.py
128. tests/test_risk_exposure_cap.py tests/test_risk_kill_switch_cooldown.py
129. tests/test_risk_sizer_determinism.py tests/test_run_alpha_multi.py
130. tests/test_run_alpha_multi_with_reports.py tests/test_run_alpha_pipeline.py
131. tests/test_run_alpha_pipeline_emits_regime_experts_pointers.py tests/test_run_daily_research_pipeline.py
132. tests/test_run_execution_canary.py tests/test_run_full_sweep.py
133. tests/test_run_scratch_calibration.py tests/test_run_transfer_matrix.py
134. tests/test_s34_bd_first_buy50_observer.py tests/test_s34_cascade_navigation.py
135. tests/test_s34_feature_availability.py tests/test_s34_knowable_anchor_continuation.py
136. tests/test_s34_live_chart_host_health.py tests/test_s34_preregistration_tools.py
137. tests/test_s34_regime_filter_shadow_eval_lookup_migration_parity.py tests/test_s34_regime_filter_shadow_eval_mark_prices_reader_migration_parity.py
138. tests/test_s34_risk_alerter.py tests/test_s34_shadow_paper_min_gap_migration.py
139. tests/test_s34_shadow_paper_min_gap_parity.py tests/test_s34_v_engine_adaptive_offset.py
140. tests/test_s34_v_engine_btc_kill_failed_short.py tests/test_s34_v_engine_cancel_replace.py
141. tests/test_s34_v_engine_confirmation_layer.py tests/test_s34_v_engine_data_incomplete_audit.py
142. tests/test_s34_v_engine_execution_frontier.py tests/test_s34_v_engine_failed_rebound.py
143. tests/test_s34_v_engine_failure_anatomy.py tests/test_s34_v_engine_forward_management_monitor.py
144. tests/test_s34_v_engine_live_executor.py tests/test_s34_v_engine_management_environment.py
145. tests/test_s34_v_engine_multi_offset_shadow.py tests/test_s34_v_engine_protective_stop.py
146. tests/test_s34_v_engine_shadow_observer.py tests/test_s34_v_engine_state_machine_observer.py
147. tests/test_scratch.py tests/test_set_latest_run.py
148. tests/test_settings_entry_min_conf.py tests/test_shadow_lane_signal_emitter.py
149. tests/test_shutdown_bypass_tripwire.py tests/test_shutdown_event_set_trace_capture.py
150. tests/test_shutdown_request_records_reason.py tests/test_signal_fixture_regression.py
151. tests/test_signal_no_lookahead.py tests/test_smoke_all.py
152. tests/test_spread_stress_alerts.py tests/test_spread_stress_state.py
153. tests/test_state_reconstruct.py tests/test_strategy_signal_diag.py
154. tests/test_summarize_event_conditioned_forward_grid.py tests/test_summarize_event_signal_bridge.py
155. tests/test_summarize_liq_regime_tag_impact.py tests/test_summarize_liq_tag_signal_behavior.py
156. tests/test_summarize_rank_attribution.py tests/test_summarize_rank_event_filter.py
157. tests/test_summarize_rank_event_filter_set.py tests/test_sweep_eval_index.py
158. tests/test_sweep_eval_micro_edge_strategy.py tests/test_sweep_exec_models.py
159. tests/test_sweep_micro_edge_costs.py tests/test_sweep_micro_edge_gates.py
160. tests/test_symbol_canonicalization.py tests/test_telegram_bot_dispatch.py
161. tests/test_tooling_audit.py tests/test_toxicity_report.py
162. tests/test_trade_logger.py tests/test_transfer_by_aligned_regime.py
163. tests/test_transfer_tools.py tests/test_triage_capacity.py
164. tests/test_ts_detection.py tests/test_validate_artifacts.py
165. tests/test_validate_canonical.py tests/test_validate_data_research_fitness.py
166. tests/test_validate_micro_edge_forward.py tests/test_validate_microstructure_contract.py
167. tests/test_volatility_burst_alerts.py tests/test_volatility_burst_state.py
168. tests/test_volume_vacuum_alerts.py tests/test_volume_vacuum_state.py
169. tests/test_walkforward_eval.py tests/test_walkforward_regime_stability.py
170. tests/test_walkforward_sweep.py tests/test_walkforward_sweep_promote.py
171. tests/test_watch_regime_recovery.py tests/test_x_twitter.py

## T3  AMI core               (warehouse/lifecycle/identity/chart/geometry/cvd)
files=43  tests~535

172. tests/test_ami_absorption_impact_canonical_migration.py tests/test_ami_absorption_impact_preregistration_v1.py
173. tests/test_ami_chart_candle_gap_repair_rehearsal.py tests/test_ami_chart_candle_morphology.py
174. tests/test_ami_chart_level_registry.py tests/test_ami_chart_push_geometry.py
175. tests/test_ami_chart_swing_extractor.py tests/test_ami_cvd_canonical_migration.py
176. tests/test_ami_cvd_primary_long_preregistration_v1.py tests/test_ami_cvd_repair_rehearsal.py
177. tests/test_ami_cvd_source_quality_contract_v1.py tests/test_ami_cvd_windowed_taker_flow.py
178. tests/test_ami_effective_path_and_experiment_immutability_hardening.py tests/test_ami_epistemic_nullifier_enforcement_wiring.py
179. tests/test_ami_epistemic_nullifier_legacy_bypass_closure.py tests/test_ami_geometry_birth_truncated_cascade_geometry.py
180. tests/test_ami_host_health_evaluator.py tests/test_ami_identity_cooldown_sensitivity.py
181. tests/test_ami_identity_cycle_resolver.py tests/test_ami_identity_event_identity.py
182. tests/test_ami_identity_shadow_ledger_ingest.py tests/test_ami_identity_split_utils.py
183. tests/test_ami_knowledge_governance.py tests/test_ami_latent_mutations.py
184. tests/test_ami_lifecycle_canonical_backfill.py tests/test_ami_lifecycle_canonical_field_provenance.py
185. tests/test_ami_lifecycle_engine_characterization.py tests/test_ami_lifecycle_migration_rehearsal.py
186. tests/test_ami_lifecycle_path_candle_repair_correction.py tests/test_ami_lifecycle_path_field_provenance.py
187. tests/test_ami_lifecycle_path_metrics.py tests/test_ami_lifecycle_path_migration_rehearsal.py
188. tests/test_ami_lifecycle_provenance_rehearsal.py tests/test_ami_mutation_suite.py
189. tests/test_ami_regime_mutations.py tests/test_ami_risk_mutations.py
190. tests/test_ami_states_research.py tests/test_ami_timing_contract.py
191. tests/test_ami_warehouse_experiment_ledger.py tests/test_ami_warehouse_funding_oi_audit.py
192. tests/test_ami_warehouse_ingest.py tests/test_ami_warehouse_question_seed.py
193. tests/test_ami_warehouse_registry_seed.py

## T4  AMI research/gov       (research OS, epistemic gates, storage)
files=37  tests~670

194. tests/test_ami_governance_epistemic_gates.py tests/test_ami_governance_next_independent_research_family_selection_v2.py
195. tests/test_ami_governance_storage_disk_usage_discrepancy_audit_v1.py tests/test_ami_research_book_spread_dynamics_canonical_migration.py
196. tests/test_ami_research_book_spread_dynamics_long_preregistration_v1.py tests/test_ami_research_book_spread_dynamics_preregistration_v1.py
197. tests/test_ami_research_candidate_universe.py tests/test_ami_research_cascade_absorption_impact_001.py
198. tests/test_ami_research_cvd_windowed_flow_001.py tests/test_ami_research_feature_gateway.py
199. tests/test_ami_research_forward_pipeline_characterization.py tests/test_ami_research_w10a_multi_tf_structural_conflict.py
200. tests/test_ami_research_w1_cycle_integrity.py tests/test_ami_research_w3_entry_timing_reconciliation.py
201. tests/test_ami_research_w4_post_event_path_taxonomy.py tests/test_ami_research_w5a_morphology_swing_grammar.py
202. tests/test_ami_research_w6rs_confound_resolution.py tests/test_ami_research_w7a_state_structure_aging_market_clocks.py
203. tests/test_ami_research_w8_hold_baseline.py tests/test_ami_research_w8_hold_baseline_004.py
204. tests/test_ami_research_w8_long_nested_path_accumulation.py tests/test_ami_research_w8_long_nested_path_accumulation_002.py
205. tests/test_ami_research_w8_long_timing_structure.py tests/test_ami_research_w8_long_timing_structure_002.py
206. tests/test_ami_research_w8_short_expanded_baseline.py tests/test_ami_research_w8_short_expanded_baseline_003.py
207. tests/test_ami_research_w8_vol_normalized_baseline.py tests/test_ami_research_w8_vol_normalized_baseline_004.py
208. tests/test_ami_storage_archive_and_verifier.py tests/test_ami_storage_catalog_reader_restorer.py
209. tests/test_ami_storage_job_state_and_cli.py tests/test_ami_storage_multi_shard_reader_restorer.py
210. tests/test_ami_storage_partition_and_planner.py tests/test_ami_storage_policy_and_registry.py
211. tests/test_ami_storage_production.py tests/test_ami_storage_production_activation.py
212. tests/test_ami_storage_research_reader_lookup.py

## T5  Execution & runtime    (entry/exit/router/reconcile/risk)
files=63  tests~798

213. tests/execution/test_entry_loop_unit.py tests/execution/test_latency_effect_on_fill_rate.py
214. tests/execution/test_latency_model.py tests/execution/test_order_router_pure_unit.py
215. tests/execution/test_order_router_unit.py tests/execution/test_queue_calibration.py
216. tests/execution/test_queue_position_dynamics.py tests/execution/test_reconcile_drift_matrix.py
217. tests/legacy_tools/test_adaptive_guard_unit.py tests/legacy_tools/test_belief_controller_unit.py
218. tests/legacy_tools/test_belief_evidence_unit.py tests/legacy_tools/test_binance_env.py
219. tests/legacy_tools/test_ci_workflow_unit.py tests/legacy_tools/test_corr_group_exposure_scale_unit.py
220. tests/legacy_tools/test_data_quality_unit.py tests/legacy_tools/test_diagnostics_unit.py
221. tests/legacy_tools/test_entry_conf_scale_unit.py tests/legacy_tools/test_entry_symbol_sizing_unit.py
222. tests/legacy_tools/test_entry_unit.py tests/legacy_tools/test_error_codes_unit.py
223. tests/legacy_tools/test_execution_chaos_scenarios.py tests/legacy_tools/test_exit_atr_scale_unit.py
224. tests/legacy_tools/test_exit_symbol_overrides_unit.py tests/legacy_tools/test_exit_telemetry_helper_unit.py
225. tests/legacy_tools/test_exit_telemetry_unit.py tests/legacy_tools/test_exit_unit.py
226. tests/legacy_tools/test_intent_ledger_unit.py tests/legacy_tools/test_position_closed_unit.py
227. tests/legacy_tools/test_position_manager_unit.py tests/legacy_tools/test_rebuild_unit.py
228. tests/legacy_tools/test_reliability_gate_runtime_unit.py tests/legacy_tools/test_reliability_gate_unit.py
229. tests/legacy_tools/test_replace_manager_unit.py tests/legacy_tools/test_strategy_audit_report_unit.py
230. tests/legacy_tools/test_strategy_unit.py tests/legacy_tools/test_telemetry_alert_summary_unit.py
231. tests/legacy_tools/test_telemetry_belief_state_unit.py tests/legacy_tools/test_telemetry_codes_by_symbol_unit.py
232. tests/legacy_tools/test_telemetry_error_classes_unit.py tests/legacy_tools/test_telemetry_latency_summary_unit.py
233. tests/legacy_tools/test_telemetry_report_unit.py tests/legacy_tools/test_telemetry_roll_alerts_unit.py
234. tests/legacy_tools/test_telemetry_smoke_assert_unit.py tests/legacy_tools/test_telemetry_smoke_workflow_unit.py
235. tests/legacy_tools/test_telemetry_threshold_alerts_unit.py tests/legacy_tools/test_telemetry_workflow_flags_unit.py
236. tests/runtime/test_alert_escalation.py tests/runtime/test_alert_rules.py
237. tests/runtime/test_circuit_breaker_unit.py tests/runtime/test_config_hot_reload.py
238. tests/runtime/test_deep_audit_fixes.py tests/runtime/test_degraded_mode.py
239. tests/runtime/test_emergency_unit.py tests/runtime/test_event_journal_unit.py
240. tests/runtime/test_guardian_profiling.py tests/runtime/test_health_gate_unit.py
241. tests/runtime/test_idempotency.py tests/runtime/test_integration_startup.py
242. tests/runtime/test_kill_switch_unit.py tests/runtime/test_order_fsm.py
243. tests/runtime/test_reconcile_unit.py tests/runtime/test_runtime_safety.py
244. tests/runtime/test_shutdown_control_unit.py

## T6  Real-data heavy        (microstructure.db, mode=ro)
files=60  tests~686

245. tests/test_alpha_candidate_availability_audit.py tests/test_ami_absorption_cascade_impact_rehearsal.py
246. tests/test_ami_chart_candle_builder.py tests/test_ami_geometry_birth_truncated_geometry_canonical_migration.py
247. tests/test_ami_geometry_birth_truncated_geometry_rehearsal.py tests/test_ami_geometry_liquidation_source_quality_contract_v2.py
248. tests/test_ami_governance_storage_rotation_retention_disposable_dry_run_v1.py tests/test_ami_governance_storage_rotation_retention_readiness_v1.py
249. tests/test_ami_host_health_observation.py tests/test_ami_lifecycle_short_noisy_v1_rehearsal.py
250. tests/test_ami_research_book_spread_dynamics_rehearsal.py tests/test_ami_research_book_spread_dynamics_row_accounting_freeze.py
251. tests/test_ami_research_spot_perp_basis_readiness_audit.py tests/test_ami_research_spread_dynamics_readiness_audit.py
252. tests/test_ami_research_w6_compression_rs_session.py tests/test_ami_research_w6rs_confirmation.py
253. tests/test_ami_storage_acceptance.py tests/test_ami_storage_production_activation_sharded.py
254. tests/test_ami_storage_research_reader.py tests/test_ami_storage_research_reader_lookup_production_parity.py
255. tests/test_ami_storage_research_reader_production_parity.py tests/test_ami_storage_reverify_hardening.py
256. tests/test_ami_storage_sharded_archive.py tests/test_ami_storage_source_access.py
257. tests/test_backtest_scratch.py tests/test_book_proxy_pressure_watchlist.py
258. tests/test_daily_research_report.py tests/test_entry_loop_event_lane_gate_shadow.py
259. tests/test_fit_adverse_model.py tests/test_health_cycle_smoke.py
260. tests/test_health_writer_ownership.py tests/test_heartbeat_watchdog.py
261. tests/test_liquidation_regime_alerts.py tests/test_liquidation_regime_tagger.py
262. tests/test_liquidation_silence_detector.py tests/test_liquidation_silence_scheduler.py
263. tests/test_liquidation_watchlist.py tests/test_monitor_h120_fill_density.py
264. tests/test_notifications.py tests/test_rank_passive_pockets_forward.py
265. tests/test_refresh_dashboard_research_events.py tests/test_research_event_watchboard.py
266. tests/test_research_s34_cluster_geometry_features_lookup_migration_parity.py tests/test_research_s34_exit_giveback_sweep_reader_migration_parity.py
267. tests/test_return_shock_watchlist.py tests/test_run_liq_reversal_e2e.py
268. tests/test_run_rank_sweep.py tests/test_run_research_event_watchboard_cycle.py
269. tests/test_s34_mechanism_taxonomy_lookup_migration_parity.py tests/test_s34_shadow_paper_runner.py
270. tests/test_s34_v_engine_v02_shadow_mirror_runtime_hardening.py tests/test_spread_stress_watchlist.py
271. tests/test_start_data_layer_duplicate_guard.py tests/test_status_snapshot.py
272. tests/test_sweep_capacity_awareness.py tests/test_validate_env.py
273. tests/test_validate_pocket_forward_api.py tests/test_verify_data_layer_status.py
274. tests/test_volatility_burst_watchlist.py tests/test_volume_vacuum_watchlist.py

## T7  Integration & parity   (dashboard, backtest/paper/live, replay)
files=14  tests~98

275. tests/legacy_tools/test_exit_quality_dashboard_unit.py tests/legacy_tools/test_telemetry_dashboard_notify_unit.py
276. tests/legacy_tools/test_telemetry_dashboard_page_unit.py tests/parity/test_backtest_paper_parity.py
277. tests/parity/test_paper_live_contract_parity.py tests/replay/test_replay_determinism.py
278. tests/test_dashboard_aggregator.py tests/test_dashboard_debug_api.py
279. tests/test_dashboard_debug_sessions_api.py tests/test_dashboard_live_metrics_api.py
280. tests/test_dashboard_market_chart_api.py tests/test_dashboard_overview_api.py
281. tests/test_dashboard_research_events_api.py tests/test_dashboard_shadow_paper_activity.py

## T9  QUARANTINED            (do not run in suite - see notes)
files=3  tests~49

282. tests/legacy_tools/test_entry_loop_unit.py tests/legacy_tools/test_entry_qty_scale_unit.py
283. tests/legacy_tools/test_order_router_unit.py

TOTAL (runnable): files=558  tests~4888  pytest-calls=283