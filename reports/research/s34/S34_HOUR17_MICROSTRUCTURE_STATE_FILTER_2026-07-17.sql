-- S34 HOUR17 — Microstructure State Filter (OOS Falsification)
-- Tarih : 2026-07-17 | Statü: COMPLETE — FROZEN OOS STUDY | seed=20260717
-- Rapor : S34_HOUR17_MICROSTRUCTURE_STATE_FILTER_2026-07-17.md
--
-- *** HİÇBİR VERİTABANINA UYGULANMADI. Salt-okunur analizin çıktısı. ***
-- microstructure.db mode=ro; yalniz event-cevresi indeksli pencereler.
-- June kaynak: S34_SELL_LIQ_REVERSAL_LONG_2026-06-07_15.json (read-only event source)
-- Boyut-bagimli market impact: NOT_MODELED.

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS sf_run (
    run_id TEXT PRIMARY KEY, run_utc TEXT, status TEXT,
    primary_verdict TEXT, econ_verdict TEXT, seed INTEGER,
    econ_thresh_bps REAL, primary_prob_thr REAL, notes TEXT
);
INSERT INTO sf_run VALUES (
 'H17-STATEFILTER-2026-07-17','2026-07-17T00:00:00Z','COMPLETE',
 'MICROSTRUCTURE_FEATURES_NOT_OOS_STABLE + INSUFFICIENT_INDEPENDENT_EVENTS',
 'FILTER_NO_BETTER_THAN_NO_TRADE', 20260717, 20.0, 0.70,
 'Read-only. June(67 canonical) leave-one-day-out train -> freeze -> July(11) external holdout apply-once. No threshold optimized on July. No feature deployed.'
);

-- 0. sample reconciliation (67 canonical resolved vs earlier-wrong 74)
CREATE TABLE IF NOT EXISTS sf_reconciliation (
    metric TEXT PRIMARY KEY, value INTEGER, definition TEXT
);
INSERT INTO sf_reconciliation VALUES
 ('raw_config_trade_rows', 4144, '64 configs x trade rows'),
 ('unique_signal_ts',       141, 'distinct signal_ts_ms'),
 ('unique_cascade_60s',      67, 'CANONICAL: chain/transitive 60s merge of signal_ts'),
 ('cluster_2h',              20, 'cascade-starts merged <=2h'),
 ('nonoverlap_6h',            6, 'non-overlapping 6h windows'),
 ('nonoverlap_10h',           4, 'non-overlapping 10h windows = day-level'),
 ('distinct_days',            4, '06-07,06-11,06-14,06-15'),
 ('earlier_wrong_dedup',     74, 'representative last-kept dedup OVER-counted 60-120s chains');
-- identity: 4 <= 20 <= 67 <= 141 <= 4144  (eff10h<=cluster<=cascade<=signal<=raw)

CREATE TABLE IF NOT EXISTS sf_cohort (
    cohort TEXT PRIMARY KEY, window TEXT, canonical_events INTEGER, days INTEGER,
    binary_n INTEGER, rev_n INTEGER, cont_n INTEGER, uncertain_n INTEGER
);
INSERT INTO sf_cohort VALUES
 ('A_JUNE','2026-06-07..06-15', 67, 4, 31, 15, 16, 26),
 ('B_JULY','2026-07-04..07-17', 11, 7,  6,  1,  5,  5);

-- 2. frozen labels (20bps primary)
CREATE TABLE IF NOT EXISTS sf_label_dist (
    cohort TEXT, label TEXT, n INTEGER, PRIMARY KEY (cohort,label)
);
INSERT INTO sf_label_dist VALUES
 ('JUNE','REVERSAL',24),('JUNE','CONTINUATION',17),('JUNE','UNCERTAIN_SMALL',11),('JUNE','UNCERTAIN_MISMATCH',15),
 ('JULY','REVERSAL',1),('JULY','CONTINUATION',5),('JULY','UNCERTAIN_SMALL',1),('JULY','UNCERTAIN_MISMATCH',1),('JULY','UNCERTAIN_GAP',3);

-- 3/4. feature leakage audit (all PASS, latest_source<=T0)
CREATE TABLE IF NOT EXISTS sf_feature_audit (
    feature TEXT PRIMARY KEY, sql_table TEXT, window TEXT, leakage TEXT,
    june_miss_pct REAL, july_miss_pct REAL
);
INSERT INTO sf_feature_audit VALUES
 ('absorption','mark_prices','T-5m..T0','PASS',0,0),
 ('accel','liquidations','T-10m..T0','PASS',0,0),
 ('sell_persistence','agg_trades','T-15m..T0','PASS',0,0),
 ('bid_refill','book_ticker','T-5m..T0','PASS',0,0),
 ('spread_z','book_ticker','T-30m..T0','PASS',28,0),
 ('imbalance','liquidations','T-60m..T0','PASS',0,0),
 ('vel_ratio_1_5','liquidations','T-5m..T0','PASS',0,0),
 ('rv_30m','mark_prices','T-30m..T0','PASS',0,0),
 ('dist_from_low_bps','mark_prices','T-60m..T0','PASS',0,0),
 ('eth_btc_rel_15m','mark_prices','T-15m..T0','PASS',0,0),
 ('btc_15m','mark_prices','T-15m..T0','PASS',0,0),
 ('sell_flow_share','agg_trades','T-15m..T0','PASS',0,0);
-- BLOCKED (latest_source>T0): 0

-- 7. June leave-one-day-out (M4 L2 logistic, thr=0.70)
CREATE TABLE IF NOT EXISTS sf_june_loocv (
    metric TEXT PRIMARY KEY, value REAL
);
INSERT INTO sf_june_loocv VALUES
 ('TP',2),('TN',11),('FP',5),('FN',13),
 ('continuation_recall',0.6875),('reversal_precision',0.2857),
 ('balanced_accuracy',0.41),('MCC',-0.21),('brier',0.358),
 ('always_long_net_bps',685),('filtered_net_bps',-454),('improvement_bps',-1139),
 ('trade_rate_num',7),('trade_rate_den',31);

-- 8. July external holdout (per-event)
CREATE TABLE IF NOT EXISTS sf_july_holdout (
    t0_utc TEXT PRIMARY KEY, p_rev REAL, pred TEXT, true_label TEXT,
    long6h_bps REAL, action TEXT
);
INSERT INTO sf_july_holdout VALUES
 ('2026-07-04T23:14Z',0.75,'REV','CONT', -60,'LONG'),
 ('2026-07-11T22:20Z',0.39,'CONT','CONT',-42,'VETO'),
 ('2026-07-12T22:00Z',0.67,'CONT','CONT',-151,'VETO'),
 ('2026-07-15T23:42Z',0.37,'CONT','REV',  42,'VETO'),   -- the ONLY reversal, WRONGLY vetoed
 ('2026-07-16T23:03Z',0.30,'CONT','CONT',-124,'VETO'),
 ('2026-07-16T23:20Z',0.49,'CONT','CONT', -80,'VETO');

-- 9. P0/P1/P2
CREATE TABLE IF NOT EXISTS sf_policy (
    policy TEXT PRIMARY KEY, n INTEGER, net_bps REAL, median_bps REAL
);
INSERT INTO sf_policy VALUES
 ('P0_NO_TRADE',0,0,0),
 ('P1_ALWAYS_LONG',8,-375,-51),
 ('P2_FILTERED_LONG',1,-60,-60);
-- improvement P2-P1=+315 but P2(-60) < P0(0): filter beats always-LONG, NOT no-trade

-- 10. feature stability June<->July
CREATE TABLE IF NOT EXISTS sf_feature_stability (
    feature TEXT PRIMARY KEY, june_dir INTEGER, july_dir INTEGER, verdict TEXT
);
INSERT INTO sf_feature_stability VALUES
 ('absorption', 1,-1,'SIGN_FLIP'),('accel',-1, 1,'SIGN_FLIP'),
 ('spread_z',   1,-1,'SIGN_FLIP'),('rv_30m', 1,-1,'SIGN_FLIP'),
 ('eth_btc_rel_15m',-1,1,'SIGN_FLIP'),
 ('sell_persistence',-1,-1,'SAME_DIRECTION'),('bid_refill',-1,-1,'SAME_DIRECTION'),
 ('imbalance',1,1,'SAME_DIRECTION'),('vel_ratio_1_5',-1,-1,'SAME_DIRECTION'),
 ('btc_15m',-1,-1,'SAME_DIRECTION'),('sell_flow_share',-1,-1,'SAME_DIRECTION'),
 ('dist_from_low_bps',1,0,'INSUFFICIENT');
-- 5 SIGN_FLIP / 6 SAME / 1 INSUFFICIENT; M3 core (absorption,accel,spread_z) all flip

-- 11. negative control
CREATE TABLE IF NOT EXISTS sf_permutation (
    test TEXT PRIMARY KEY, observed_bps REAL, p_value REAL, n_perms INTEGER
);
INSERT INTO sf_permutation VALUES
 ('june_loocv_improvement_day_perm', -1139, 0.7616, 10000);
-- observed improvement WORSE than 76% of random-label models

-- 13. shadow vs live stop
CREATE TABLE IF NOT EXISTS sf_stop_mirror (
    mirror TEXT PRIMARY KEY, note TEXT
);
INSERT INTO sf_stop_mirror VALUES
 ('shadow_faithful','no-stop; all reported numbers'),
 ('live_faithful','300bps stop; NO July LONG6h event exceeded -300 MAE (deepest -200.9) -> live=shadow this sample');

-- 15. verdicts
CREATE TABLE IF NOT EXISTS sf_verdict (
    domain TEXT PRIMARY KEY, verdict TEXT, basis TEXT
);
INSERT INTO sf_verdict VALUES
 ('PRIMARY','MICROSTRUCTURE_FEATURES_NOT_OOS_STABLE',
  '5/12 SIGN_FLIP incl M3 core; June-fit model vetoes July sole reversal; day-perm p=0.76. Also INSUFFICIENT_INDEPENDENT_EVENTS (June eff_n=4 days, July 1 reversal).'),
 ('june_internal','FAILS','MCC -0.21, balanced_acc 0.41, improvement -1139 bps, perm p=0.76'),
 ('july_holdout','FAILS','sole reversal vetoed; sole traded event lost; MODEL_VETO=9 LONG=1'),
 ('economic_value','FILTER_NO_BETTER_THAN_NO_TRADE','P2(-60) < P0(0); only reduces losses toward no-trade'),
 ('feature_stability','SIGN_FLIP_DOMINANT','5/12 flip including absorption/accel/spread_z'),
 ('shadow_live_consistency','CONSISTENT_THIS_SAMPLE','300bps stop never triggered; divergence risk still open for future -300 breach'),
 ('data_quality','ISOLATED','07-13 gap as DATA_QUALITY_VETO not model success; spread_z 28% June-missing NaN-guarded');

COMMIT;

-- SUMMARY:
-- SELECT * FROM sf_verdict;
-- SELECT * FROM sf_policy;              -- P2 < P0
-- SELECT * FROM sf_feature_stability WHERE verdict='SIGN_FLIP';
-- SELECT * FROM sf_permutation;         -- p=0.76
-- SELECT * FROM sf_june_loocv;          -- improvement -1139
