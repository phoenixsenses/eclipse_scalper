-- S34 HOUR17 — Forward Ayrıştırma ve DIRECT SHORT Audit
-- Tarih      : 2026-07-17
-- Statü      : AUDIT_PARTIAL — ITEMS_5_10_NOT_EXECUTED
-- Rapor      : S34_HOUR17_DIRECT_SHORT_AUDIT_2026-07-17.md
--
-- *** BU DOSYA HİÇBİR VERİTABANINA UYGULANMADI. ***
-- Salt-okunur analizin çıktısıdır. S34_ALL.db'ye veya microstructure.db'ye
-- YAZILMAMIŞTIR. Uygulanacaksa operatör kararıyla ve ayrı bir migration olarak
-- yapılmalıdır (CLAUDE.md: canonical store mutasyonu = bağımsız inceleme gerektirir).
--
-- Kaynak: reports/shadow/s34_state_machine_shadow.jsonl
--         data/microstructure.db (mode=ro): mark_prices, book_ticker, liquidations
-- Seed  : random.seed(20260717)
-- FEE   : 5.0 bps/RT | spread @T0 0.0760 bps | spread @kontrol 0.0617 bps
-- exec_disadvantage = +0.0142 bps (alpha'lardan düşüldü)
-- Boyut-bağımlı market impact: NOT_MODELED

BEGIN TRANSACTION;

-- ============================================================
-- 1. RUN METADATA
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_audit_run (
    run_id            TEXT PRIMARY KEY,
    run_utc           TEXT NOT NULL,
    scope             TEXT NOT NULL,
    status            TEXT NOT NULL,
    primary_verdict   TEXT NOT NULL,
    retracted_verdict TEXT,
    seed              INTEGER NOT NULL,
    fee_bps_rt        REAL NOT NULL,
    spread_sig_bps    REAL NOT NULL,
    spread_ctl_bps    REAL NOT NULL,
    exec_disadv_bps   REAL NOT NULL,
    impact_modeled    INTEGER NOT NULL,
    db_max_utc        TEXT NOT NULL,
    raw_trades        INTEGER NOT NULL,
    clusters          INTEGER NOT NULL,
    notes             TEXT
);

INSERT INTO h17_audit_run VALUES (
    'H17-AUDIT-2026-07-17',
    '2026-07-17T00:00:00Z',
    'LONG_HOUR17_HOLD6H forward re-measurement + DIRECT SHORT (P2) audit',
    'AUDIT_PARTIAL — ITEMS_5_10_NOT_EXECUTED',
    'DIRECT_SHORT_SUPPORT_DEPENDS_ON_MICRO_EFFECT',
    'DIRECT_SHORT_ALPHA_SUPPORTED',
    20260717,
    5.0, 0.0760, 0.0617, 0.0142,
    0,
    '2026-07-17T13:26:00Z',
    11, 7,
    'Read-only. No repo/param/strategy mutation. Live has NO hour17 SHORT route; all SHORT risk rules are research assumptions.'
);

-- ============================================================
-- 2. SHADOW LEDGER INTEGRITY
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_ledger_integrity (
    metric TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
INSERT INTO h17_ledger_integrity VALUES
 ('pnl_block_n',                '11'),
 ('pnl_block_total_net_bps',    '2477.9'),
 ('pnl_block_wr',               '0.455'),
 ('ledger_close_distinct',      '11'),
 ('ledger_open_rows',           '22'),
 ('ledger_open_distinct_ids',   '11'),
 ('open_duplicate_factor',      '2.0'),
 ('duplicate_logging_class',    'same as SYSTEM_STATE 127 BUY_FADE duplicate-close'),
 ('live_executor_hour17_refs',  '0'),
 ('real_orders_placed',         '0'),
 ('shadow_mode',                'OBSERVE_ONLY_NO_ORDER'),
 ('runner_gap_start_utc',       '2026-07-13T20:50:24Z'),
 ('runner_gap_end_utc',         '2026-07-15T19:13:39Z'),
 ('runner_gap_hours',           '46.4'),
 ('markprice_gap_start_utc',    '2026-07-13T21:07:11Z'),
 ('markprice_gap_end_utc',      '2026-07-14T12:21:46Z'),
 ('markprice_gap_hours',        '15.2'),
 ('artifact_trades',            '3'),
 ('artifact_total_bps',         '2854.3'),
 ('honest_trades',              '8'),
 ('honest_total_bps',           '-376.4'),
 ('honest_wr',                  '0.25'),
 ('honest_avg_bps',             '-47.0'),
 ('close_reason_distribution',  'CLOSED_TIME_EXIT:11'),
 ('max_mae_bps',                '-200.9'),
 ('stop_hits',                  '0');

-- ============================================================
-- 3. CONTRACT FINDING — shadow/live stop divergence
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_contract_finding (
    finding_id  TEXT PRIMARY KEY,
    severity    TEXT NOT NULL,
    summary     TEXT NOT NULL,
    shadow_ref  TEXT NOT NULL,
    live_ref    TEXT NOT NULL,
    current_impact TEXT NOT NULL,
    future_risk TEXT NOT NULL
);
INSERT INTO h17_contract_finding VALUES (
 'H17-STOP-DIVERGENCE-V1',
 'CONTRACT',
 'Shadow route is stopless by construction; live route applies 300bps stop. Forward evidence is collected from a non-faithful mirror.',
 'tools/s34_realtime_shadow_runner.py:680 observer_note=hold_predictor_hour17_no_early_exit_no_stop',
 'tools/s34_state_machine_live_executor.py:70 HOUR17_STOP_BPS=300.0; :678 stop_bps_override',
 'NONE in this sample: 11/11 TIME_EXIT, deepest MAE -200.9 bps, 99.1 bps margin to stop',
 'First trade with MAE < -300 diverges: shadow holds, live closes. E-HOUR17-FWD-001 silently stops representing live behaviour. 11/11 TIME_EXIT masks this.'
);

-- ============================================================
-- 4. EVENT CLUSTERS
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_cluster (
    cluster_id  TEXT PRIMARY KEY,
    n_members   INTEGER NOT NULL,
    first_utc   TEXT NOT NULL,
    last_utc    TEXT NOT NULL,
    linkage     TEXT NOT NULL
);
INSERT INTO h17_cluster VALUES
 ('C1', 1, '2026-07-04T23:14:40Z', '2026-07-04T23:14:40Z', 'single-linkage <=2h'),
 ('C2', 1, '2026-07-10T18:54:18Z', '2026-07-10T18:54:18Z', 'single-linkage <=2h'),
 ('C3', 2, '2026-07-11T22:20:33Z', '2026-07-11T23:21:20Z', 'single-linkage <=2h'),
 ('C4', 1, '2026-07-12T22:00:15Z', '2026-07-12T22:00:15Z', 'single-linkage <=2h'),
 ('C5', 3, '2026-07-13T17:36:55Z', '2026-07-13T18:19:32Z', 'single-linkage <=2h'),
 ('C6', 1, '2026-07-15T23:42:01Z', '2026-07-15T23:42:01Z', 'single-linkage <=2h'),
 ('C7', 2, '2026-07-16T23:03:42Z', '2026-07-16T23:20:23Z', 'single-linkage <=2h');

-- ============================================================
-- 5. AUDIT 1 — SAMPLE COUNT RECONCILIATION  (BLOCKING, FIXED)
--    corrected eff_n definition:
--      S_h = { cluster i : has data at horizon h }
--      W_i = [ min_j(entry_ij), max_j(entry_ij) + h ]   for i in S_h
--      M_h = merge_overlapping({W_i})
--      effective_independent_n(h) = |M_h|      => |M_h| <= |S_h| = cluster_n  (guaranteed)
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_count_reconciliation (
    horizon       TEXT PRIMARY KEY,
    horizon_sec   INTEGER NOT NULL,
    raw_n         INTEGER NOT NULL,
    cluster_n     INTEGER NOT NULL,
    eff_n_old     INTEGER NOT NULL,
    eff_n_fixed   INTEGER NOT NULL,
    identity_ok   INTEGER NOT NULL,
    old_violation INTEGER NOT NULL
);
INSERT INTO h17_count_reconciliation VALUES
 ('1m',      60, 11, 7, 7, 7, 1, 0),
 ('2m',     120, 11, 7, 7, 7, 1, 0),
 ('5m',     300, 11, 7, 7, 7, 1, 0),
 ('10m',    600, 11, 7, 7, 7, 1, 0),
 ('15m',    900, 11, 7, 7, 7, 1, 0),
 ('20m',   1200, 11, 7, 7, 7, 1, 0),
 ('30m',   1800, 11, 7, 7, 7, 1, 0),
 ('45m',   2700, 11, 7, 7, 7, 1, 0),
 ('1h',    3600, 11, 7, 7, 7, 1, 0),
 ('1.5h',  5400, 11, 7, 7, 7, 1, 0),
 ('2h',    7200, 11, 7, 7, 7, 1, 0),
 ('3h',   10800, 10, 7, 7, 7, 1, 0),
 ('4h',   14400,  8, 6, 7, 6, 1, 1),
 ('5h',   18000,  8, 6, 7, 6, 1, 1),
 ('6h',   21600,  8, 6, 7, 6, 1, 1),
 ('7h',   25200,  8, 6, 7, 6, 1, 1),
 ('8h',   28800,  8, 6, 7, 6, 1, 1),
 ('10h',  36000,  8, 6, 7, 6, 1, 1),
 ('12h',  43200,  8, 6, 7, 6, 1, 1),
 ('18h',  64800,  7, 6, 7, 6, 1, 1),
 ('24h',  86400,  8, 5, 4, 5, 1, 0),
 ('36h', 129600,  8, 5, 3, 4, 1, 0),
 ('48h', 172800,  6, 4, 3, 3, 1, 0),
 ('72h', 259200,  8, 5, 2, 2, 1, 0),
 ('1w',  604800,  1, 1, 1, 1, 1, 0);
-- old violations: 8 horizons (4h..18h) had eff_n_old=7 > cluster_n=6
-- fixed violations: 0
-- IMPACT: old family filter 'cluster_n>=6 AND eff_n>=7' wrongly admitted 4h..18h
--         => max-T family composition was invalid => FWER recomputed

-- ============================================================
-- 6. AUDIT 2 — MICRO / MEDIUM FAMILY SPLIT (studentized max-T)
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_family_test (
    family          TEXT NOT NULL,
    horizon         TEXT NOT NULL,
    raw_n           INTEGER NOT NULL,
    cluster_n       INTEGER NOT NULL,
    eff_n           INTEGER NOT NULL,
    alpha_bps       REAL NOT NULL,
    studentized_T   REAL NOT NULL,
    cluster_perm_p  REAL NOT NULL,
    ci95_lo         REAL NOT NULL,
    ci95_hi         REAL NOT NULL,
    sign_consistency REAL NOT NULL,
    PRIMARY KEY (family, horizon)
);
INSERT INTO h17_family_test VALUES
 ('MICRO',  '1m',  11, 7, 7,  10.0, 6.67, 0.0000,    3.0,  18.0, 0.86),
 ('MICRO',  '2m',  11, 7, 7,  11.0, 5.32, 0.0000,    2.0,  20.0, 0.86),
 ('MICRO',  '5m',  11, 7, 7,  11.0, 2.54, 0.0039,    1.0,  22.0, 0.71),
 ('MEDIUM', '6h',   8, 6, 6,  41.0, 1.01, 0.1601,  -16.0,  95.0, 0.67),
 ('MEDIUM', '7h',   8, 6, 6,  70.0, 1.71, 0.0406,   -3.0, 142.0, 0.83),
 ('MEDIUM', '8h',   8, 6, 6,  77.0, 1.88, 0.0266,   -2.0, 149.0, 0.83),
 ('MEDIUM', '10h',  8, 6, 6, 107.0, 2.01, 0.0192,    7.0, 208.0, 0.83),
 ('MEDIUM', '12h',  8, 6, 6,  90.0, 1.39, 0.0780,  -18.0, 190.0, 0.83),
 ('MEDIUM', '18h',  7, 6, 6,   6.0, 0.07, 0.4740, -165.0, 207.0, 0.50);

CREATE TABLE IF NOT EXISTS h17_family_fwer (
    family        TEXT PRIMARY KEY,
    n_horizons    INTEGER NOT NULL,
    best_horizon  TEXT NOT NULL,
    best_T        REAL NOT NULL,
    best_alpha_bps REAL NOT NULL,
    null_median_T REAL NOT NULL,
    null_p95_T    REAL NOT NULL,
    null_max_T    REAL NOT NULL,
    fwer_p        REAL NOT NULL,
    significant   INTEGER NOT NULL,
    n_placebo     INTEGER NOT NULL
);
INSERT INTO h17_family_fwer VALUES
 ('MICRO',  3, '1m',  6.67,  10.0, 0.54, 1.94, 4.90, 0.0000, 1, 10000),
 ('MEDIUM', 6, '10h', 2.01, 107.0, 0.59, 2.03, 3.51, 0.0523, 0, 10000);
-- KEY ANSWER: MEDIUM family alone (micro fully removed) is NOT family-wise significant.
--             best T (10h, 2.01) sits BELOW the family max-T 95% threshold (2.03).

-- ============================================================
-- 7. AUDIT 3 — 10h CELL EXACT TESTS
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_10h_event_alpha (
    t0_utc     TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    alpha_bps  REAL,
    status     TEXT NOT NULL
);
INSERT INTO h17_10h_event_alpha VALUES
 ('2026-07-04T23:14:40Z', 'C1',   71.0, 'OBSERVED'),
 ('2026-07-10T18:54:18Z', 'C2',  -85.0, 'OBSERVED'),
 ('2026-07-11T22:20:33Z', 'C3',   47.0, 'OBSERVED'),
 ('2026-07-11T23:21:20Z', 'C3',   12.0, 'OBSERVED'),
 ('2026-07-12T22:00:15Z', 'C4',   85.0, 'OBSERVED'),
 ('2026-07-13T17:36:55Z', 'C5',   NULL, 'GAP_NA'),
 ('2026-07-13T17:57:36Z', 'C5',   NULL, 'GAP_NA'),
 ('2026-07-13T18:19:32Z', 'C5',   NULL, 'GAP_NA'),
 ('2026-07-15T23:42:01Z', 'C6',  274.0, 'OBSERVED'),
 ('2026-07-16T23:03:42Z', 'C7',  293.0, 'OBSERVED'),
 ('2026-07-16T23:20:23Z', 'C7',  249.0, 'OBSERVED');

CREATE TABLE IF NOT EXISTS h17_10h_cluster_alpha (
    cluster_id TEXT PRIMARY KEY,
    n_obs      INTEGER NOT NULL,
    n_total    INTEGER NOT NULL,
    alpha_bps  REAL,
    status     TEXT NOT NULL
);
INSERT INTO h17_10h_cluster_alpha VALUES
 ('C1', 1, 1,   71.0, 'OBSERVED'),
 ('C2', 1, 1,  -85.0, 'OBSERVED'),
 ('C3', 2, 2,   29.0, 'OBSERVED'),
 ('C4', 1, 1,   85.0, 'OBSERVED'),
 ('C5', 0, 3,   NULL, 'GAP_NA_ALL'),
 ('C6', 1, 1,  274.0, 'OBSERVED'),
 ('C7', 2, 2,  271.0, 'OBSERVED');

CREATE TABLE IF NOT EXISTS h17_10h_exact_test (
    test_name  TEXT PRIMARY KEY,
    statistic  TEXT,
    p_value    REAL,
    n_units    INTEGER NOT NULL,
    passes_05  INTEGER NOT NULL,
    note       TEXT
);
INSERT INTO h17_10h_exact_test VALUES
 ('exact_randomization_signflip', '2^6=64 enumeration', 0.0781, 6, 0, 'FULL enumeration, not sampled'),
 ('sign_test',                    '5/6 positive',       0.1094, 6, 0, 'binomial exact'),
 ('wilcoxon_signed_rank_exact',   'W=17',               0.1094, 6, 0, 'full enumeration'),
 ('bootstrap_ci95',               'mean=+107 med=+78',  NULL,   6, 1, 'CI [+7,+208] — ANTI-CONSERVATIVE at n=6; exact tests are authoritative');

CREATE TABLE IF NOT EXISTS h17_10h_loco (
    excluded_cluster TEXT PRIMARY KEY,
    n_remaining INTEGER NOT NULL,
    mean_bps    REAL NOT NULL,
    median_bps  REAL NOT NULL,
    exact_p     REAL NOT NULL,
    sign_p      REAL NOT NULL,
    ci95_lo     REAL NOT NULL,
    ci95_hi     REAL NOT NULL
);
INSERT INTO h17_10h_loco VALUES
 ('C2', 5, 114.0, 85.0, 0.1250, 0.1875,  -5.0, 235.0),
 ('C3', 5, 146.0, 85.0, 0.0312, 0.0312,  57.0, 235.0),
 ('C4', 5, 123.0, 85.0, 0.1250, 0.1875,  11.0, 235.0),
 ('C5', 5, 111.0, 71.0, 0.1250, 0.1875,  -8.0, 232.0),
 ('C6', 5,  74.0, 71.0, 0.1562, 0.1875, -20.0, 185.0),
 ('C7', 5,  74.0, 71.0, 0.1562, 0.1875, -20.0, 187.0);
-- only C3-removal reaches p<0.05; all other removals p>=0.125

-- ============================================================
-- 8. AUDIT 4 — C5 MISSING-CLUSTER SENSITIVITY BOUNDS
--    *** SENSITIVITY ONLY. NOT imputation. NOT observed. ***
--    'unchanged when C5 removed' MUST NOT be cited as robustness:
--    C5 was never observed at this horizon.
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_c5_status (
    metric TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
INSERT INTO h17_c5_status VALUES
 ('observed_cluster_n', '6'),
 ('missing_cluster_n',  '1'),
 ('censored_cluster_n', '0'),
 ('gap_NA_cluster_n',   '1'),
 ('c5_exit_1', '2026-07-14T03:36Z GAP_NA'),
 ('c5_exit_2', '2026-07-14T03:57Z GAP_NA'),
 ('c5_exit_3', '2026-07-14T04:19Z GAP_NA');

CREATE TABLE IF NOT EXISTS h17_c5_sensitivity (
    assumed_c5_alpha_bps INTEGER PRIMARY KEY,
    n_clusters      INTEGER NOT NULL,
    mean_bps        REAL NOT NULL,
    median_bps      REAL NOT NULL,
    sign_consistency REAL NOT NULL,
    exact_p         REAL NOT NULL,
    ci95_lo         REAL NOT NULL,
    ci95_hi         REAL NOT NULL
);
INSERT INTO h17_c5_sensitivity VALUES
 (-500, 7,  18.0, 71.0, 0.71, 0.4297, -177.0, 182.0),
 (-300, 7,  48.0, 71.0, 0.71, 0.2891, -101.0, 182.0),
 (-200, 7,  62.0, 71.0, 0.71, 0.1797,  -57.0, 183.0),
 (-100, 7,  77.0, 71.0, 0.71, 0.1406,  -22.0, 183.0),
 (0,    7,  92.0, 71.0, 0.71, 0.0781,    2.0, 183.0),
 (100,  7, 106.0, 85.0, 0.86, 0.0391,   21.0, 194.0),
 (200,  7, 120.0, 85.0, 0.86, 0.0391,   27.0, 210.0),
 (300,  7, 134.0, 85.0, 0.86, 0.0391,   34.0, 228.0),
 (500,  7, 162.0, 85.0, 0.86, 0.0391,   37.0, 307.0);
-- 10h result swings p in [0.0391, 0.4297] on a single UNOBSERVED cluster.

-- ============================================================
-- 9. LONG-SIDE FINDINGS (for reconciliation with prior sections)
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_long_horizon (
    horizon    TEXT PRIMARY KEY,
    n          INTEGER NOT NULL,
    wr         REAL,
    mean_bps   REAL,
    alpha_bps  REAL,
    alpha_ci_lo REAL,
    alpha_ci_hi REAL,
    note       TEXT
);
INSERT INTO h17_long_horizon VALUES
 ('1m',  11, 0.18,   -9.0,  -10.0,  -18.0,   -3.0, NULL),
 ('5m',  11, 0.36,   -6.0,  -11.0,  -22.0,   -1.0, NULL),
 ('1h',  11, 0.45,   -4.0,  -11.0,  -39.0,   18.0, NULL),
 ('4h',   8, 0.38,  -20.0,  -38.0,  -80.0,    6.0, NULL),
 ('6h',   8, 0.25,  -47.0,  -41.0,  -96.0,   15.0, 'LIVE RULE'),
 ('24h',  8, 0.62,  231.0,   64.0, -240.0,  391.0, 'median alpha -28; rally-driven'),
 ('48h',  6, 1.00,  610.0,  325.0,  -18.0,  731.0, 'n=4 clusters, 3 eff'),
 ('1w',   1, NULL,  182.0,   NULL,   NULL,   NULL, 'eff_n=1; blind control +486 BEATS signal');

CREATE TABLE IF NOT EXISTS h17_policy_identity (
    identity TEXT PRIMARY KEY,
    meaning  TEXT NOT NULL,
    max_abs_error REAL NOT NULL,
    space    TEXT NOT NULL
);
INSERT INTO h17_policy_identity VALUES
 ('P5 = P4 + P3',   'initial LONG leg contribution P5-P3 EQUALS P4 exactly', 1.14e-13, 'simple'),
 ('P1 + P2 = -2k',  'LONG+SHORT same window = two round-trip costs',         7.46e-14, 'simple'),
 ('l(a,c)=l(a,b)+l(b,c)', 'log-return additivity (removes 11.61bps simple-return error)', 1.34e-16, 'log');

-- ============================================================
-- 10. VERDICTS
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_verdict (
    domain   TEXT PRIMARY KEY,
    verdict  TEXT NOT NULL,
    evidence TEXT NOT NULL
);
INSERT INTO h17_verdict VALUES
 ('PRIMARY',
  'DIRECT_SHORT_SUPPORT_DEPENDS_ON_MICRO_EFFECT',
  'All statistical support carried by MICRO family (FWER p=0.0000, T=6.67 at 1m). MEDIUM alone p=0.0523. Micro is the most execution-fragile cell; size-dependent impact NOT_MODELED.'),
 ('micro_1m_5m',
  'SUPPORTED_STATISTICALLY_PENDING_EXECUTION_AUDIT',
  'FWER p=0.0000; LOCO 7/7 positive; but AUDIT 7 (book_ticker executable fill, $1k-$100k notional) NOT EXECUTED.'),
 ('medium_6h_18h',
  'NOT_FAMILY_WISE_SUPPORTED',
  'FWER p=0.0523; best T=2.01 below family 95% threshold 2.03.'),
 ('10h_fixed_candidate',
  'FAILS_EXACT_TEST',
  'exact randomization p=0.0781; sign p=0.1094; Wilcoxon p=0.1094; C5 sensitivity p in [0.0391,0.4297].'),
 ('execution_realism',
  'NOT_ASSESSED',
  'AUDIT 7/8 not executed; size-dependent market impact NOT_MODELED.'),
 ('direction_semantics',
  'NOT_AUDITED',
  'AUDIT 9 not executed; liquidation BUY/SELL semantics and historical/live direction mapping NOT verified.'),
 ('historical_artifact_readiness',
  'NOT_LOCATED',
  'AUDIT 10 not executed; ~4MB HOUR17/history artifact not searched.'),
 ('one_day',
  'PATH_UNSTABLE',
  'alpha -62, CI [-377,+242], cluster_p=0.7524, eff_n=4; sign flips when C5 removed.'),
 ('one_week',
  'INSUFFICIENT_INDEPENDENT_EVENTS',
  'complete_n=1, censored_n=10, eff_n=1; no CI computable; excluded from max-T family (inclusion would let one window dominate FWER, producing spurious p=0.0001).'),
 ('RETRACTED',
  'DIRECT_SHORT_ALPHA_SUPPORTED',
  'Retracted: (1) count reconciliation bug invalidated family composition; (2) family not decomposed - FWER was carried by micro; (3) 10h fails exact tests.');

-- ============================================================
-- 11. OPEN AUDIT ITEMS
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_open_audit (
    item_no INTEGER PRIMARY KEY,
    name    TEXT NOT NULL,
    status  TEXT NOT NULL,
    why_it_matters TEXT
);
INSERT INTO h17_open_audit VALUES
 (5,  'Time-of-day / session control (same UTC hour +-15/30/60m, weekday, funding-cycle, combined)',
      'NOT_EXECUTED',
      'CRITICAL: 10h/600min bump maps onto night->morning UTC session transition (entries 17:36-23:42 UTC, exits 03:36-09:42 UTC). Current control matches only hour>=17 — a 7-hour window that may NOT absorb the session effect. Until run, even MEDIUM p=0.0523 must be read as optimistic.'),
 (6,  'Same-day placebo (T0-6h..T0-1h, T0+1h..T0+6h, >=10k sets)', 'NOT_EXECUTED', 'Nearest-time null not yet tested.'),
 (7,  'Micro execution audit (book_ticker bid/ask, $1k/$5k/$10k/$25k/$50k/$100k)', 'NOT_EXECUTED', 'The ONLY surviving effect is micro; unaudited for executability.'),
 (8,  '10h execution audit (notional-dependent impact)', 'NOT_EXECUTED', NULL),
 (9,  'Direction semantics audit (liquidation side -> pressure -> position, 7 layers)', 'NOT_EXECUTED', 'Any sign inversion or historical/live mismatch would be BLOCKING.'),
 (10, 'Historical ~4MB HOUR17/history artifact discovery', 'NOT_EXECUTED', NULL);

-- ============================================================
-- 12. PERMANENT LIMITATIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_limitation (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL
);
INSERT INTO h17_limitation VALUES
 (1, '7 independent events (effective 6/5/4/3/2/1 at longer horizons)'),
 (2, '12-day single regime (ETH 1751->1826 uptrend); no downtrend regime observed'),
 (3, 'Single asset (ETHUSDT)'),
 (4, 'C5 (3 raw trades) structurally unobserved at 10h-region cells'),
 (5, 'SHORT 300bps stop is a research assumption; live has no hour17 SHORT route'),
 (6, 'Size-dependent market impact NOT_MODELED'),
 (7, 'Forward binding E-HOUR17-FWD-001 requires min_sample=20; true independent event count is 7');

-- ============================================================
-- 13. AUDIT 5 — TIME-OF-DAY / SESSION CONTROL
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_a5_event_time (
    t0_utc     TEXT PRIMARY KEY,
    weekday    TEXT NOT NULL,
    fc_pos     INTEGER NOT NULL,
    entry_session TEXT NOT NULL,
    exit_session_10h TEXT NOT NULL,
    btc4h_bps  REAL, eth4h_bps REAL
);
INSERT INTO h17_a5_event_time VALUES
 ('2026-07-04T23:14Z','Sat',7,'LATE','EU', -18,-62),
 ('2026-07-10T18:54Z','Fri',2,'US','ASIA', -37,-20),
 ('2026-07-11T22:20Z','Sat',6,'LATE','EU',  -6,-47),
 ('2026-07-11T23:21Z','Sat',7,'LATE','EU', -41,-112),
 ('2026-07-12T22:00Z','Sun',6,'LATE','EU', -40,-68),
 ('2026-07-13T17:36Z','Mon',1,'US','ASIA', -94,-35),
 ('2026-07-13T17:57Z','Mon',1,'US','ASIA', -47,-62),
 ('2026-07-13T18:19Z','Mon',2,'US','ASIA', -92,-108),
 ('2026-07-15T23:42Z','Wed',7,'LATE','EU', -36,-52),
 ('2026-07-16T23:03Z','Thu',7,'LATE','EU', -51,-46),
 ('2026-07-16T23:20Z','Thu',7,'LATE','EU', -56,-60);
-- ALL entries 17:36-23:42 UTC (US/LATE); ALL 10h exits 03:36-09:42 UTC (ASIA/EU)
-- => 10h policy systematically samples night->morning session transition

CREATE TABLE IF NOT EXISTS h17_a5_control_10h (
    family TEXT PRIMARY KEY,
    definition TEXT NOT NULL,
    pool_min INTEGER, pool_med INTEGER, pool_max INTEGER,
    alpha_bps REAL, exact_p REAL, sign_p REAL, wilcoxon_p REAL,
    studentized_T REAL, degenerate INTEGER, valid_pool INTEGER
);
INSERT INTO h17_a5_control_10h VALUES
 ('A','same UTC hour +-15m',                14, 40, 84,  99, 0.1250,0.3438,0.1562, 2.00,0,1),
 ('B','same UTC hour +-30m',                26, 73,156,  97, 0.1250,0.3438,0.1562, 1.95,0,1),
 ('C','same UTC hour +-60m',                50,145,300,  96, 0.1250,0.3438,0.1562, 1.87,0,1),
 ('D','same weekday + hour +-30m',           0, 13, 13, 158, 0.0312,0.0312,0.0312,12.93,1,0),
 ('E','same funding-cycle + hour +-30m',    14, 50,108, 101, 0.1250,0.3438,0.1562, 1.97,0,1),
 ('F','hour +-30m + weekday + BTC4h regime', 0,  0, 13, 204, 0.0625,0.0625,0.0625,30.06,1,0),
 ('G','hour +-30m + ETH4h regime',           7, 14, 93, 170, 0.0625,0.1094,0.0781, 3.81,1,0),
 ('H','same entry+exit session',            40,136,264,  96, 0.1250,0.3438,0.1562, 1.90,0,1);
-- valid-pool families (A/B/C/E/H): exact_p=0.1250, NOT significant
-- D/F/G significance is SPURIOUS (degenerate/tiny control pool inflates T; F has events with 0 controls)

CREATE TABLE IF NOT EXISTS h17_a5_medium_fwer (
    control_family TEXT PRIMARY KEY,
    best_horizon TEXT, best_T REAL, maxT_null_p95 REAL,
    fwer_p REAL, significant INTEGER, pool_valid INTEGER
);
INSERT INTO h17_a5_medium_fwer VALUES
 ('A', '7h', 2.07, 2.43, 0.1207, 0, 1),
 ('B', '7h', 2.04, 2.43, 0.1247, 0, 1),
 ('C', '7h', 1.99, 2.42, 0.1450, 0, 1),
 ('E', '7h', 2.11, 2.33, 0.0860, 0, 1),
 ('H', '7h', 1.98, 2.36, 0.1327, 0, 1),
 ('D', '10h',13.30,2.64, 0.0000, 1, 0),
 ('F', '7h',31.43, 2.14, 0.0000, 1, 0),
 ('G', '8h', 3.93, 2.24, 0.0000, 1, 0);
-- ALL valid-pool families FAIL FWER (p=0.086-0.145)

CREATE TABLE IF NOT EXISTS h17_a5_sameday_placebo (
    test TEXT PRIMARY KEY,
    real_alpha_bps REAL, null_median_bps REAL, null_p95_bps REAL,
    real_percentile REAL, p_value REAL, n_sets INTEGER
);
INSERT INTO h17_a5_sameday_placebo VALUES
 ('sameday_T0-6h..T0-1h_and_T0+1h..T0+6h', 97, 107, 155, 37.2, 0.6276, 10000),
 ('nearest_time_+-15m',                    97, 116, 128,  0.2, 0.9982, 10000),
 ('nearest_time_+-30m',                    97, 121, 142,  1.1, 0.9893, 10000),
 ('nearest_time_+-60m',                    97, 125, 155,  3.3, 0.9672, 10000);
-- Random same-day entry BEATS the HOUR17 signal time (real at 37th pct, p=0.63)

-- ============================================================
-- 14. AUDIT 7 — MICRO EXECUTION AUDIT (book_ticker executable)
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_a7_exec_net_1m (
    latency_ms INTEGER PRIMARY KEY,
    net_bps_all_notionals REAL,   -- notional-independent (top-of-book fills)
    feasible_pct REAL
);
INSERT INTO h17_a7_exec_net_1m VALUES
 (0,   -2.3, 80),(100,-2.3,82),(250,-2.1,80),(500,-2.0,74),
 (1000,-1.5,88),(2000,-1.0,71),(5000,-0.2,71);
-- mark-price 1m return ~+10 bps -> executable bid/ask + fee = NEGATIVE (-2.3)

CREATE TABLE IF NOT EXISTS h17_a7_exec_alpha (
    horizon TEXT, latency_ms INTEGER,
    alpha_bps REAL, exact_p REAL, sign_p REAL, sign_consistency REAL,
    PRIMARY KEY (horizon, latency_ms)
);
INSERT INTO h17_a7_exec_alpha VALUES
 ('1m',0,   3.4,0.1719,0.5000,0.57),('2m',0,   3.7,0.2891,0.5000,0.57),('5m',0,   3.1,0.3516,0.5000,0.57),
 ('1m',500, 3.5,0.1797,0.5000,0.57),('2m',500, 3.3,0.3047,0.5000,0.57),('5m',500, 3.0,0.3438,0.5000,0.57),
 ('1m',2000,4.9,0.1250,0.5000,0.57),('2m',2000,5.1,0.1641,0.2266,0.71),('5m',2000,5.1,0.2578,0.5000,0.57);
-- mark-price 1m alpha +10 (studentized T=6.67, p=0.0000) collapses to +3.4 exec (exact_p=0.17)

CREATE TABLE IF NOT EXISTS h17_a7_breakeven (
    adverse_bps_per_leg INTEGER PRIMARY KEY,
    net_alpha_bps REAL, sign TEXT
);
INSERT INTO h17_a7_breakeven VALUES
 (0, 3.4,'+'),(1, 1.4,'+'),(2,-0.6,'-'),(5,-6.6,'-'),(10,-16.6,'-'),(15,-26.6,'-'),(20,-36.6,'-');
-- break-even adverse impact = 1.72 bps/leg

CREATE TABLE IF NOT EXISTS h17_a7_depth_feasibility (
    notional_usd INTEGER PRIMARY KEY,
    feasible_events INTEGER, total_events INTEGER, depth_note TEXT
);
INSERT INTO h17_a7_depth_feasibility VALUES
 (1000, 11,11,'top-of-book sufficient'),
 (5000, 11,11,'top-of-book sufficient'),
 (10000, 9,11,'DEPTH_UNKNOWN (top-of-book only)'),
 (25000, 8,11,'DEPTH_UNKNOWN'),
 (50000, 8,11,'DEPTH_UNKNOWN'),
 (100000,6,11,'DEPTH_UNKNOWN');

-- ============================================================
-- 15. AUDIT 5/7 VERDICTS + COMBINED
-- ============================================================
CREATE TABLE IF NOT EXISTS h17_final_verdict (
    audit TEXT PRIMARY KEY,
    verdict TEXT NOT NULL,
    basis TEXT NOT NULL
);
INSERT INTO h17_final_verdict VALUES
 ('AUDIT_5_time_control',
  'MEDIUM_SHORT_FAILS_EXACT_TIME_CONTROL',
  'Valid-pool control families A/B/C/E/H all FAIL FWER (p=0.086-0.145). Same-day placebo: real T0 at 37.2 pct (p=0.63); nearest-time placebo beats real 97-99.8%. 10h effect is session/time-of-day, not HOUR17 signal. D/F/G significance spurious (degenerate pools).'),
 ('AUDIT_7_micro_execution',
  'MICRO_SHORT_FAILS_EXECUTION_MODEL',
  'Executable bid/ask 1m net = -2.3 bps (mark +10 was mid-to-mid artifact). Executable paired alpha +3.4 bps, exact_p=0.17, sign_consistency 57%. Break-even adverse impact 1.72 bps/leg. Depth UNKNOWN above $5k.'),
 ('COMBINED',
  'BOTH_FAIL',
  'Two independent falsification audits each dropped a remaining pillar: MEDIUM statistical (time confound), MICRO execution (spread+depth). Neither rescues the other. DIRECT_SHORT_SUPPORT_DEPENDS_ON_MICRO_EFFECT collapses because the micro effect vanishes under execution. No harvestable edge proven in any direction in this forward window.'),
 ('STILL_OPEN',
  'AUDIT_9_direction_semantics + AUDIT_10_historical_artifact NOT_EXECUTED',
  'Do not change BOTH_FAIL, but would additionally verify direction integrity of the historical 93-cycle evidence.');

COMMIT;

-- ============================================================
-- SUMMARY QUERIES
-- ============================================================
-- SELECT * FROM h17_final_verdict;
-- SELECT * FROM h17_a5_medium_fwer WHERE pool_valid=1;      -- all FAIL
-- SELECT * FROM h17_a5_sameday_placebo;
-- SELECT * FROM h17_a7_exec_alpha WHERE latency_ms=0;
-- SELECT * FROM h17_a7_breakeven;
-- SELECT * FROM h17_family_fwer;
-- SELECT * FROM h17_count_reconciliation WHERE old_violation=1;
-- SELECT * FROM h17_10h_exact_test;
-- SELECT * FROM h17_c5_sensitivity;
-- SELECT domain, verdict FROM h17_verdict;
-- SELECT * FROM h17_open_audit WHERE status='NOT_EXECUTED';
