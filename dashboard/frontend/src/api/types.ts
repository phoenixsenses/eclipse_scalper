export type AnyDict = Record<string, unknown>;

export interface Scoreboard {
  generated_ts?: string | null;
  window_sec?: number | null;
  paper_trading?: boolean;
  orders_total?: number;
  fills_total?: number;
  cancels_total?: number;
  replaces_total?: number;
  reconcile_mismatches_total?: number;
  kill_switch_trips_total?: number;
  circuit_breaker_trips_total?: number;
  blocked_total?: number;
  blocked_by_reason?: Record<string, unknown>;
  orders_by_status?: Record<string, unknown>;
  orders_by_symbol?: Record<string, unknown>;
  fills_by_symbol?: Record<string, unknown>;
  last_order_ts?: string | null;
  last_fill_ts?: string | null;
  last_reconcile_ts?: number | null;
}

export interface GateSymbol {
  symbol: string;
  rule_name?: string | null;
  hit_rate?: number | null;
  delta_vs_baseline?: number | null;
  thresholds?: Record<string, unknown>;
}

export interface GatesSummaryResponse {
  generated_utc?: string | null;
  lookback_min?: number | null;
  symbols?: GateSymbol[];
}

export interface MicroEdgeGatesResponse {
  generated_utc?: string | null;
  lookback_min?: number | null;
  symbols?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface RegimeEvent {
  ts?: string | null;
  symbol?: string | null;
  timeframe?: string | null;
  effective_regime?: string | null;
  raw_regime?: string | null;
  confidence?: number | null;
  prev_effective?: string | null;
  streak?: number | null;
}

export interface SignalEvent {
  ts_utc?: string | null;
  symbol?: string | null;
  regime?: string | null;
  group?: string | null;
  allow_entry?: boolean | null;
  direction?: string | null;
  confidence?: number | null;
  reason?: string | null;
  rule_id?: string | null;
}

export interface StabilityEvent {
  ts?: string | null;
  symbol?: string | null;
  signal_type?: string | null;
  allowed?: boolean | null;
  reason?: string | null;
  streak?: number | null;
  cooldown_sec?: number | null;
}

export interface DataQualityEvent {
  ts?: string | null;
  symbol?: string | null;
  timeframe?: string | null;
  severity?: string | null;
  issues?: string[];
}

export interface LogFile {
  name: string;
  path: string;
  size_bytes: number;
  mtime: number;
}

export interface LogTailResponse {
  file: string;
  lines: string[];
}

export interface ConfigEntry {
  key: string;
  value: string;
  sensitive?: boolean;
  source?: string;
}

export interface CollectorStatus {
  alive?: boolean;
  last_log_ts?: string | null;
  uptime_sec?: number | null;
  trades_per_sec_60s?: number | null;
  mark_per_sec_60s?: number | null;
  liquidations_per_sec_60s?: number | null;
}

export interface DatabaseStatus {
  path?: string | null;
  size_bytes?: number;
  last_write_ts?: string | null;
  growth_bytes_5min?: number;
}

export type FreshnessStatus = "LIVE" | "DEGRADED" | "STALE";

export interface DataFreshnessStatus {
  last_trade_ts?: string | null;
  seconds_since_last_trade?: number | null;
  status?: FreshnessStatus | string;
}

export interface RuntimeStatus {
  collector?: CollectorStatus;
  database?: DatabaseStatus;
  data_freshness?: DataFreshnessStatus;
  system?: Record<string, unknown>;
}

export interface LiveAlertConfig {
  trade_age_alert_sec?: number;
  fill_flatline_alert_min?: number;
}

export interface LiveAlertState {
  any_alert?: boolean;
  trade_age_alert?: boolean;
  fill_flatline_alert?: boolean;
  trade_age_sec?: number | null;
  fill_age_min?: number | null;
  config?: LiveAlertConfig;
}

export interface LivePnlStrip {
  today?: number;
  h24?: number;
  d7?: number;
  sample?: number;
}

export interface LiveFillQuality {
  avg_delay_ms?: number | null;
  avg_adverse_bps?: number | null;
  with_delay?: number;
  with_adverse?: number;
}

export interface LiveTailKpis {
  window_lines?: number;
  order_count?: number;
  fill_count?: number;
  blocked_count?: number;
  fill_per_order_pct?: number | null;
}

export interface LiveFillRow {
  ts?: string | null;
  symbol?: string | null;
  side?: string | null;
  price?: string | null;
  qty?: string | null;
  pnl?: string | null;
  ts_ms?: number | null;
  pnl_num?: number | null;
  delay_ms?: number | null;
  adverse_bps?: number | null;
}

export interface LiveReasonRow {
  reason: string;
  count: number;
}

export interface LiveTrendSeries {
  trades_per_sec?: number[];
  fills_tail?: number[];
}

export interface LiveMetricsResponse {
  ts_utc?: string;
  runtime?: RuntimeStatus;
  scoreboard?: Scoreboard;
  pnl_strip?: LivePnlStrip;
  fill_quality?: LiveFillQuality;
  tail_kpis?: LiveTailKpis;
  blocked_reasons?: LiveReasonRow[];
  last_fills?: LiveFillRow[];
  alerts?: LiveAlertState;
  trends?: LiveTrendSeries;
  paper_file?: string | null;
}

export interface LiveMonitorTestsStatusResponse {
  ts_utc?: string;
  state?: string;
  stage?: string;
  message?: string;
  strict_mode?: boolean;
  backend_ok?: boolean;
  frontend_typecheck_ok?: boolean;
  frontend_smoke_ok?: boolean;
  frontend_smoke_skipped?: boolean;
  pid?: number | null;
  run_command?: string;
  log_path?: string | null;
  status_path?: string | null;
  status_age_sec?: number | null;
  log_tail?: string[];
}

export interface LiveMonitorTestsRunResponse {
  ok?: boolean;
  started?: boolean;
  pid?: number | null;
  script?: string;
  runner_log?: string;
  command?: string;
}

export interface OverviewResponse {
  scoreboard?: Scoreboard;
  gates?: GatesSummaryResponse;
  recent_regimes?: RegimeEvent[];
  exit_quality?: Record<string, unknown>;
  preflight?: Record<string, unknown>;
  reliability?: Record<string, unknown>;
}

export type PassiveProfilesResponse = Record<string, unknown>;

export interface HealthResponse {
  status: string;
  api_version?: string;
}

export interface OpsHealthResponse {
  ts_utc?: string;
  status?: {
    data_integrity?: string;
    network?: string;
  };
  thresholds?: {
    disk_free_min_gb?: number;
    wal_warn_mb?: number;
    backup_warn_sec?: number;
    reconnect_warn_5m?: number;
    rate_warn_pct?: number;
  };
  data_integrity?: {
    db_path?: string;
    db_size_bytes?: number;
    wal_path?: string;
    wal_size_bytes?: number;
    wal_ratio?: number;
    disk_free_gb?: number | null;
    disk_total_gb?: number | null;
    backup_dir?: string;
    backup_count?: number;
    last_backup_ts?: string | null;
    backup_age_sec?: number | null;
    last_maintenance_ts?: string | null;
    maintenance_age_sec?: number | null;
  };
  network?: {
    collector_connected?: boolean | null;
    reconnects_last_5m?: number;
    errors_last_5m?: number;
    used_1m?: number | null;
    cap_1m?: number | null;
    usage_pct?: number | null;
    samples?: Array<{
      used_1m: number;
      cap_1m: number;
      usage_pct: number;
    }>;
  };
  history?: Array<{
    ts_utc?: string;
    data_integrity?: {
      status?: string;
      disk_free_gb?: number | null;
      wal_size_bytes?: number | null;
    };
    network?: {
      status?: string;
      reconnects_last_5m?: number | null;
      usage_pct?: number | null;
    };
  }>;
}

export interface DiagConnectivityResponse {
  ts_utc?: string;
  status?: string;
  logs_dir?: {
    path?: string;
    exists?: boolean;
    writable?: boolean;
  };
  items?: Record<string, {
    path?: string;
    exists?: boolean;
    is_file?: boolean;
    readable?: boolean;
    size_bytes?: number | null;
    age_sec?: number | null;
  }>;
  hints?: string[];
}

export interface SupervisorStatusResponse {
  ts_utc?: string;
  supervisor_log_path?: string;
  supervisor_running?: boolean;
  backend_runtime_present?: boolean;
  backend_pid?: number | null;
  backend_host?: string | null;
  backend_port?: number | null;
  backend_runtime_age_sec?: number | null;
  last_event?: string | null;
  last_event_age_sec?: number | null;
  restarts_last_1h?: number;
}

export interface ControlActionInfo {
  action: string;
  description: string;
  timeout_sec: number;
}

export interface ControlActionRequest {
  action: string;
}

export interface ControlActionResult {
  action: string;
  ok: boolean;
  exit_code: number;
  duration_sec: number;
  output: string;
  started_ts: number;
  ended_ts: number;
}

export interface ControlHistoryItem {
  ts: number;
  action: string;
  command: string[];
  ok?: boolean | null;
  exit_code?: number | null;
  duration_sec?: number | null;
  error?: string | null;
}

export interface RunbookRequest {
  actions?: string[];
}

export interface RunbookContextRequest {
  actions?: string[];
  file?: string;
  query?: string;
  level?: string;
}

export interface IncidentHint {
  type?: string | null;
  title?: string | null;
  detail?: string | null;
  file?: string | null;
  query?: string | null;
  level?: string | null;
  suggested_command?: string | null;
  confidence?: number | null;
}

export interface RunbookSessionSummary {
  session_id: string;
  started_ts?: number | null;
  ended_ts?: number | null;
  duration_sec?: number | null;
  ok: boolean;
  failed_action?: string | null;
  incident_type?: string | null;
  tag?: string | null;
  note_preview?: string | null;
}

export interface RunbookSessionDetail {
  session_id: string;
  started_ts: number;
  ended_ts: number;
  duration_sec: number;
  ok: boolean;
  failed_action?: string | null;
  steps: ControlActionResult[];
  incident_hint?: IncidentHint | null;
  context?: Record<string, unknown>;
  tag?: string | null;
  note?: string | null;
  updated_ts?: number | null;
  log_snippets?: Array<{
    file: string;
    line_no: number;
    match: string;
    context_before: string[];
    context_after: string[];
  }>;
}

export interface RunbookSessionPatchRequest {
  tag?: string;
  note?: string;
}

export interface SessionTimelineEvent {
  ts?: number | null;
  kind: string;
  title: string;
  detail?: string | null;
  status?: string | null;
  action?: string | null;
  line_no?: number | null;
}

export interface IncidentInboxItem {
  incident_id: string;
  session_id: string;
  ts?: number | null;
  type: string;
  title: string;
  level: string;
  file?: string | null;
  query?: string | null;
  status: string;
  snoozed_until?: number | null;
  muted: boolean;
  failed_action?: string | null;
  ack_ts?: number | null;
  resolved_ts?: number | null;
}

export interface IncidentActionRequest {
  action: string;
  incident_type?: string;
  snooze_minutes?: number;
}

export interface IncidentBulkActionRequest {
  action: string;
  incident_type?: string;
  status_scope?: string;
}

export interface IncidentBulkPreviewRequest {
  incident_type?: string;
  status_scope?: string;
}

export interface IncidentBulkPreviewResult {
  ok: boolean;
  eligible: number;
  incident_type?: string | null;
  status_scope?: string | null;
}

export interface IncidentAuditEvent {
  ts?: number | null;
  operator?: string | null;
  kind?: string | null;
  action?: unknown;
  incident_id?: string | null;
  incident_type?: string | null;
  updated?: number | null;
  status_scope?: string | null;
}

export interface SecurityAuditEvent {
  ts?: number | null;
  kind: string;
  path?: string | null;
  method?: string | null;
  operator?: string | null;
  role?: string | null;
  ip?: string | null;
  user_agent?: string | null;
  request_id?: string | null;
  detail?: string | null;
}

export interface AutoRunbookPolicy {
  enabled: boolean;
  min_level: string;
  cooldown_sec: number;
  last_run_ts_by_type?: Record<string, number>;
}

export interface AutoRunbookResult {
  ran: boolean;
  reason?: string | null;
  incident_id?: string | null;
  session_id?: string | null;
}

export interface MacroPresetConfig {
  preset: string;
  ackFiltered: boolean;
  autoRun: boolean;
  exportMd: boolean;
  refresh: boolean;
  owner?: string | null;
  updated_ts?: number | null;
}

export type LogsStreamLine = string;

export interface LogsStreamEvent {
  data: LogsStreamLine;
}

export interface LiqAlertState {
  available: boolean;
  stale?: boolean;
  age_sec?: number;
  symbol?: string;
  rule?: string;
  state: {
    level: "quiet" | "elevated" | "severe";
    reasons: string[];
    primary_side_bias: string;
    dominant_severity: string;
  };
  card: {
    headline?: string;
    operator_note?: string;
    recent_alert_count?: number;
    tagged_rate?: number;
    max_consecutive_tagged?: number;
    max_liq_rate_recent?: number;
    primary_side_bias?: string;
    dominant_severity?: string;
    latest_alert_ts_ms?: number;
  };
  summary_snapshot: {
    rows_total?: number;
    tagged_count?: number;
    tagged_rate?: number;
    recent_alert_count?: number;
    max_consecutive_tagged?: number;
    max_liq_rate_recent?: number;
    side_bias_counts?: Record<string, number>;
    severity_counts?: Record<string, number>;
  };
  alerts?: Array<{
    ts_ms?: number;
    tag?: string;
    side_bias?: string;
    severity?: string;
    liq_rate_per_sec?: number;
    liq_imbalance?: number;
    spread?: number;
    trade_intensity?: number;
    ret_1?: number;
  }>;
}

export type StressLevel = "quiet" | "elevated" | "severe";

export interface SpreadStressState {
  available: boolean;
  stale?: boolean;
  age_sec?: number;
  symbol?: string;
  state: {
    level: StressLevel;
    reasons: string[];
    freshness_status?: string;
  };
  card: {
    headline?: string;
    operator_note?: string;
    recent_alert_count?: number;
    high_count?: number;
    medium_count?: number;
    avg_spread_tagged?: number;
    avg_trade_intensity_tagged?: number;
    freshness_status?: string;
    age_sec?: number;
  };
  dashboard_summary?: string;
  recommended_action?: string;
  watchlist?: {
    summary?: Record<string, unknown>;
    top_summary?: {
      symbol?: string;
      state_level?: string;
      freshness_status?: string;
      recommended_action?: string;
      dashboard_summary?: string;
    };
    banner?: {
      headline?: string;
      recommended_action?: string;
    };
    rows?: Array<{
      symbol?: string;
      state_level?: string;
      freshness_status?: string;
      recommended_action?: string;
      recent_alert_count?: number;
      high_count?: number;
      medium_count?: number;
      avg_spread_tagged?: number;
      priority_score?: number;
      dashboard_summary?: string;
    }>;
  } | null;
}

export interface FillToxicityState {
  available: boolean;
  stale?: boolean;
  age_sec?: number;
  source?: string;
  rows?: number;
  top_side?: string;
  state: {
    level: StressLevel;
    reasons: string[];
  };
  card: {
    headline?: string;
    operator_note?: string;
    top_side?: string;
    rows?: number;
    toxicity_score?: number;
    adverse_bps_mean?: number;
    pnl_bps_mean?: number;
  };
  dashboard_summary?: string;
  recommended_action?: string;
}

export interface LatencyStressState {
  available: boolean;
  stale?: boolean;
  age_sec?: number;
  source?: string;
  state: {
    level: StressLevel;
    reasons: string[];
  };
  card: {
    headline?: string;
    operator_note?: string;
    rows?: number;
    fill_rate?: number;
    latency_fill_delay_sec_p50?: number;
    latency_fill_delay_sec_p95?: number;
    latency_impact_vs_net_corr?: number;
  };
  dashboard_summary?: string;
  recommended_action?: string;
}
