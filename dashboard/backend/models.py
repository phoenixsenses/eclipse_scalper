"""Pydantic response models for Eclipse Scalper Dashboard."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Scoreboard(BaseModel):
    generated_ts: Optional[str] = None
    window_sec: Optional[int] = None
    paper_trading: bool = False
    runtime_mode: Optional[str] = None
    paper_execution_mode: Optional[str] = None
    paper_execution_label: Optional[str] = None
    paper_fill_model: Optional[str] = None
    binance_testnet: Optional[bool] = None
    paper_allow_live_private_api: Optional[bool] = None
    orders_total: int = 0
    fills_total: int = 0
    cancels_total: int = 0
    replaces_total: int = 0
    reconcile_mismatches_total: int = 0
    kill_switch_trips_total: int = 0
    circuit_breaker_trips_total: int = 0
    blocked_total: int = 0
    blocked_by_reason: dict[str, Any] = Field(default_factory=dict)
    orders_by_status: dict[str, Any] = Field(default_factory=dict)
    orders_by_symbol: dict[str, Any] = Field(default_factory=dict)
    fills_by_symbol: dict[str, Any] = Field(default_factory=dict)
    last_order_ts: Optional[str] = None
    last_fill_ts: Optional[str] = None
    last_reconcile_ts: Optional[int] = None


class GateSymbol(BaseModel):
    symbol: str
    rule_name: Optional[str] = None
    hit_rate: Optional[float] = None
    delta_vs_baseline: Optional[float] = None
    thresholds: dict[str, Any] = Field(default_factory=dict)


class GatesSummaryResponse(BaseModel):
    generated_utc: Optional[str] = None
    lookback_min: Optional[int] = None
    symbols: list[GateSymbol] = Field(default_factory=list)


class MicroEdgeGatesResponse(BaseModel):
    generated_utc: Optional[str] = None
    lookback_min: Optional[int] = None
    symbols: dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "allow"}


class PassiveProfilesResponse(BaseModel):
    model_config = {"extra": "allow"}


class RegimeEvent(BaseModel):
    ts: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    effective_regime: Optional[str] = None
    raw_regime: Optional[str] = None
    confidence: Optional[float] = None
    prev_effective: Optional[str] = None
    streak: Optional[int] = None


class SignalEvent(BaseModel):
    ts_utc: Optional[str] = None
    symbol: Optional[str] = None
    regime: Optional[str] = None
    group: Optional[str] = None
    allow_entry: Optional[bool] = None
    direction: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    rule_id: Optional[str] = None


class StabilityEvent(BaseModel):
    ts: Optional[str] = None
    symbol: Optional[str] = None
    signal_type: Optional[str] = None
    allowed: Optional[bool] = None
    reason: Optional[str] = None
    streak: Optional[int] = None
    cooldown_sec: Optional[float] = None


class DataQualityEvent(BaseModel):
    ts: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    severity: Optional[str] = None
    issues: list[str] = Field(default_factory=list)


class LogFile(BaseModel):
    name: str
    path: str
    size_bytes: int
    mtime: float


class LogTailResponse(BaseModel):
    file: str
    lines: list[str] = Field(default_factory=list)


class ConfigEntry(BaseModel):
    key: str
    value: str
    sensitive: bool = False
    source: str = "env_file"


class CollectorStatus(BaseModel):
    alive: bool = False
    last_log_ts: Optional[str] = None
    uptime_sec: Optional[int] = None
    trades_per_sec_60s: Optional[float] = None
    mark_per_sec_60s: Optional[float] = None
    liquidations_per_sec_60s: Optional[float] = None


class DatabaseStatus(BaseModel):
    path: Optional[str] = None
    size_bytes: int = 0
    last_write_ts: Optional[str] = None
    growth_bytes_5min: int = 0


class DataFreshnessStatus(BaseModel):
    last_trade_ts: Optional[str] = None
    seconds_since_last_trade: Optional[float] = None
    status: str = "STALE"


class SystemStatus(BaseModel):
    server_time: Optional[str] = None
    python_version: Optional[str] = None
    platform: Optional[str] = None


class RuntimeResponse(BaseModel):
    collector: CollectorStatus = Field(default_factory=CollectorStatus)
    database: DatabaseStatus = Field(default_factory=DatabaseStatus)
    data_freshness: DataFreshnessStatus = Field(default_factory=DataFreshnessStatus)
    system: SystemStatus = Field(default_factory=SystemStatus)


class LiveAlertConfig(BaseModel):
    trade_age_alert_sec: int = 10
    fill_flatline_alert_min: int = 15


class LiveAlertState(BaseModel):
    any_alert: bool = False
    trade_age_alert: bool = False
    fill_flatline_alert: bool = False
    trade_age_sec: Optional[float] = None
    fill_age_min: Optional[float] = None
    config: LiveAlertConfig = Field(default_factory=LiveAlertConfig)


class LivePnlStrip(BaseModel):
    today: float = 0.0
    h24: float = 0.0
    d7: float = 0.0
    sample: int = 0


class LiveFillQuality(BaseModel):
    avg_delay_ms: Optional[float] = None
    avg_adverse_bps: Optional[float] = None
    with_delay: int = 0
    with_adverse: int = 0


class LiveTailKpis(BaseModel):
    window_lines: int = 0
    order_count: int = 0
    fill_count: int = 0
    blocked_count: int = 0
    fill_per_order_pct: Optional[float] = None


class LiveFillRow(BaseModel):
    ts: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    price: Optional[str] = None
    qty: Optional[str] = None
    pnl: Optional[str] = None
    ts_ms: Optional[int] = None
    pnl_num: Optional[float] = None
    delay_ms: Optional[float] = None
    adverse_bps: Optional[float] = None


class LiveReasonRow(BaseModel):
    reason: str
    count: int


class LiveTrendSeries(BaseModel):
    trades_per_sec: list[float] = Field(default_factory=list)
    fills_tail: list[float] = Field(default_factory=list)


class LiveMetricsResponse(BaseModel):
    ts_utc: str
    runtime: RuntimeResponse = Field(default_factory=RuntimeResponse)
    scoreboard: Scoreboard = Field(default_factory=Scoreboard)
    pnl_strip: LivePnlStrip = Field(default_factory=LivePnlStrip)
    fill_quality: LiveFillQuality = Field(default_factory=LiveFillQuality)
    tail_kpis: LiveTailKpis = Field(default_factory=LiveTailKpis)
    blocked_reasons: list[LiveReasonRow] = Field(default_factory=list)
    last_fills: list[LiveFillRow] = Field(default_factory=list)
    alerts: LiveAlertState = Field(default_factory=LiveAlertState)
    trends: LiveTrendSeries = Field(default_factory=LiveTrendSeries)
    paper_file: Optional[str] = None


class PaperRunProcessChain(BaseModel):
    launcher_present: bool = False
    watchdog_present: bool = False
    bootstrap_present: bool = False
    launcher_pid: Optional[int] = None
    watchdog_pids: list[int] = Field(default_factory=list)
    bootstrap_pids: list[int] = Field(default_factory=list)
    launcher_started_ts: Optional[str] = None
    summary: str = "process chain unknown"


class PaperRunSession(BaseModel):
    status: str = "unknown"
    started_ts: Optional[str] = None
    uptime_sec: Optional[float] = None
    active_symbols: list[str] = Field(default_factory=list)
    telemetry_age_sec: Optional[float] = None
    telemetry_present: bool = False


class PaperRunEntryState(BaseModel):
    allow_entries: Optional[bool] = None
    guard_mode: Optional[str] = None
    runtime_gate_degraded: Optional[bool] = None
    runtime_gate_reason: Optional[str] = None
    data_state: str = "unknown"
    risk_state: str = "unknown"
    regime_state: str = "unknown"


class PaperRunTradeState(BaseModel):
    trade_count: int = 0
    last_trade_ts: Optional[str] = None
    no_trades_yet: bool = True
    db_present: bool = False
    db_path: Optional[str] = None
    execution_note: Optional[str] = None


class PaperRunDiagnosis(BaseModel):
    code: str = "unknown"
    summary: str = "paper run status unknown"
    detail: str = "awaiting telemetry"
    severity: str = "warning"
    execution_context: Optional[str] = None


class PaperRunReasonBreakdown(BaseModel):
    signal_not_present: int = 0
    gate_blocked: int = 0
    data_degraded: int = 0
    risk_blocked: int = 0
    regime_blocked: int = 0
    unknown: int = 0


class PaperRunSymbolState(BaseModel):
    symbol: str
    last_blocker_reason: Optional[str] = None
    last_blocker_ts: Optional[str] = None
    last_signal_ts: Optional[str] = None
    last_belief_ts: Optional[str] = None
    recent_blocked_count: int = 0


class PaperRunStartupManifest(BaseModel):
    env_profile: Optional[str] = None
    paper_profile_active: Optional[bool] = None
    paper_execution_mode: Optional[str] = None
    paper_execution_label: Optional[str] = None
    paper_fill_model: Optional[str] = None
    binance_testnet: Optional[bool] = None
    paper_allow_live_private_api: Optional[bool] = None
    private_api_key_present: Optional[bool] = None
    private_api_secret_present: Optional[bool] = None
    dotenv_source: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict, alias="_meta")
    model_config = {"populate_by_name": True}


class PaperRunStatusResponse(BaseModel):
    ts_utc: str
    session: PaperRunSession = Field(default_factory=PaperRunSession)
    startup_manifest: PaperRunStartupManifest = Field(default_factory=PaperRunStartupManifest)
    process_chain: PaperRunProcessChain = Field(default_factory=PaperRunProcessChain)
    entry_state: PaperRunEntryState = Field(default_factory=PaperRunEntryState)
    trade_state: PaperRunTradeState = Field(default_factory=PaperRunTradeState)
    diagnosis: PaperRunDiagnosis = Field(default_factory=PaperRunDiagnosis)
    reason_breakdown: PaperRunReasonBreakdown = Field(default_factory=PaperRunReasonBreakdown)
    symbols: list[PaperRunSymbolState] = Field(default_factory=list)


class LiveMonitorTestsStatusResponse(BaseModel):
    ts_utc: str
    state: str = "unknown"
    stage: str = "unknown"
    message: str = ""
    strict_mode: bool = False
    backend_ok: bool = False
    frontend_typecheck_ok: bool = False
    frontend_smoke_ok: bool = False
    frontend_smoke_skipped: bool = False
    pid: Optional[int] = None
    run_command: str = "powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\run_live_monitor_tests.ps1"
    log_path: Optional[str] = None
    status_path: Optional[str] = None
    status_age_sec: Optional[float] = None
    log_tail: list[str] = Field(default_factory=list)


class MarketChartIndicatorSeries(BaseModel):
    name: str
    values: list[Optional[float]] = Field(default_factory=list)


class MarketChartCandle(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketChartResponse(BaseModel):
    source: str = "binance_spot"
    symbol: str
    interval: str
    limit: int
    generated_ts: str
    candles: list[MarketChartCandle] = Field(default_factory=list)
    overlays: list[MarketChartIndicatorSeries] = Field(default_factory=list)
    oscillator: Optional[MarketChartIndicatorSeries] = None
    pocket_markers: list[dict[str, Any]] = Field(default_factory=list)


class OverviewResponse(BaseModel):
    scoreboard: Scoreboard = Field(default_factory=Scoreboard)
    gates: GatesSummaryResponse = Field(default_factory=GatesSummaryResponse)
    recent_regimes: list[RegimeEvent] = Field(default_factory=list)
    exit_quality: dict[str, Any] = Field(default_factory=dict)
    preflight: dict[str, Any] = Field(default_factory=dict)
    reliability: dict[str, Any] = Field(default_factory=dict)
    health_overall: dict[str, Any] = Field(default_factory=dict)
    research_events: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    api_version: str = "1.0.0"


class OpsHealthResponse(BaseModel):
    model_config = {"extra": "allow"}


class HostHealthResponse(BaseModel):
    model_config = {"extra": "allow"}


class DiagConnectivityResponse(BaseModel):
    model_config = {"extra": "allow"}


class ControlActionInfo(BaseModel):
    action: str
    description: str
    timeout_sec: int


class ControlActionRequest(BaseModel):
    action: str


class ControlActionResult(BaseModel):
    action: str
    ok: bool
    exit_code: int
    duration_sec: float
    output: str = ""
    started_ts: float
    ended_ts: float


class ControlHistoryItem(BaseModel):
    ts: float
    action: str
    command: list[str] = Field(default_factory=list)
    ok: Optional[bool] = None
    exit_code: Optional[int] = None
    duration_sec: Optional[float] = None
    error: Optional[str] = None


class RunbookRequest(BaseModel):
    actions: list[str] = Field(default_factory=list)


class RunbookContextRequest(BaseModel):
    actions: list[str] = Field(default_factory=list)
    file: Optional[str] = None
    query: Optional[str] = None
    level: Optional[str] = None


class IncidentHint(BaseModel):
    type: Optional[str] = None
    title: Optional[str] = None
    detail: Optional[str] = None
    file: Optional[str] = None
    query: Optional[str] = None
    level: Optional[str] = None
    suggested_command: Optional[str] = None
    confidence: Optional[float] = None


class RunbookSessionSummary(BaseModel):
    session_id: str
    started_ts: Optional[float] = None
    ended_ts: Optional[float] = None
    duration_sec: Optional[float] = None
    ok: bool = False
    failed_action: Optional[str] = None
    incident_type: Optional[str] = None
    tag: Optional[str] = None
    note_preview: Optional[str] = None


class RunbookSessionDetail(BaseModel):
    session_id: str
    started_ts: float
    ended_ts: float
    duration_sec: float
    ok: bool
    failed_action: Optional[str] = None
    steps: list[ControlActionResult] = Field(default_factory=list)
    incident_hint: Optional[IncidentHint] = None
    log_snippets: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    tag: Optional[str] = None
    note: Optional[str] = None
    updated_ts: Optional[float] = None


class RunbookSessionPatchRequest(BaseModel):
    tag: Optional[str] = None
    note: Optional[str] = None


class SessionTimelineEvent(BaseModel):
    ts: Optional[float] = None
    kind: str
    title: str
    detail: Optional[str] = None
    status: Optional[str] = None
    action: Optional[str] = None
    line_no: Optional[int] = None


class IncidentInboxItem(BaseModel):
    incident_id: str
    session_id: str
    ts: Optional[float] = None
    type: str = "unknown"
    title: str = "Unknown incident"
    level: str = "WARNING"
    file: Optional[str] = None
    query: Optional[str] = None
    status: str = "new"  # new | ack | resolved
    snoozed_until: Optional[float] = None
    muted: bool = False
    failed_action: Optional[str] = None
    ack_ts: Optional[float] = None
    resolved_ts: Optional[float] = None


class IncidentActionRequest(BaseModel):
    action: str  # ack | resolve | snooze_type | mute_type | unmute_type
    incident_type: Optional[str] = None
    snooze_minutes: Optional[int] = 60


class IncidentBulkActionRequest(BaseModel):
    action: str  # ack | resolve
    incident_type: Optional[str] = None
    status_scope: Optional[str] = "active"  # active | all


class IncidentBulkPreviewRequest(BaseModel):
    incident_type: Optional[str] = None
    status_scope: Optional[str] = "active"


class IncidentBulkPreviewResult(BaseModel):
    ok: bool = True
    eligible: int = 0
    incident_type: Optional[str] = None
    status_scope: Optional[str] = "active"


class IncidentAuditEvent(BaseModel):
    ts: Optional[float] = None
    operator: Optional[str] = None
    role: Optional[str] = None
    ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    kind: Optional[str] = None
    action: Optional[Any] = None
    incident_id: Optional[str] = None
    incident_type: Optional[str] = None
    updated: Optional[int] = None
    status_scope: Optional[str] = None


class SecurityAuditEvent(BaseModel):
    ts: Optional[float] = None
    kind: str
    path: Optional[str] = None
    method: Optional[str] = None
    operator: Optional[str] = None
    role: Optional[str] = None
    ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    detail: Optional[str] = None


class AutoRunbookPolicy(BaseModel):
    enabled: bool = False
    min_level: str = "WARNING"
    cooldown_sec: int = 900
    last_run_ts_by_type: dict[str, float] = Field(default_factory=dict)


class AutoRunbookResult(BaseModel):
    ran: bool = False
    reason: Optional[str] = None
    incident_id: Optional[str] = None
    session_id: Optional[str] = None


class MacroPresetConfig(BaseModel):
    preset: str = "full"
    ackFiltered: bool = True
    autoRun: bool = True
    exportMd: bool = True
    refresh: bool = True
    owner: Optional[str] = None
    updated_ts: Optional[float] = None
