import type {
  ConfigEntry,
  ControlActionInfo,
  ControlActionResult,
  ControlHistoryItem,
  IncidentInboxItem,
  IncidentActionRequest,
  IncidentBulkActionRequest,
  IncidentBulkPreviewRequest,
  IncidentBulkPreviewResult,
  IncidentAuditEvent,
  SecurityAuditEvent,
  AutoRunbookPolicy,
  AutoRunbookResult,
  MacroPresetConfig,
  RunbookSessionDetail,
  RunbookSessionPatchRequest,
  RunbookSessionSummary,
  SessionTimelineEvent,
  DataQualityEvent,
  HealthResponse,
  OpsHealthResponse,
  DiagConnectivityResponse,
  SupervisorStatusResponse,
  LogFile,
  LogsStreamLine,
  LogTailResponse,
  MicroEdgeGatesResponse,
  OverviewResponse,
  PassiveProfilesResponse,
  RegimeEvent,
  RuntimeStatus,
  LiveMetricsResponse,
  PaperRunStatusResponse,
  MarketChartResponse,
  LiveMonitorTestsStatusResponse,
  LiveMonitorTestsRunResponse,
  Scoreboard,
  SignalEvent,
  StabilityEvent,
  LiqAlertState,
  SpreadStressState,
  FillToxicityState,
  LatencyStressState,
  WatchboardState,
  BookProxyPressureState,
  ReturnShockState,
  VolatilityBurstState,
  VolumeVacuumState,
  RiskOverview,
} from "./types";

const BASE = "/api";
const DEFAULT_TIMEOUT_MS = 10_000;
const LOG_LIST_TIMEOUT_MS = 30_000;
const LOG_TAIL_TIMEOUT_MS = 20_000;
const AUTH_CTX_KEY = "eclipse.dashboard.auth.ctx.v1";

export interface DashboardAuthContext {
  apiKey: string;
  operator: string;
  role: "viewer" | "operator" | "admin";
  idempotency: boolean;
}

const DEFAULT_AUTH_CTX: DashboardAuthContext = {
  apiKey: "",
  operator: "local",
  role: "admin",
  idempotency: true,
};

function genIdempotencyKey(scope: string): string {
  const base = typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  return `${scope}:${base}`;
}

export function getDashboardAuthContext(): DashboardAuthContext {
  try {
    const raw = window.localStorage.getItem(AUTH_CTX_KEY);
    if (!raw) return { ...DEFAULT_AUTH_CTX };
    const parsed = JSON.parse(raw) as Partial<DashboardAuthContext>;
    return {
      apiKey: typeof parsed.apiKey === "string" ? parsed.apiKey : DEFAULT_AUTH_CTX.apiKey,
      operator: typeof parsed.operator === "string" && parsed.operator.trim() ? parsed.operator : DEFAULT_AUTH_CTX.operator,
      role:
        parsed.role === "viewer" || parsed.role === "operator" || parsed.role === "admin"
          ? parsed.role
          : DEFAULT_AUTH_CTX.role,
      idempotency: typeof parsed.idempotency === "boolean" ? parsed.idempotency : DEFAULT_AUTH_CTX.idempotency,
    };
  } catch {
    return { ...DEFAULT_AUTH_CTX };
  }
}

export function setDashboardAuthContext(patch: Partial<DashboardAuthContext>): DashboardAuthContext {
  const next = { ...getDashboardAuthContext(), ...patch };
  try {
    window.localStorage.setItem(AUTH_CTX_KEY, JSON.stringify(next));
  } catch {
    // ignore storage failures
  }
  return next;
}

function buildControlHeaders(isWrite: boolean, scope: string): Record<string, string> {
  const ctx = getDashboardAuthContext();
  const headers: Record<string, string> = {};
  if (ctx.apiKey) headers["X-Api-Key"] = ctx.apiKey;
  if (ctx.operator) headers["X-Operator"] = ctx.operator;
  if (ctx.role) headers["X-Role"] = ctx.role;
  if (isWrite && ctx.idempotency) headers["X-Idempotency-Key"] = genIdempotencyKey(scope);
  return headers;
}

export class ApiError extends Error {
  public readonly status: number;
  public readonly details?: unknown;

  constructor(status: number, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

async function get<T>(
  path: string,
  params?: Record<string, string | number>,
  signal?: AbortSignal,
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<T> {
  const url = new URL(BASE + path, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) {
        url.searchParams.set(k, String(v));
      }
    }
  }

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort("timeout"), timeoutMs);
  const onAbort = () => controller.abort("upstream_abort");
  signal?.addEventListener("abort", onAbort, { once: true });

  let res: Response;
  try {
    res = await fetch(url.toString(), { signal: controller.signal });
  } catch (err) {
    if (controller.signal.aborted && controller.signal.reason === "timeout") {
      throw new ApiError(408, `Request timeout (${timeoutMs}ms) - ${path}`);
    }
    if (err instanceof TypeError) {
      throw new ApiError(503, `Network error - ${path}`);
    }
    throw err;
  } finally {
    window.clearTimeout(timeoutId);
    signal?.removeEventListener("abort", onAbort);
  }

  let payload: unknown = null;
  try {
    payload = await res.json();
  } catch {
    payload = null;
  }

  if (!res.ok) {
    throw new ApiError(res.status, `${res.status} ${res.statusText} - ${path}`, payload);
  }

  return payload as T;
}

async function getWithMeta<T>(
  path: string,
  params?: Record<string, string | number>,
  signal?: AbortSignal,
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<{ data: T; headers: Headers }> {
  const url = new URL(BASE + path, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) {
        url.searchParams.set(k, String(v));
      }
    }
  }

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort("timeout"), timeoutMs);
  const onAbort = () => controller.abort("upstream_abort");
  signal?.addEventListener("abort", onAbort, { once: true });

  let res: Response;
  try {
    res = await fetch(url.toString(), { signal: controller.signal });
  } catch (err) {
    if (controller.signal.aborted && controller.signal.reason === "timeout") {
      throw new ApiError(408, `Request timeout (${timeoutMs}ms) - ${path}`);
    }
    if (err instanceof TypeError) {
      throw new ApiError(503, `Network error - ${path}`);
    }
    throw err;
  } finally {
    window.clearTimeout(timeoutId);
    signal?.removeEventListener("abort", onAbort);
  }

  let payload: unknown = null;
  try {
    payload = await res.json();
  } catch {
    payload = null;
  }
  if (!res.ok) {
    throw new ApiError(res.status, `${res.status} ${res.statusText} - ${path}`, payload);
  }
  return { data: payload as T, headers: res.headers };
}

async function post<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal
): Promise<T> {
  const url = new URL(BASE + path, window.location.origin);
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort("timeout"), DEFAULT_TIMEOUT_MS);
  const onAbort = () => controller.abort("upstream_abort");
  signal?.addEventListener("abort", onAbort, { once: true });

  let res: Response;
  try {
    res = await fetch(url.toString(), {
      method: "POST",
      headers: { "Content-Type": "application/json", ...buildControlHeaders(true, path) },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    if (controller.signal.aborted && controller.signal.reason === "timeout") {
      throw new ApiError(408, `Request timeout (${DEFAULT_TIMEOUT_MS}ms) - ${path}`);
    }
    if (err instanceof TypeError) {
      throw new ApiError(503, `Network error - ${path}`);
    }
    throw err;
  } finally {
    window.clearTimeout(timeoutId);
    signal?.removeEventListener("abort", onAbort);
  }

  let payload: unknown = null;
  try {
    payload = await res.json();
  } catch {
    payload = null;
  }

  if (!res.ok) {
    throw new ApiError(res.status, `${res.status} ${res.statusText} - ${path}`, payload);
  }

  return payload as T;
}

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function asObject<T extends object>(value: unknown, fallback: T): T {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as T) : fallback;
}

function normalizeOverview(value: unknown): OverviewResponse {
  const obj = asObject<OverviewResponse>(value, {});
  return {
    ...obj,
    scoreboard: asObject<Scoreboard>(obj.scoreboard, {}),
    gates: {
      ...asObject(obj.gates, {}),
      symbols: asArray(obj.gates?.symbols),
    },
    recent_regimes: asArray<RegimeEvent>(obj.recent_regimes),
    preflight: asObject(obj.preflight, {}),
    reliability: asObject(obj.reliability, {}),
    health_overall: asObject(obj.health_overall, {}),
    exit_quality: asObject(obj.exit_quality, {}),
    research_events: asObject(obj.research_events, {}),
  };
}

function normalizeRuntime(value: unknown): RuntimeStatus {
  const obj = asObject<RuntimeStatus>(value, {});
  return {
    ...obj,
    collector: asObject(obj.collector, {}),
    database: asObject(obj.database, {}),
    data_freshness: asObject(obj.data_freshness, {}),
    system: asObject(obj.system, {}),
  };
}

function normalizeLogTail(value: unknown, file: string): LogTailResponse {
  const obj = asObject<LogTailResponse>(value, { file, lines: [] });
  return {
    file: typeof obj.file === "string" ? obj.file : file,
    lines: asArray<string>(obj.lines),
  };
}

export const api = {
  overview: async (signal?: AbortSignal): Promise<OverviewResponse> =>
    normalizeOverview(await get<unknown>("/overview", undefined, signal)),
  runtime: async (signal?: AbortSignal): Promise<RuntimeStatus> =>
    normalizeRuntime(await get<unknown>("/runtime", undefined, signal)),
  liveMetrics: (signal?: AbortSignal): Promise<LiveMetricsResponse> =>
    get<LiveMetricsResponse>("/live/metrics", undefined, signal),
  paperRunStatus: (signal?: AbortSignal): Promise<PaperRunStatusResponse> =>
    get<PaperRunStatusResponse>("/live/paper-run", undefined, signal),
  marketChart: (
    symbol = "BTCUSDT",
    interval = "5m",
    limit = 240,
    signal?: AbortSignal
  ): Promise<MarketChartResponse> =>
    get<MarketChartResponse>("/market/chart", { symbol, interval, limit }, signal),
  liveTestsStatus: (limit = 80, signal?: AbortSignal): Promise<LiveMonitorTestsStatusResponse> =>
    get<LiveMonitorTestsStatusResponse>("/live/tests/status", { limit }, signal),
  liveTestsRun: (signal?: AbortSignal): Promise<LiveMonitorTestsRunResponse> =>
    post<LiveMonitorTestsRunResponse>("/live/tests/run", {}, signal),
  scoreboard: (signal?: AbortSignal): Promise<Scoreboard> => get<Scoreboard>("/scoreboard", undefined, signal),
  gates: (signal?: AbortSignal): Promise<MicroEdgeGatesResponse> =>
    get<MicroEdgeGatesResponse>("/gates", undefined, signal),
  passiveProfiles: (signal?: AbortSignal): Promise<PassiveProfilesResponse> =>
    get<PassiveProfilesResponse>("/passive-profiles", undefined, signal),
  regimes: (limit = 100, symbol?: string, signal?: AbortSignal): Promise<RegimeEvent[]> =>
    get<RegimeEvent[]>("/events/regimes", symbol ? { limit, symbol } : { limit }, signal).then(asArray<RegimeEvent>),
  signals: (limit = 100, symbol?: string, signal?: AbortSignal): Promise<SignalEvent[]> =>
    get<SignalEvent[]>("/events/signals", symbol ? { limit, symbol } : { limit }, signal).then(asArray<SignalEvent>),
  stability: (limit = 100, symbol?: string, signal?: AbortSignal): Promise<StabilityEvent[]> =>
    get<StabilityEvent[]>("/events/stability", symbol ? { limit, symbol } : { limit }, signal).then(asArray<StabilityEvent>),
  quality: (limit = 100, symbol?: string, signal?: AbortSignal): Promise<DataQualityEvent[]> =>
    get<DataQualityEvent[]>("/events/quality", symbol ? { limit, symbol } : { limit }, signal).then(asArray<DataQualityEvent>),
  logFiles: (signal?: AbortSignal): Promise<LogFile[]> =>
    get<LogFile[]>("/logs", undefined, signal, LOG_LIST_TIMEOUT_MS).then(asArray<LogFile>),
  logFilesMeta: async (signal?: AbortSignal): Promise<{ files: LogFile[]; listMs?: number; cacheHit?: boolean }> => {
    const res = await getWithMeta<LogFile[]>("/logs", undefined, signal, LOG_LIST_TIMEOUT_MS);
    const listMs = Number(res.headers.get("X-Logs-List-Ms") || "");
    const cacheHit = (res.headers.get("X-Logs-Cache-Hit") || "") === "1";
    return {
      files: asArray<LogFile>(res.data),
      listMs: Number.isFinite(listMs) ? listMs : undefined,
      cacheHit,
    };
  },
  logTail: async (file: string, limit = 200, signal?: AbortSignal): Promise<LogTailResponse> =>
    normalizeLogTail(await get<unknown>("/logs/tail", { file, limit }, signal, LOG_TAIL_TIMEOUT_MS), file),
  logTailMeta: async (
    file: string,
    limit = 200,
    signal?: AbortSignal
  ): Promise<{ tail: LogTailResponse; tailMs?: number; source?: string }> => {
    const res = await getWithMeta<unknown>("/logs/tail", { file, limit }, signal, LOG_TAIL_TIMEOUT_MS);
    const tailMs = Number(res.headers.get("X-Logs-Tail-Ms") || "");
    return {
      tail: normalizeLogTail(res.data, file),
      tailMs: Number.isFinite(tailMs) ? tailMs : undefined,
      source: res.headers.get("X-Logs-Tail-Source") || undefined,
    };
  },
  config: (signal?: AbortSignal): Promise<ConfigEntry[]> =>
    get<ConfigEntry[]>("/config", undefined, signal).then(asArray<ConfigEntry>),
  health: (signal?: AbortSignal): Promise<HealthResponse> => get<HealthResponse>("/health", undefined, signal),
  opsHealth: (signal?: AbortSignal): Promise<OpsHealthResponse> => get<OpsHealthResponse>("/ops/health", undefined, signal),
  diagConnectivity: (signal?: AbortSignal): Promise<DiagConnectivityResponse> =>
    get<DiagConnectivityResponse>("/diag/connectivity", undefined, signal),
  liqAlertState: (signal?: AbortSignal): Promise<LiqAlertState> =>
    get<LiqAlertState>("/liq-alert-state", undefined, signal),
  spreadStressState: (signal?: AbortSignal): Promise<SpreadStressState> =>
    get<SpreadStressState>("/spread-stress-state", undefined, signal),
  fillToxicityState: (signal?: AbortSignal): Promise<FillToxicityState> =>
    get<FillToxicityState>("/fill-toxicity-state", undefined, signal),
  latencyStressState: (signal?: AbortSignal): Promise<LatencyStressState> =>
    get<LatencyStressState>("/latency-stress-state", undefined, signal),
  watchboardState: (signal?: AbortSignal): Promise<WatchboardState> =>
    get<WatchboardState>("/watchboard-state", undefined, signal),
  bookProxyPressureState: (signal?: AbortSignal): Promise<BookProxyPressureState> =>
    get<BookProxyPressureState>("/book-proxy-pressure-state", undefined, signal),
  returnShockState: (signal?: AbortSignal): Promise<ReturnShockState> =>
    get<ReturnShockState>("/return-shock-state", undefined, signal),
  volatilityBurstState: (signal?: AbortSignal): Promise<VolatilityBurstState> =>
    get<VolatilityBurstState>("/volatility-burst-state", undefined, signal),
  volumeVacuumState: (signal?: AbortSignal): Promise<VolumeVacuumState> =>
    get<VolumeVacuumState>("/volume-vacuum-state", undefined, signal),
  riskOverview: (signal?: AbortSignal): Promise<RiskOverview> =>
    get<RiskOverview>("/risk-overview", undefined, signal),
  supervisorStatus: (signal?: AbortSignal): Promise<SupervisorStatusResponse> =>
    get<SupervisorStatusResponse>("/ops/supervisor", undefined, signal),
  debugActions: (signal?: AbortSignal): Promise<ControlActionInfo[]> =>
    get<ControlActionInfo[]>("/debug/actions", undefined, signal).then(asArray<ControlActionInfo>),
  debugHistory: (limit = 30, signal?: AbortSignal): Promise<ControlHistoryItem[]> =>
    get<ControlHistoryItem[]>("/debug/history", { limit }, signal).then(asArray<ControlHistoryItem>),
  runDebugAction: (action: string, signal?: AbortSignal): Promise<ControlActionResult> =>
    post<ControlActionResult>("/debug/run", { action }, signal),
  runDebugRunbook: (actions?: string[], signal?: AbortSignal): Promise<RunbookSessionDetail> =>
    post<RunbookSessionDetail>("/debug/runbook", { actions: actions ?? [] }, signal),
  runDebugRunbookFromIncident: (
    file: string,
    query: string,
    level: string,
    actions?: string[],
    signal?: AbortSignal
  ): Promise<RunbookSessionDetail> =>
    post<RunbookSessionDetail>(
      "/debug/runbook/from-incident",
      { file, query, level, actions: actions ?? [] },
      signal
    ),
  debugSessions: (limit = 30, signal?: AbortSignal): Promise<RunbookSessionSummary[]> =>
    get<RunbookSessionSummary[]>("/debug/sessions", { limit }, signal).then(asArray<RunbookSessionSummary>),
  debugSessionDetail: (sessionId: string, signal?: AbortSignal): Promise<RunbookSessionDetail> =>
    get<RunbookSessionDetail>(`/debug/sessions/${encodeURIComponent(sessionId)}`, undefined, signal),
  patchDebugSession: (
    sessionId: string,
    patch: RunbookSessionPatchRequest,
    signal?: AbortSignal
  ): Promise<RunbookSessionDetail> =>
    fetch(new URL(BASE + `/debug/sessions/${encodeURIComponent(sessionId)}`, window.location.origin).toString(), {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        ...buildControlHeaders(true, `/debug/sessions/${encodeURIComponent(sessionId)}`),
      },
      body: JSON.stringify(patch),
      signal,
    }).then(async (res) => {
      let payload: unknown = null;
      try {
        payload = await res.json();
      } catch {
        payload = null;
      }
      if (!res.ok) {
        throw new ApiError(res.status, `${res.status} ${res.statusText} - /debug/sessions/{id}`, payload);
      }
      return payload as RunbookSessionDetail;
    }),
  debugSessionTimeline: (sessionId: string, signal?: AbortSignal): Promise<SessionTimelineEvent[]> =>
    get<SessionTimelineEvent[]>(`/debug/sessions/${encodeURIComponent(sessionId)}/timeline`, undefined, signal)
      .then(asArray<SessionTimelineEvent>),
  debugIncidents: (limit = 50, signal?: AbortSignal): Promise<IncidentInboxItem[]> =>
    get<IncidentInboxItem[]>("/debug/incidents", { limit }, signal).then(asArray<IncidentInboxItem>),
  patchDebugIncident: (incidentId: string, patch: IncidentActionRequest, signal?: AbortSignal): Promise<Record<string, unknown>> =>
    fetch(new URL(BASE + `/debug/incidents/${encodeURIComponent(incidentId)}`, window.location.origin).toString(), {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        ...buildControlHeaders(true, `/debug/incidents/${encodeURIComponent(incidentId)}`),
      },
      body: JSON.stringify(patch),
      signal,
    }).then(async (res) => {
      let payload: unknown = null;
      try {
        payload = await res.json();
      } catch {
        payload = null;
      }
      if (!res.ok) {
        throw new ApiError(res.status, `${res.status} ${res.statusText} - /debug/incidents/{id}`, payload);
      }
      return (payload ?? {}) as Record<string, unknown>;
    }),
  bulkDebugIncidents: (payload: IncidentBulkActionRequest, signal?: AbortSignal): Promise<Record<string, unknown>> =>
    post<Record<string, unknown>>("/debug/incidents/bulk", payload, signal),
  previewBulkDebugIncidents: (
    payload: IncidentBulkPreviewRequest,
    signal?: AbortSignal
  ): Promise<IncidentBulkPreviewResult> =>
    post<IncidentBulkPreviewResult>("/debug/incidents/bulk/preview", payload, signal),
  undoDebugIncidents: (signal?: AbortSignal): Promise<Record<string, unknown>> =>
    post<Record<string, unknown>>("/debug/incidents/undo", {}, signal),
  debugIncidentAudit: (limit = 50, signal?: AbortSignal): Promise<IncidentAuditEvent[]> =>
    get<IncidentAuditEvent[]>("/debug/incidents/audit", { limit }, signal).then(asArray<IncidentAuditEvent>),
  debugSecurityAudit: (limit = 100, signal?: AbortSignal): Promise<SecurityAuditEvent[]> =>
    get<SecurityAuditEvent[]>("/debug/security-audit", { limit }, signal).then(asArray<SecurityAuditEvent>),
  runDebugIncidentRunbook: (incidentId: string, signal?: AbortSignal): Promise<RunbookSessionDetail> =>
    post<RunbookSessionDetail>(`/debug/incidents/${encodeURIComponent(incidentId)}/runbook`, {}, signal),
  debugIncidentPolicy: (signal?: AbortSignal): Promise<AutoRunbookPolicy> =>
    get<AutoRunbookPolicy>("/debug/incidents-policy", undefined, signal),
  patchDebugIncidentPolicy: (policy: AutoRunbookPolicy, signal?: AbortSignal): Promise<AutoRunbookPolicy> =>
    fetch(new URL(BASE + "/debug/incidents-policy", window.location.origin).toString(), {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        ...buildControlHeaders(true, "/debug/incidents-policy"),
      },
      body: JSON.stringify(policy),
      signal,
    }).then(async (res) => {
      let payload: unknown = null;
      try {
        payload = await res.json();
      } catch {
        payload = null;
      }
      if (!res.ok) {
        throw new ApiError(res.status, `${res.status} ${res.statusText} - /debug/incidents/policy`, payload);
      }
      return payload as AutoRunbookPolicy;
    }),
  runAutoRunbookOnce: (signal?: AbortSignal): Promise<AutoRunbookResult> =>
    post<AutoRunbookResult>("/debug/incidents/auto-run", {}, signal),
  debugMacroPreset: (signal?: AbortSignal): Promise<MacroPresetConfig> =>
    get<MacroPresetConfig>("/debug/macro-preset", undefined, signal),
  patchDebugMacroPreset: (cfg: MacroPresetConfig, signal?: AbortSignal): Promise<MacroPresetConfig> =>
    fetch(new URL(BASE + "/debug/macro-preset", window.location.origin).toString(), {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        ...buildControlHeaders(true, "/debug/macro-preset"),
      },
      body: JSON.stringify(cfg),
      signal,
    }).then(async (res) => {
      let payload: unknown = null;
      try {
        payload = await res.json();
      } catch {
        payload = null;
      }
      if (!res.ok) {
        throw new ApiError(res.status, `${res.status} ${res.statusText} - /debug/macro-preset`, payload);
      }
      return payload as MacroPresetConfig;
    }),
};

export function streamLog(file: string, lastN = 50): EventSource {
  return new EventSource(`${BASE}/logs/stream?file=${encodeURIComponent(file)}&last_n=${lastN}`);
}

export type { LogsStreamLine };
