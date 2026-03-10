# Eclipse Scalper — Mobile App Plan

## Stack Decision

**React Native (Expo)** — chosen because:
- Same TypeScript codebase as dashboard
- Reuse all existing `api/client.ts` types and fetch logic verbatim
- Same FastAPI backend, zero backend changes
- Cross-platform: iOS + Android from one codebase
- Expo Router gives file-based routing (familiar to web devs)
- OTA updates without App Store review

**Backend**: Existing FastAPI at port 8765 — unchanged. Mobile connects via LAN IP or tunnel (ngrok / Tailscale).

---

## Architecture

```
eclipse-scalper-mobile/
  app/
    (tabs)/
      index.tsx        ← Status (home tab)
      live.tsx         ← Live Monitor
      incidents.tsx    ← Control Tower / Incidents
      tools.tsx        ← Debug Tools
      logs.tsx         ← Log Browser
    settings.tsx
    _layout.tsx        ← Tab navigator shell
  components/
    StatusCard.tsx
    RiskCard.tsx
    IncidentRow.tsx
    LogViewer.tsx
    CommandSheet.tsx   ← bottom sheet (replaces Ctrl+K)
    StatusBar.tsx      ← top status pill
  api/
    client.ts          ← COPY from dashboard (same types)
    types.ts           ← COPY from dashboard
  context/
    BackendContext.tsx
    AuthContext.tsx
  hooks/
    usePoll.ts         ← COPY from dashboard
  constants/
    theme.ts           ← amber/dark design tokens
```

---

## Design Language (carry over from web)

- Colors: identical CSS vars translated to JS constants
  - bg: `#07080f`, surface: `#0c0f1a`, accent: `#f0a020`
  - green: `#00c96b`, red: `#ff4040`
- Font: `JetBrains Mono` (numbers) + system font (UI)
- Bottom tab bar (5 tabs max, most critical first)
- Pull-to-refresh on every screen
- Bottom sheets instead of modals
- Haptic feedback on critical alerts and one-click actions

---

## Phase 1 — Foundation

**Goal**: App shell boots, connects to backend, shows API status.

### Tasks
1. `expo init eclipse-scalper-mobile --template expo-router`
2. Copy `api/client.ts` and `api/types.ts` from dashboard verbatim
3. Copy `hooks/usePoll.ts` from dashboard verbatim
4. `BackendContext`: poll `/api/health` every 5s, expose `backendUp` globally
5. `AuthContext`: store `apiKey + operator + role` in `SecureStore`
6. `constants/theme.ts`: all color/spacing tokens as JS constants
7. `_layout.tsx`: bottom tab navigator (5 tabs) + top status pill (API UP/DOWN)
8. Settings screen: enter backend URL (default `http://192.168.x.x:8765`), role toggle, test connection button
9. `StatusBar` component: fixed top strip showing API status dot + last poll time

### Deliverable
App opens → connects to backend → shows UP/DOWN pill → can navigate between empty tabs.

---

## Phase 2 — Status & Live (Core Monitoring)

**Goal**: Two most-used screens fully working. Operator can monitor system health from phone.

### Status Tab (`/`)
- Scoreboard strip: Orders / Fills / Cancels / Kill switches (horizontal scroll cards)
- Data freshness badge: LIVE / DEGRADED / STALE (large, color-coded)
- Collector health row: alive, trades/sec, mark/sec
- Gate status: per-symbol, hit rate, delta vs baseline
- Recent regime events: last 5, scrollable list
- Pull-to-refresh

### Live Tab (`/live`)
- Microstructure metrics: trades/sec + mark/sec (large numbers, JetBrains Mono)
- 9 risk cards in 2-column grid (tap to expand detail):
  - Liq Alert, Spread Stress, Fill Toxicity, Latency Stress
  - Watchboard, Book Proxy Pressure, Return Shock, Volatility Burst, Volume Vacuum
- Each card: level badge (QUIET / ELEVATED / SEVERE) + headline + operator note
- Auto-refresh every 15s with countdown indicator
- SEVERE cards pulse red to attract attention

### Deliverable
Operator can check system health and risk levels without opening laptop.

---

## Phase 3 — Incidents & Tools (One-Click Operations)

**Goal**: All critical operator actions completable from phone in 1-2 taps.

### Incidents Tab (`/incidents`)
- Inbox list: incident rows with type / level / status / age
- Swipe right → **ACK** (one swipe, no confirm dialog)
- Swipe left → **Snooze 30 min**
- Tap → detail sheet: title, file, query, suggested runbook
- Detail sheet has: [Run Runbook] [Ack] [Snooze] [Mute] buttons
- Badge count on tab icon (red dot when > 0 open incidents)
- Bulk action bottom sheet: "Ack All" / "Snooze All filtered"
- Auto-runbook policy toggle (enabled / min_level / cooldown)
- Pull-to-refresh + 15s auto-poll

### Tools Tab (`/tools`)
- Action list: all debug actions from `/api/debug/actions`
- Each row: action name + description + timeout + [▶ Run] button
- Tap [▶ Run] → confirmation sheet → execute → inline output viewer (scrollable)
- Recent history: last 10 runs with ok/fail badge + duration
- Quick runbook: select 2-4 actions → [Run Sequence] → step-by-step result

### Command Sheet (global, swipe up from bottom)
- Replaces Ctrl+K from web
- Swipe up from tab bar → bottom sheet with search input
- Search across: pages, actions, log packs, incidents
- Tap result → navigates + pre-fills filters

### Deliverable
Full incident lifecycle manageable from phone. Debug actions runnable in 2 taps.

---

## Phase 4 — Logs & Signals

**Goal**: Log browsing and signal inspection on mobile.

### Logs Tab (`/logs`)
- File list: tap to open
- Quick filter packs as chips: [Regime] [Shutdown] [Timeout] [No Match]
- Log tail viewer: last 200 lines, monospace, color-coded by level
  - ERROR → red, WARNING → yellow, INFO → normal, DEBUG → muted
- Search bar: filter lines containing text
- [Stream] toggle: SSE live stream (auto-scrolls to bottom)
- Copy line on long-press
- Incident session: [Start] / [Stop] / [Export] buttons

### Signals (inside Status tab, bottom section OR separate sub-tab)
- Signal events: symbol / direction / regime / confidence list
- Stability events: allowed/blocked with reason
- Quality events: severity + issues list
- Symbol filter dropdown (ETH / BTC / All)

### Deliverable
Log triage fully doable on mobile. Signals inspectable without laptop.

---

## Phase 5 — Research & Polish

**Goal**: Feature parity with web + mobile-native polish.

### Research Screen (inside Live tab or separate)
- Watchboard lanes: card list with level + headline + recommended action
- Research fitness state: OK / WARNING / DEGRADED
- Pocket overlay: tap pocket → detail sheet with trade stats

### Push Notifications (via Expo Notifications)
- Trigger: any incident with level >= WARNING added to inbox
- Notification payload: title + level + incident type
- Tap notification → opens Incidents tab, scrolled to that incident
- Implementation: poll `/api/debug/incidents` every 60s in background task

### Offline Resilience
- Cache last known data per screen in AsyncStorage
- Show cached data with "stale" badge when API unreachable
- Auto-reconnect with exponential backoff (same logic as `usePoll`)

### Performance
- FlatList with `getItemLayout` for log viewer (no jank on 1000+ lines)
- Memoized risk cards (only re-render when level changes)
- SSE connection managed as singleton (one connection per file, reuse across navigations)

### Polish
- Haptic feedback:
  - Light: pull-to-refresh complete
  - Medium: action button tap
  - Heavy: SEVERE alert appears / incident acked
- Dark mode only (matches web)
- Landscape support for log viewer and risk cards grid
- Accessibility: all interactive elements have `accessibilityLabel`

### Deliverable
Full feature parity. App publishable to TestFlight / internal Android track.

---

## API / Backend Compatibility

All endpoints are already REST + SSE. Zero backend changes needed.

| Mobile feature         | Existing endpoint                        |
|------------------------|------------------------------------------|
| Status overview        | `GET /api/overview`                      |
| Live metrics           | `GET /api/live/metrics`                  |
| Risk cards             | `GET /api/risk-overview`                 |
| Incidents              | `GET /api/debug/incidents`               |
| Ack / patch incident   | `PATCH /api/debug/incidents/{id}`        |
| Run debug action       | `POST /api/debug/run`                    |
| Run runbook            | `POST /api/debug/runbook`                |
| Log files              | `GET /api/logs`                          |
| Log tail               | `GET /api/logs/tail`                     |
| Log stream (SSE)       | `GET /api/logs/stream`                   |
| Signals / regimes      | `GET /api/events/signals` etc.           |
| Health                 | `GET /api/health`                        |

---

## Connection Setup (Local Network)

1. Find backend LAN IP: `ipconfig` → IPv4 address (e.g. `192.168.1.42`)
2. Backend already listens on `0.0.0.0:8765` — accessible on LAN
3. Settings screen: enter `http://192.168.1.42:8765` as backend URL
4. For remote access: run `tailscale` on the server — enter Tailscale IP in settings

---

## File Reuse from Dashboard

Copy these files verbatim (no changes needed):

```
dashboard/frontend/src/api/client.ts   → mobile/api/client.ts
dashboard/frontend/src/api/types.ts    → mobile/api/types.ts
dashboard/frontend/src/hooks/usePoll.ts → mobile/hooks/usePoll.ts
```

Adapt these (replace CSS vars with JS theme constants):
```
dashboard/frontend/src/context/AuthContext.tsx
dashboard/frontend/src/context/BackendStatusContext.tsx
dashboard/frontend/src/context/ApiErrorContext.tsx
```

---

## Codex Instructions Per Phase

Each phase should be given as a separate Codex task with this preamble:

```
Project: Eclipse Scalper mobile app (Expo + React Native)
Backend: FastAPI at configurable URL (default http://192.168.x.x:8765)
Language: TypeScript
Design: Dark theme, amber accent (#f0a020), JetBrains Mono for numbers
Reuse: Copy api/client.ts and api/types.ts from dashboard/frontend/src/api/ verbatim
Do NOT modify the backend.
```

Then paste the specific phase tasks from this document.
