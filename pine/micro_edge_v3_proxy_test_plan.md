# MicroEdge V3 — Test Plan & Bug Report

## Bug Report (Pre-Fix)

| ID  | Severity | Line(s) | Description | Fix Applied |
|-----|----------|---------|-------------|-------------|
| B1  | CRITICAL | 108/122 | `n_filled` declared but **never incremented** → `fill_rate` always `0.0` after first attempt → gate permanently blocked after warmup | Implemented fill proxy check loop that increments `n_filled` on limit touch |
| B2  | CRITICAL | 122     | Integer division `n_filled / n_attempts` (both `int`) truncates to `0` — gate could never open even if fills occurred | Changed to `float(n_filled) / float(n_attempts)` |
| B3  | CRITICAL | 152/159 | `ta.barssince(entry_bar == bar_index[0])` — wrong API for time exit; `ta.barssince()` returns bars since condition was last true, but `bar_index[0]` is a constant so comparison is always false after entry bar | Replaced with `bar_index - strategy.opentrades.entry_bar_index(0) >= max_bars_hold` |
| B4  | HIGH     | 117–118 | `limit_long` / `limit_short` calculated but **never referenced** — fill proxy was dead code | Variables moved into entry blocks and used in `pending_limit` tracking |
| B5  | HIGH     | 149–160 | `atr` recalculated every bar inside position → SL/TP levels shift each bar (unintended dynamic trailing, not declared behavior) | SL/TP locked at entry bar into `var float trade_sl / trade_tp` |
| B6  | MEDIUM   | 167–168 | `plot(v3_score)` + `hline(0.5)` on overlay chart → score is `0–1`, price axis is in thousands → lines invisible | Removed; replaced with `barcolor()` which is overlay-safe |

---

## Test Plan

### T1 — Score Boundary Tests

**Goal:** Confirm V3 score components produce values in `[0, 1]`.

| Test | Condition | Expected |
|------|-----------|----------|
| T1.1 | `hl_spread == spread_ma` (range exactly average) | `spread_score = 0.5` |
| T1.2 | `hl_spread << spread_ma` (tight bar = compressed) | `spread_score → 1.0` |
| T1.3 | `hl_spread >> spread_ma` (wide bar = expanding) | `spread_score → 0.0` |
| T1.4 | Volume flat (fast SMA ≈ slow SMA) | `intensity_score ≈ 0.5` |
| T1.5 | Follow-through matches close direction | `followthru_score = 1.0` |
| T1.6 | Follow-through opposite to close direction | `followthru_score = 0.0` |
| T1.7 | All components max | `v3_raw = 1.0` |
| T1.8 | All components min | `v3_raw = 0.0` |

**How to verify:** Add `plot(spread_score)`, `plot(intensity_score)`, `plot(followthru_score)` temporarily in a separate indicator panel. Confirm values stay in `[0, 1]`.

---

### T2 — Toxicity Penalty Tests

**Goal:** Confirm penalty blends correctly and never pushes score out of `[0, 1]`.

| Test | Condition | Expected |
|------|-----------|----------|
| T2.1 | 50% up bars, 50% down bars | `tox_ratio = 0.0` (balanced = toxic) |
| T2.2 | All bars up | `tox_ratio = 1.0` (directional = clean) |
| T2.3 | `tox_penalty = 0.0` | `v3_score == v3_raw` (no penalty applied) |
| T2.4 | `tox_penalty = 1.0` | `v3_score == tox_score` (only toxicity) |

---

### T3 — Regime Filter Tests

**Goal:** Confirm the regime filter blocks entries during high-vol periods.

| Test | Condition | Expected |
|------|-----------|----------|
| T3.1 | ATR > 75th percentile of past 50 bars | `high_vol_regime = true`, no entries |
| T3.2 | ATR < 75th percentile | `regime_ok = true`, entries allowed |
| T3.3 | Chart background | Red tint visible during high-vol periods |

**How to verify:** Apply on a volatile crypto pair (e.g., BTC/USDT 1m during a news spike). Background should turn red on spike bars and no signal triangles should appear.

---

### T4 — Fill Proxy Tests

**Goal:** Confirm `n_filled` increments correctly and `fill_rate` is computed as float.

| Test | Condition | Expected |
|------|-----------|----------|
| T4.1 | Long signal; price drops to `close * (1 - bp/10000)` within `fill_wait_bars` | `n_filled` increments |
| T4.2 | Long signal; price never touches limit within window | `n_filled` unchanged (unfilled = 0) |
| T4.3 | `n_attempts = 3`, `n_filled = 2` | `fill_rate = 0.667` (not `0` from int division) |
| T4.4 | `n_attempts = 0` | `fill_rate = 1.0` (no division by zero) |

**How to verify:** Add `plot(fill_rate, "Fill Rate")` in a separate panel. After 5+ attempts, value should be between 0.0 and 1.0 (not always 0.0).

---

### T5 — Attempt Gate Tests

**Goal:** Confirm gate correctly transitions WARMUP → OPEN / BLOCKED.

| Test | Condition | Expected Dashboard |
|------|-----------|-------------------|
| T5.1 | `n_signals < gate_min_signals` | Gate shows `WARMUP`, all signals trade freely |
| T5.2 | `n_signals >= 20`, `fill_rate >= 0.5` | Gate shows `OPEN`, trading continues |
| T5.3 | `n_signals >= 20`, `fill_rate < 0.5` | Gate shows `BLOCKED`, orange X marks appear, no new entries |
| T5.4 | `gate_min_signals = 0` | Gate immediately active from bar 0 |

**How to verify:** Set `gate_min_signals = 5` and observe dashboard transitions in Strategy Tester.

---

### T6 — Exit Logic Tests

**Goal:** Confirm SL, TP, and time exits fire correctly and SL/TP are fixed at entry.

| Test | Condition | Expected |
|------|-----------|----------|
| T6.1 | Price drops `sl_atr * ATR` below entry close | Long trade closes at SL |
| T6.2 | Price rises `tp_atr * ATR` above entry close | Long trade closes at TP |
| T6.3 | Neither SL nor TP hit after `max_bars_hold` bars | Trade closes with `comment="timeout"` |
| T6.4 | ATR spikes during trade | SL/TP do **not** move (locked at entry) |
| T6.5 | Mirror tests for Short | Short SL = entry + ATR * sl_atr, TP = entry - ATR * tp_atr |

**How to verify:** Check Strategy Tester trade list for exit reason. Timeout exits will show `"timeout"` in comment column.

---

### T7 — Visualization Tests

**Goal:** Confirm all visual elements are visible and correctly positioned.

| Test | Expected |
|------|----------|
| T7.1 | Green triangle appears below bar on long entry |
| T7.2 | Red triangle appears above bar on short entry |
| T7.3 | Orange X appears when gate is blocked |
| T7.4 | Bar color turns blue when `v3_score >= score_threshold` and `regime_ok` |
| T7.5 | Red background tint during high-vol regime bars |
| T7.6 | Dashboard table visible top-right, 7 rows |
| T7.7 | Dashboard fill rate shows `%` format (e.g. `"66.7%"`) |

---

### T8 — No-Lookahead Validation

**Goal:** Confirm strategy does not use future data.

| Test | Method | Expected |
|------|--------|----------|
| T8.1 | Enable "Recalculate on every tick" in TradingView settings | Bar count and signal count should not change between realtime and confirmed bar mode |
| T8.2 | `followthru_len` references `close[N]`, `open[N]` | All indexed values use past bars (positive offset = past) |
| T8.3 | `barstate.isconfirmed` check | Signals only fire on closed bars |
| T8.4 | Regime uses `ta.percentile_nearest_rank(atr, regime_len, pct)` | Uses rolling past window only |

---

### T9 — Edge Case Tests

| Test | Condition | Expected |
|------|-----------|----------|
| T9.1 | `spread_ma = 0` (all bars have zero range, e.g. halted market) | `spread_shrink = 0.0` (guarded) |
| T9.2 | `close[followthru_len] = 0` | `ret_since = 0.0` (division guard in place) |
| T9.3 | No open trades when `strategy.position_size < 0` check runs | `opentrades.entry_bar_index(0)` returns `na`; `bar_index - na = na`; `na >= max_bars_hold = false` → safe |
| T9.4 | `tox_len` bars not yet available (early bars) | `ta.sum` returns `na` on early bars → `tox_ratio = na` → `v3_score = na` → signals = false → no crash |

---

## Checklist for TradingView Paste

- [ ] Script compiles without errors (Pine Editor green checkmark)
- [ ] Strategy Tester shows at least 1 trade on BTCUSDT 1h 1-year lookback
- [ ] Dashboard table visible top-right
- [ ] No red `"na"` values in dashboard after warmup
- [ ] Gate transitions from WARMUP to OPEN or BLOCKED after `gate_min_signals` signals
- [ ] Signal triangles visible on chart
- [ ] No signals appear during red-background (high-vol) bars
