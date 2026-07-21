# S34 Handoff Brief — for Codex

Hand this file to Codex to continue the S34 liquidation-cascade research without
re-deriving context. Everything below is the state as of 2026-06-28.

---

## 1. One-paragraph state

We exhaustively tested the S34 liquidation-cascade family as a *reactive taker*
signal and it does not produce a deployable edge. The one genuinely real
phenomenon: **large liquidation cascades mean-revert at swing scale** (fade, not
continue), and the reversal is **sharper after a deeper price overshoot** (V-depth
conditioning). But as a directional taker it is killed by **negative skew** (wins
often — 55-62% — but the minority of runaways lose big, so total P&L is negative)
plus real spread + only ~2 months of tradeable (book-fillable) data.

## 2. Signal map — dead vs real

DEAD (do not re-mine as taker):
- Continuation at threshold cross — all 12 routes RESEARCH_ONLY/BLOCKED, 0 PAPER_CANDIDATE.
- Conditional continuation — 13 knowable features × terciles, no cal+hold-positive subset.
- Fade at scalp horizons (<=1h) — net negative; move is spent before the knowable cross.
- Onset entry / early-build velocity — coin flip (move happens faster than it can be confirmed).
- Order-flow (agg-trade OFI) lead — direction is RIGHT (gross ~+1-2 bps, 52-58% win) but smaller than taker cost.

REAL but not harvested as taker:
- Swing reversal (fade big cascades, 1h-24h): beta-controlled signal is consistently REVERSAL.
- V-depth conditioning: deep-overshoot cascades fade best; deep bin at 4h is cal+hold positive on a mark/median screen.
- BLOCKER: negative skew + real bid/ask spread + ~2 months data → 0 leads in the real book-fill stop backtest.

## 3. Recommended starting point — the maker reframe

The negative skew IS the taker problem: you cross the spread to fade, and when the
cascade doesn't revert you chase it. A **maker** does the opposite — rests a limit
order at/beyond the cascade extreme, into the liquidity vacuum the cascade creates.
You EARN the spread (+ maker rebate) instead of paying it, and you only get filled
when price reaches your level (favorable entry). This can flip the skew. This is
the highest-leverage untested idea and it starts from the one real signal (deep-V
reversal).

## 4. Step-by-step plan

1. **Reproduce baseline** (sanity): run
   `python tools/research_s34_vshape_conditioning.py` and
   `python tools/research_s34_reversal_stop_backtest.py --symbol ETHUSDT --threshold 200000 --min-vdepth-bps 28 --horizon-hr 4`.
   Confirm: deep-V mark-median positive, real book-fill total still negative (skew).
2. **Build a maker-entry model.** New tool, e.g. `research_s34_maker_fade.py`. For
   each deep-V cascade anchor, instead of a taker fill at bid/ask, model a LIMIT
   order resting at the cascade extreme (or extreme ± k bps). Fill ONLY if the mark
   path trades through the limit by a conservative margin (book_ticker is top-of-book
   only — no queue position, so be pessimistic: require the limit to be crossed, not
   just touched). Entry price = the limit (favorable); add maker fee/rebate, not taker.
   Exit: fixed horizon and/or a maker take-profit. Per-trade P&L, chronological holdout.
3. **Test skew flip.** Does maker entry turn the deep-V fade total positive on BOTH
   calibration and holdout? Report TOTAL sum and skew, not just median/win-rate.
4. **If yes → forward shadow-paper.** Add a maker fade rule (deep-V filter) to the
   `tools/s34_shadow_paper_runner.py` pattern and run it forward, validating live.
   Wire it into `tools/s34_cascade_navigation_dashboard.py` as the first real
   `FADE_VIABLE` permission once it clears the gate.
5. **In parallel — fix the data ceiling.** The binding constraint is ~2 months of
   book-fillable data (liquidations span Feb-Jun with a May gap; book_ticker fills
   only cover ~Apr+Jun). Ensure continuous liquidation + book_ticker collection to
   build 6+ months before trusting any swing edge.

## 5. Tools already built (reuse, do not redo)

- `tools/s34_cascade_navigation_dashboard.py` — point-in-time, no-lookahead 5-panel permission layer (+ `tests/test_s34_cascade_navigation.py`).
- `tools/s34_cascade_navigation_survey.py` — historical phase/shape/executability survey.
- `tools/research_s34_knowable_anchor_route_recheck.py` — clean route backtest (supports `--invert-direction` fade, V-shape/day-trend filters).
- `tools/research_s34_conditional_edge_screen.py` — knowable-feature conditioning screen.
- `tools/research_s34_horizon_decay.py` — pure directional decay across horizons.
- `tools/research_s34_onset_precursor.py`, `tools/research_s34_early_build_entry.py` — onset/velocity precursor tests.
- `tools/research_s34_orderflow_lead.py` — agg-trade OFI leading-signal test.
- `tools/research_s34_liq_swing_event.py` — beta-controlled swing-event (reversal) screen.
- `tools/research_s34_reversal_backtest.py`, `..._regime_diag.py`, `..._stop_backtest.py` (has `--min-vdepth-bps`), `..._vshape_conditioning.py` — the reversal backtest chain.
- Reports + JSON for each in `reports/research/s34/`.

## 6. Methodology guardrails (hard rules — this is how the old evidence got contaminated)

- **No lookahead.** Use only data with ts <= decision time. Run feature sets through
  `tools/s34_feature_availability.py::assert_feature_set_available`. The old shadow
  paper "+1805 bps" was lookahead (final bucket notional stamped at an early ts).
- **Chronological holdout.** Judge on the held-out later period, never the fit period.
- **Median is a trap.** Always report TOTAL P&L and skew. This family wins often but
  loses big — median/win-rate look great while expectancy is negative.
- **Pay for fills.** Taker = cross bid/ask + fee. Maker = model fills conservatively
  (top-of-book data only). Never use mark-to-mark as if it were tradeable.
- **Machine limits:** no parallel Python/PowerShell processes (RAM); max 2 test files
  per pytest call; use D: drive; on Windows run pytest with `--basetemp=D:/...`.

## 7. The honest framing for the operator

The reversal/V-depth pattern is real — the market read is correct. The open question
is purely structural: can you exploit a high-win-rate / negative-skew pattern by
being the liquidity provider (maker) instead of the taker, and is there enough data
to trust it. Start there. Don't re-test the taker directional bet — that's settled.
