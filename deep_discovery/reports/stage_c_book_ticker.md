# Stage C — Book Ticker Deep Mining
## Deep Discovery Expansion Protocol

**Run at:** 2026-06-01 19:36 UTC
**Caveat:** book_ticker data available only from Apr 11, 2026. Analysis restricted to Apr11–May08 VALIDATE window (531 signals).

---

## C1–C4. Per-Dimension Results (vs btc_aligned baseline, Apr11+ window)

| Filter | N (VALIDATE) | P&L 120s | Delta vs base | Verdict |
|---|---|---|---|---|
| book_imb_aligned | 251 | +2.576 | -0.547 | FAIL |
| spread_tightening | 0 | +0.000 | +0.000 | SKIP_N |
| hi_quote_intensity | 133 | +3.472 | +0.349 | FAIL |
| micro_aligned | 301 | +3.217 | +0.094 | FAIL |
| book_imb+micro_aligned | 164 | +2.573 | -0.551 | FAIL |
| book_imb+tightening | 0 | +0.000 | +0.000 | SKIP_N |

**Multiple-testing:** 8 tests, Bonferroni p < 0.0063

---

## Critical Structural Constraint

book_ticker is only available from **Apr 11, 2026** onward. This covers only the VALIDATE window (Apr 08–May 08). Any strategy using book_ticker features:
- Cannot be tested on the full TRAIN window (Feb 15 – Apr 08)
- Reduces the development sample dramatically
- Creates a potential look-ahead if thresholds are tuned on this narrow window

This limits book_ticker features to a **secondary qualifier** role, not a primary filter.

---

## Stage C Verdict: FAIL

**Chain-reaction lead:** book_ticker features do NOT improve edge in available window. Stage D must carry the conditioning load.
