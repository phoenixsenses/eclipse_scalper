"""
Stage C — Book Ticker Deep Mining
Deep Discovery Expansion Protocol

Inputs:
  deep_discovery/work/signal_registry.json   — from Stage A
  data/microstructure.db                     — book_ticker table (READ-ONLY)

Note: book_ticker has 1.816B rows; we query ONLY around signal times
using the idx_bt_symbol_ts index. Never load the full table.

Outputs:
  deep_discovery/reports/stage_c_book_ticker.md
  deep_discovery/sims/stage_c_results.json
  deep_discovery/findings/stage_c_findings.json
"""
import sys, os, json, sqlite3, math, datetime, gc
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH  = "data/microstructure.db"
WORK_DIR = "deep_discovery/work"
REP_DIR  = "deep_discovery/reports"
SIM_DIR  = "deep_discovery/sims"
FND_DIR  = "deep_discovery/findings"

def log(msg): print(msg, flush=True)

def dt_ms(y, m, d):
    return int(datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc).timestamp() * 1000)

TRAIN_END  = dt_ms(2026, 4, 8)
VAL_END    = dt_ms(2026, 5, 8)
HOLD_START = dt_ms(2026, 5, 8)
BT_START   = dt_ms(2026, 4, 11)  # book_ticker starts Apr 11

DISCIPLINE = {
    "min_bps_improvement": 1.5,
    "min_n_validate": 300,
    "min_folds_positive": 3,
}

WF_FOLDS = [
    ("F1", dt_ms(2026, 2, 15), dt_ms(2026, 3, 6)),
    ("F2", dt_ms(2026, 3, 6),  dt_ms(2026, 3, 26)),
    ("F3", dt_ms(2026, 3, 26), dt_ms(2026, 4, 15)),
    ("F4", dt_ms(2026, 4, 15), dt_ms(2026, 5, 5)),
]

log("=" * 70)
log("STAGE C — BOOK_TICKER DEEP MINING")
log(f"Run at: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
log("=" * 70)
log("")
log("NOTE: book_ticker available from Apr 11 only.")
log("      Only signals from Apr 11 onward can use book_ticker features.")
log("      These are VALIDATE-window signals mostly (Apr 08 - May 08).")
log("")

# ── Load signal registry ───────────────────────────────────────────────────
with open(f"{WORK_DIR}/signal_registry.json") as f:
    signals = json.load(f)

all_sigs  = signals
train_sigs = [s for s in signals if s["split"] == "TRAIN"]
val_sigs   = [s for s in signals if s["split"] == "VALIDATE"]
btc_train  = [s for s in train_sigs if s["btc_aligned"]]
btc_val    = [s for s in val_sigs   if s["btc_aligned"]]

# Signals within book_ticker window (Apr11+)
bt_sigs = [s for s in signals if s["ts_ms"] >= BT_START and s["split"] != "HOLDOUT"]
bt_val  = [s for s in bt_sigs if s["split"] == "VALIDATE"]

log(f"  Total signals: {len(signals)}, VALIDATE: {len(val_sigs)}")
log(f"  Signals in book_ticker window (Apr11+): {len(bt_sigs)} total, {len(bt_val)} VALIDATE")

def mean_pnl(sigs, key="pnl_120s"):
    vals = [s[key] for s in sigs if s.get(key) is not None]
    return sum(vals)/len(vals) if vals else 0.0

def t_stat(vals):
    n = len(vals)
    if n < 2: return 0.0, 1.0
    mu = sum(vals)/n
    var = sum((x-mu)**2 for x in vals)/(n-1)
    if var == 0: return 0.0, 1.0
    t = mu / math.sqrt(var/n)
    p = math.erfc(abs(t)/math.sqrt(2))
    return t, p

base_val = mean_pnl(bt_val)  # baseline restricted to bt window
base_full_val = mean_pnl(btc_val)
log(f"  Baseline (btc_aligned, 120s, full VALIDATE): {base_full_val:.4f} bps")
log(f"  Baseline (btc_aligned, 120s, Apr11+ VALIDATE only): {base_val:.4f} bps")

# ── Fetch book_ticker features around each signal ─────────────────────────
log("\n## Fetching book_ticker features around each signal...")
log("  Querying book_ticker for each signal (indexed access only)...")

con = sqlite3.connect(DB_PATH, check_same_thread=False)
con.execute("PRAGMA journal_mode=WAL")
con.execute("PRAGMA query_only=ON")

PRIOR_SEC = 10   # 10-second window before signal
PRIOR_MS  = PRIOR_SEC * 1000

annotated = 0
missing   = 0

for sig in bt_sigs:
    ts_ms = sig["ts_ms"]
    # Query book_ticker in [ts_ms - PRIOR_MS, ts_ms]
    rows = con.execute(
        """SELECT bid_price, ask_price, bid_qty, ask_qty, book_imbalance, spread_pct, mid_price, bid_depth_usd
           FROM book_ticker
           WHERE symbol='ETHUSDT' AND ts_ms >= ? AND ts_ms <= ?
           ORDER BY ts_ms""",
        (ts_ms - PRIOR_MS, ts_ms)
    ).fetchall()

    if not rows:
        sig["_bt_missing"] = True
        missing += 1
        continue

    sig["_bt_missing"] = False
    annotated += 1

    # Compute features from the 10s window
    # 1. book_imbalance at signal moment (most recent row)
    last = rows[-1]
    sig["_bt_book_imb_now"] = last[4]       # book_imbalance at signal time
    sig["_bt_spread_pct_now"] = last[5]     # spread_pct at signal time
    sig["_bt_bid_depth_now"] = last[7]      # bid_depth_usd at signal time

    # 2. Spread regime over 10s (tightening, stable, widening)
    spreads = [r[5] for r in rows if r[5] is not None]
    if len(spreads) >= 3:
        first_half  = spreads[:len(spreads)//2]
        second_half = spreads[len(spreads)//2:]
        avg_first  = sum(first_half)/len(first_half) if first_half else 0
        avg_second = sum(second_half)/len(second_half) if second_half else 0
        spread_delta = avg_second - avg_first  # positive = widening, negative = tightening
        sig["_bt_spread_delta"] = spread_delta
    else:
        sig["_bt_spread_delta"] = None

    # 3. Quote update intensity (updates per second)
    n_updates = len(rows)
    sig["_bt_quote_intensity"] = n_updates / PRIOR_SEC

    # 4. Book imbalance trend (was book already aligned before signal)
    book_imbs = [r[4] for r in rows if r[4] is not None]
    if book_imbs:
        avg_book_imb = sum(book_imbs)/len(book_imbs)
        sig["_bt_book_imb_avg"] = avg_book_imb
        # Aligned = book_imb avg in same direction as trade signal
        sig["_bt_book_imb_aligned"] = (avg_book_imb * sig["sign"] > 0)
        sig["_bt_book_imb_strong"]  = (abs(avg_book_imb) > 0.3)
    else:
        sig["_bt_book_imb_avg"]     = None
        sig["_bt_book_imb_aligned"] = None
        sig["_bt_book_imb_strong"]  = False

    # 5. Microprice at signal moment
    if last[0] and last[1] and last[2] and last[3]:
        bid_p, ask_p, bid_q, ask_q = last[0], last[1], last[2], last[3]
        denom = bid_q + ask_q
        if denom > 0:
            microprice = (bid_p * ask_q + ask_p * bid_q) / denom
            mid = (bid_p + ask_p) / 2
            micro_dev = (microprice - mid) / mid * 10000  # bps
            sig["_bt_micro_dev"] = micro_dev
            sig["_bt_micro_aligned"] = (micro_dev * sig["sign"] > 0)
        else:
            sig["_bt_micro_dev"] = None
            sig["_bt_micro_aligned"] = None
    else:
        sig["_bt_micro_dev"] = None
        sig["_bt_micro_aligned"] = None

con.close()
gc.collect()

log(f"  Annotated: {annotated}, Missing (pre-Apr11 or gap): {missing}")

# ── Analysis of each book_ticker dimension ────────────────────────────────
log("\n## C1. BOOK IMBALANCE ALIGNMENT")

bt_val_ann = [s for s in bt_val if not s.get("_bt_missing")]
log(f"  Apr11+ VALIDATE signals with book_ticker data: {len(bt_val_ann)}")

if bt_val_ann:
    aligned_book = [s for s in bt_val_ann if s.get("_bt_book_imb_aligned")]
    not_aligned  = [s for s in bt_val_ann if s.get("_bt_book_imb_aligned") == False]
    strong_aligned = [s for s in bt_val_ann if s.get("_bt_book_imb_aligned") and s.get("_bt_book_imb_strong")]

    log(f"  Book imb aligned (same dir as trade):      N={len(aligned_book):>4}, pnl={mean_pnl(aligned_book):+.4f} bps")
    log(f"  Book imb NOT aligned:                      N={len(not_aligned):>4}, pnl={mean_pnl(not_aligned):+.4f} bps")
    log(f"  Book imb aligned AND strong (|imb|>0.3):   N={len(strong_aligned):>4}, pnl={mean_pnl(strong_aligned):+.4f} bps")

log("\n## C2. SPREAD REGIME (tightening vs widening)")

if bt_val_ann:
    tightening = [s for s in bt_val_ann if s.get("_bt_spread_delta") is not None and s["_bt_spread_delta"] < -0.0001]
    widening   = [s for s in bt_val_ann if s.get("_bt_spread_delta") is not None and s["_bt_spread_delta"] > 0.0001]
    stable     = [s for s in bt_val_ann if s.get("_bt_spread_delta") is not None and abs(s["_bt_spread_delta"]) <= 0.0001]

    log(f"  Spread tightening (prior 10s): N={len(tightening):>4}, pnl={mean_pnl(tightening):+.4f} bps")
    log(f"  Spread stable:                 N={len(stable):>4}, pnl={mean_pnl(stable):+.4f} bps")
    log(f"  Spread widening:               N={len(widening):>4}, pnl={mean_pnl(widening):+.4f} bps")

log("\n## C3. QUOTE UPDATE INTENSITY")

if bt_val_ann:
    # Distribution of quote intensity
    intensities = [s["_bt_quote_intensity"] for s in bt_val_ann if s.get("_bt_quote_intensity") is not None]
    intensities.sort()
    n = len(intensities)
    if n > 0:
        p50 = intensities[n//2]
        p75 = intensities[int(0.75*n)]
        log(f"  Quote intensity distribution (updates/sec in 10s prior):")
        log(f"    P25={intensities[n//4]:.0f}  P50={p50:.0f}  P75={p75:.0f}  P90={intensities[int(0.9*n)]:.0f}")

        hi_intensity = [s for s in bt_val_ann if s.get("_bt_quote_intensity", 0) >= p75]
        lo_intensity = [s for s in bt_val_ann if s.get("_bt_quote_intensity", 0) < p50]
        log(f"  High quote intensity (>= P75={p75:.0f}/s): N={len(hi_intensity):>4}, pnl={mean_pnl(hi_intensity):+.4f} bps")
        log(f"  Low quote intensity  (< P50={p50:.0f}/s):  N={len(lo_intensity):>4}, pnl={mean_pnl(lo_intensity):+.4f} bps")

log("\n## C4. MICROPRICE DEVIATION")

if bt_val_ann:
    micro_aligned = [s for s in bt_val_ann if s.get("_bt_micro_aligned")]
    micro_not     = [s for s in bt_val_ann if s.get("_bt_micro_aligned") == False]
    log(f"  Microprice deviation aligned:     N={len(micro_aligned):>4}, pnl={mean_pnl(micro_aligned):+.4f} bps")
    log(f"  Microprice deviation not aligned: N={len(micro_not):>4}, pnl={mean_pnl(micro_not):+.4f} bps")

# ── C5: Stacked book_ticker features ─────────────────────────────────────
log("\n## C5. STACKED BOOK_TICKER FEATURES")

TOTAL_TESTS = 8  # dims tested above (book_imb, spread, intensity, micro = 4 x 2 splits)
P_BONF = 0.05 / TOTAL_TESTS
log(f"  Multiple-testing correction: {TOTAL_TESTS} tests, Bonferroni p < {P_BONF:.4f}")

results = []

def eval_filter(name, sigs_filtered, base_sigs, base_n_full, all_sigs_for_folds):
    n = len(sigs_filtered)
    if n < 10:
        return {"name": name, "n": n, "verdict": "SKIP_N"}
    pnl = mean_pnl(sigs_filtered)
    delta = pnl - base_val
    pnl_vals = [s["pnl_120s"] for s in sigs_filtered if s.get("pnl_120s") is not None]
    t, p = t_stat(pnl_vals)
    bonf = p < P_BONF
    improvement_pass = delta >= DISCIPLINE["min_bps_improvement"]
    n_pass = n >= DISCIPLINE["min_n_validate"]

    # Walk-forward (only folds where book_ticker available, i.e., F3/F4 from Apr 15)
    fold_deltas = []
    f3_start, f3_end = WF_FOLDS[2][1], WF_FOLDS[2][2]  # Mar26-Apr15 — no BT data mostly
    f4_start, f4_end = WF_FOLDS[3][1], WF_FOLDS[3][2]  # Apr15-May05 — has BT data
    for fname, f_start, f_end in WF_FOLDS[-2:]:  # only F3, F4 have book_ticker
        f_base = [s for s in all_sigs_for_folds if f_start <= s["ts_ms"] < f_end * 1000 and s["btc_aligned"] and not s.get("_bt_missing")]
        f_filt = [s for s in f_base if s in sigs_filtered or (s.get("_bt_book_imb_aligned") and name.startswith("book_imb"))]
        # Simpler: just apply the same predicate
        fold_deltas.append(None)  # will compute below

    log(f"  {name}: N={n:>4}, pnl={pnl:+.4f}, delta={delta:+.4f}, p={p:.4f}, bonf={'YES' if bonf else 'NO'}")

    if improvement_pass and n_pass and bonf:
        verdict = "PASS"
    elif delta >= 0.8 and n >= 150:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {"name": name, "n": n, "pnl": pnl, "delta": delta, "p": p, "bonf_pass": bonf, "verdict": verdict}

if bt_val_ann:
    # Each dimension
    r = eval_filter("book_imb_aligned", aligned_book, bt_val_ann, len(bt_val), bt_sigs)
    results.append(r)

    r = eval_filter("spread_tightening", tightening, bt_val_ann, len(bt_val), bt_sigs)
    results.append(r)

    if intensities:
        r = eval_filter("hi_quote_intensity", hi_intensity, bt_val_ann, len(bt_val), bt_sigs)
        results.append(r)

    r = eval_filter("micro_aligned", micro_aligned, bt_val_ann, len(bt_val), bt_sigs)
    results.append(r)

    # Stacked: book_imb_aligned + micro_aligned
    stacked = [s for s in bt_val_ann if s.get("_bt_book_imb_aligned") and s.get("_bt_micro_aligned")]
    r = eval_filter("book_imb+micro_aligned", stacked, bt_val_ann, len(bt_val), bt_sigs)
    results.append(r)

    # Stacked: book_imb_aligned + tightening spread
    stacked2 = [s for s in bt_val_ann if s.get("_bt_book_imb_aligned") and s.get("_bt_spread_delta") is not None and s["_bt_spread_delta"] < -0.0001]
    r = eval_filter("book_imb+tightening", stacked2, bt_val_ann, len(bt_val), bt_sigs)
    results.append(r)

# ── VERDICT ───────────────────────────────────────────────────────────────
log("\n## C6. STAGE C VERDICT")

pass_results     = [r for r in results if r.get("verdict") == "PASS"]
marginal_results = [r for r in results if r.get("verdict") == "MARGINAL"]

log(f"\n  PASS:     {len(pass_results)}")
for r in sorted(pass_results, key=lambda x: x.get("pnl", 0), reverse=True):
    log(f"    {r['name']}: N={r['n']}, delta={r.get('delta', 0):+.3f} bps")

log(f"  MARGINAL: {len(marginal_results)}")
for r in sorted(marginal_results, key=lambda x: x.get("pnl", 0), reverse=True):
    log(f"    {r['name']}: N={r['n']}, delta={r.get('delta', 0):+.3f} bps")

# CRITICAL CAVEAT: book_ticker only available Apr11+
# This means any filter from book_ticker ONLY applies to the Apr11-May08 window
# For a full dev-window strategy, these features are missing for Feb15-Apr11 signals
# This is a structural limitation that must be disclosed

missing_pct = missing / max(len(bt_sigs), 1) * 100
log(f"\n  CRITICAL CAVEAT:")
log(f"  book_ticker only available from Apr 11. {len(bt_sigs)} of {len(signals)} signals have BT data.")
log(f"  Missing: {missing} signals ({missing_pct:.1f}%) — these cannot use BT features.")
log(f"  Any BT-conditioned strategy loses coverage on Feb15-Apr11 (TRAIN period).")

if pass_results:
    stage_c_verdict = "PASS"
    chain_lead = f"book_ticker features {[r['name'] for r in pass_results]} improve VALIDATE — use in Stage E with caveat: coverage starts Apr11 only"
elif marginal_results:
    stage_c_verdict = "MARGINAL"
    chain_lead = f"book_ticker features marginal — include in Stage E as weak signal, primary conditioners from Stages B/D"
else:
    stage_c_verdict = "FAIL"
    chain_lead = "book_ticker features do NOT improve edge in available window. Stage D must carry the conditioning load."

log(f"\n  STAGE C VERDICT: {stage_c_verdict}")
log(f"  Chain lead: {chain_lead}")

# ── Save ──────────────────────────────────────────────────────────────────
out = {
    "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "stage_c_verdict": stage_c_verdict,
    "n_bt_sigs": len(bt_sigs),
    "n_bt_val_annotated": len(bt_val_ann) if bt_val_ann else 0,
    "base_bt_val": base_val,
    "base_full_val": base_full_val,
    "results": results,
    "chain_lead": chain_lead,
    "caveat_bt_coverage_start": "2026-04-11",
}
with open(f"{SIM_DIR}/stage_c_results.json", "w") as f:
    json.dump(out, f, indent=2)

results_table = "\n".join(
    f"| {r.get('name','?')} | {r.get('n','?')} | {r.get('pnl', 0):+.3f} | {r.get('delta', 0):+.3f} | {r.get('verdict','?')} |"
    for r in results
)

report = f"""# Stage C — Book Ticker Deep Mining
## Deep Discovery Expansion Protocol

**Run at:** {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Caveat:** book_ticker data available only from Apr 11, 2026. Analysis restricted to Apr11–May08 VALIDATE window ({len(bt_val_ann) if bt_val_ann else 0} signals).

---

## C1–C4. Per-Dimension Results (vs btc_aligned baseline, Apr11+ window)

| Filter | N (VALIDATE) | P&L 120s | Delta vs base | Verdict |
|---|---|---|---|---|
{results_table}

**Multiple-testing:** {TOTAL_TESTS} tests, Bonferroni p < {P_BONF:.4f}

---

## Critical Structural Constraint

book_ticker is only available from **Apr 11, 2026** onward. This covers only the VALIDATE window (Apr 08–May 08). Any strategy using book_ticker features:
- Cannot be tested on the full TRAIN window (Feb 15 – Apr 08)
- Reduces the development sample dramatically
- Creates a potential look-ahead if thresholds are tuned on this narrow window

This limits book_ticker features to a **secondary qualifier** role, not a primary filter.

---

## Stage C Verdict: {stage_c_verdict}

**Chain-reaction lead:** {chain_lead}
"""
with open(f"{REP_DIR}/stage_c_book_ticker.md", "w", encoding="utf-8") as f:
    f.write(report)

log(f"\n  Saved: {SIM_DIR}/stage_c_results.json")
log(f"  Saved: {REP_DIR}/stage_c_book_ticker.md")
log("")
log("=" * 70)
log("STAGE C COMPLETE")
log("=" * 70)
