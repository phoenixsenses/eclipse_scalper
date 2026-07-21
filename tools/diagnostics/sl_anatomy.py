# encoding: utf-8
"""
SL trade anatomy + liq_count correlation with MAE.
Finds what SL trades have in common and whether liq_count predicts outcome quality.
"""
import sqlite3, json
from pathlib import Path
from datetime import datetime, timezone

ROOT     = Path("D:/eclipse_scalper")
INTEL_DB = ROOT / "data" / "s34_intelligence.db"
MICRO_DB = ROOT / "data" / "microstructure.db"

RULES = [
    "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
    "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
]

intel = sqlite3.connect(f"file:{INTEL_DB}?mode=ro", uri=True)
micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)

rows = intel.execute(
    "SELECT trade_id, rule_name, entry_ts_ms, net_bps, exit_reason, trade_json "
    "FROM s34_trades WHERE status='CLOSED' AND net_bps IS NOT NULL "
    "AND rule_name IN (?,?) ORDER BY entry_ts_ms",
    RULES
).fetchall()
intel.close()

def get_min_price(symbol, ts_start_ms, ts_end_ms):
    r = micro.execute(
        "SELECT MIN(mark_price) FROM mark_prices WHERE symbol=? AND ts_ms BETWEEN ? AND ?",
        (symbol, ts_start_ms, ts_end_ms)
    ).fetchone()
    return r[0] if r else None

trades = []
for tid, rule, entry_ms, net_bps, exit_r, tj in rows:
    try:
        t  = json.loads(tj)
        sig = t.get("signal") or {}
        symbol = t.get("symbol") or ("ETHUSDT" if "ETH" in rule else "SOLUSDT")
        ep     = float(t.get("entry_price") or 0)
        cnt    = sig.get("liq_count")
        cas    = sig.get("liq_total_notional")
        mx     = sig.get("liq_max_notional")
        share  = (mx / cas * 100) if (cas and mx and cas > 0) else None
        hour   = datetime.fromtimestamp(entry_ms / 1000, tz=timezone.utc).hour
        is_sl  = bool(exit_r and "SL" in exit_r)
        is_be  = bool(exit_r and "BE" in exit_r)
        mae_px = get_min_price(symbol, entry_ms, entry_ms + 600_000)
        mae_bps = (mae_px - ep) / ep * 10000 if (mae_px and ep) else None
        trades.append({
            "tid": tid, "rule": rule, "symbol": symbol,
            "entry_ms": entry_ms, "ep": ep,
            "net": float(net_bps), "exit_r": exit_r,
            "is_sl": is_sl, "is_be": is_be,
            "cnt": cnt, "cas": cas, "share": share, "hour": hour,
            "mae_bps": mae_bps,
        })
    except Exception:
        pass

micro.close()

def stats(group, key="net"):
    vals = [t[key] for t in group if t[key] is not None]
    if not vals: return None
    wr  = sum(1 for t in group if t["net"] > 0) / len(group) * 100
    med = sorted(vals)[len(vals) // 2]
    return {"n": len(group), "wr": wr, "med": med, "cum": sum(v for t in group for v in [t["net"]])}

def mae_stats(group):
    vals = [t["mae_bps"] for t in group if t["mae_bps"] is not None]
    if not vals: return None
    s = sorted(vals)
    return {"n": len(vals), "med": s[len(s)//2], "p10": s[max(0,int(len(s)*0.1))]}

N = len(trades)
sl_trades = [t for t in trades if t["is_sl"]]
win_trades = [t for t in trades if not t["is_sl"]]

print("=" * 70)
print(f"SL ANATOMY + LIQ_COUNT CORRELATION  N={N} trades  SL={len(sl_trades)}")
print("=" * 70)

# ── 1. SL trade fingerprint ───────────────────────────────────────────────────
print("\n--- SL TRADE FINGERPRINT ---\n")
print(f"  {'Date':>5}  {'Sym':>3}  {'cnt':>4}  {'Cascade':>9}  {'Share':>6}  {'Hour':>4}  {'MAE':>7}  {'Net':>7}")
print("  " + "-" * 58)
for t in sl_trades:
    dt  = datetime.fromtimestamp(t["entry_ms"]/1000, tz=timezone.utc).strftime("%m/%d")
    sym = "ETH" if "ETH" in t["rule"] else "SOL"
    cnt_s  = str(t["cnt"]) if t["cnt"] is not None else "?"
    cas_s  = f"${t['cas']/1_000_000:.2f}M" if t["cas"] else "?"
    sh_s   = f"{t['share']:.0f}%" if t["share"] else "?"
    mae_s  = f"{t['mae_bps']:+.1f}" if t["mae_bps"] is not None else "?"
    print(f"  {dt:>5}  {sym:>3}  {cnt_s:>4}  {cas_s:>9}  {sh_s:>6}  {t['hour']:>3}h  {mae_s:>7}  {t['net']:>+7.1f}")

print(f"\n  vs WINNER avg liq_count: "
      f"{sum(t['cnt'] for t in win_trades if t['cnt'])/max(1,sum(1 for t in win_trades if t['cnt'])):.1f}")
sl_cnt_avg = sum(t['cnt'] for t in sl_trades if t['cnt']) / max(1, sum(1 for t in sl_trades if t['cnt']))
print(f"  SL trade avg liq_count:  {sl_cnt_avg:.1f}")

# ── 2. liq_count buckets — performance + MAE ──────────────────────────────────
print("\n--- LIQ_COUNT BUCKETS: net bps + MAE ---\n")
print(f"  {'Bucket':>8}  {'N':>3}  {'WR':>5}  {'Med net':>8}  {'Cum':>8}  {'MAE med':>8}  {'MAE p10':>8}  {'SL':>3}")
print("  " + "-" * 66)

buckets = [
    ("<=3",   lambda t: t["cnt"] is not None and t["cnt"] <= 3),
    ("4-7",   lambda t: t["cnt"] is not None and 4 <= t["cnt"] <= 7),
    ("8-12",  lambda t: t["cnt"] is not None and 8 <= t["cnt"] <= 12),
    ("13-20", lambda t: t["cnt"] is not None and 13 <= t["cnt"] <= 20),
    (">20",   lambda t: t["cnt"] is not None and t["cnt"] > 20),
    ("none",  lambda t: t["cnt"] is None),
]
for label, fn in buckets:
    g = [t for t in trades if fn(t)]
    if not g: continue
    s = stats(g)
    m = mae_stats(g)
    sl_n = sum(1 for t in g if t["is_sl"])
    mae_med_s = f"{m['med']:+.1f}" if m else "?"
    mae_p10_s = f"{m['p10']:+.1f}" if m else "?"
    print(f"  {label:>8}  {s['n']:>3}  {s['wr']:>4.0f}%  {s['med']:>+8.1f}  {s['cum']:>+8.1f}  "
          f"{mae_med_s:>8}  {mae_p10_s:>8}  {sl_n:>3}")

# ── 3. Per-rule liq_count breakdown ──────────────────────────────────────────
for rule_short, rule_full in [("ETH_500K", RULES[0]), ("SOL_200K", RULES[1])]:
    rt = [t for t in trades if t["rule"] == rule_full]
    if not rt: continue
    print(f"\n--- {rule_short}: liq_count split ---\n")
    print(f"  {'Bucket':>8}  {'N':>3}  {'WR':>5}  {'Med':>7}  {'Cum':>8}  {'MAE med':>8}  {'SL':>3}")
    print("  " + "-" * 52)

    # Find natural split for this rule
    sym_buckets = [
        ("<=3",  lambda t: t["cnt"] is not None and t["cnt"] <= 3),
        ("4-7",  lambda t: t["cnt"] is not None and 4 <= t["cnt"] <= 7),
        ("8-12", lambda t: t["cnt"] is not None and 8 <= t["cnt"] <= 12),
        (">12",  lambda t: t["cnt"] is not None and t["cnt"] > 12),
        ("none", lambda t: t["cnt"] is None),
    ]
    for label, fn in sym_buckets:
        g = [t for t in rt if fn(t)]
        if not g: continue
        s = stats(g)
        m = mae_stats(g)
        sl_n = sum(1 for t in g if t["is_sl"])
        mae_med_s = f"{m['med']:+.1f}" if m else "?"
        print(f"  {label:>8}  {s['n']:>3}  {s['wr']:>4.0f}%  {s['med']:>+7.1f}  {s['cum']:>+8.1f}  "
              f"{mae_med_s:>8}  {sl_n:>3}")

# ── 4. MAE vs outcome correlation ────────────────────────────────────────────
print("\n--- MAE vs LIQ_COUNT CORRELATION ---\n")
print("  Do high-count trades have smaller MAE (cleaner entries)?")
print()
# High vs low count
high = [t for t in trades if t["cnt"] is not None and t["cnt"] > 10]
low  = [t for t in trades if t["cnt"] is not None and t["cnt"] <= 10]
mh = mae_stats(high)
ml = mae_stats(low)
if mh: print(f"  cnt >10   N={mh['n']}  MAE med={mh['med']:+.1f}  MAE p10={mh['p10']:+.1f}")
if ml: print(f"  cnt <=10  N={ml['n']}  MAE med={ml['med']:+.1f}  MAE p10={ml['p10']:+.1f}")

# ── 5. Hour of day ────────────────────────────────────────────────────────────
print("\n--- HOUR OF DAY (UTC): SL cluster? ---\n")
hour_map = {}
for t in trades:
    h = t["hour"]
    if h not in hour_map:
        hour_map[h] = {"all": 0, "sl": 0}
    hour_map[h]["all"] += 1
    if t["is_sl"]:
        hour_map[h]["sl"] += 1

print(f"  {'Hour':>5}  {'N':>3}  {'SL':>3}  {'SL%':>5}  bar")
for h in sorted(hour_map):
    v  = hour_map[h]
    sl_pct = v["sl"] / v["all"] * 100
    bar = "#" * v["all"] + ("!" * v["sl"])
    print(f"  {h:>4}h  {v['all']:>3}  {v['sl']:>3}  {sl_pct:>4.0f}%  {bar}")

# ── 6. Cascade size vs SL ─────────────────────────────────────────────────────
print("\n--- CASCADE SIZE vs OUTCOME ---\n")
cas_buckets = [
    ("200K-500K", lambda t: t["cas"] and 200_000 <= t["cas"] < 500_000),
    ("500K-1M",   lambda t: t["cas"] and 500_000 <= t["cas"] < 1_000_000),
    ("1M-2M",     lambda t: t["cas"] and 1_000_000 <= t["cas"] < 2_000_000),
    (">2M",       lambda t: t["cas"] and t["cas"] >= 2_000_000),
]
print(f"  {'Bucket':>10}  {'N':>3}  {'WR':>5}  {'Med':>7}  {'SL':>3}  {'MAE med':>8}")
print("  " + "-" * 48)
for label, fn in cas_buckets:
    g = [t for t in trades if fn(t)]
    if not g: continue
    s = stats(g)
    m = mae_stats(g)
    sl_n = sum(1 for t in g if t["is_sl"])
    mae_s = f"{m['med']:+.1f}" if m else "?"
    print(f"  {label:>10}  {s['n']:>3}  {s['wr']:>4.0f}%  {s['med']:>+7.1f}  {sl_n:>3}  {mae_s:>8}")

# ── 7. Composite: high-count + large cascade ──────────────────────────────────
print("\n--- COMPOSITE FILTER: cnt>10 AND cascade>500K ---\n")
strong = [t for t in trades if t["cnt"] is not None and t["cnt"] > 10
          and t["cas"] is not None and t["cas"] >= 500_000]
weak   = [t for t in trades if not (t["cnt"] is not None and t["cnt"] > 10
          and t["cas"] is not None and t["cas"] >= 500_000)]
for label, g in [("STRONG (cnt>10, cas>500K)", strong), ("WEAK (rest)", weak)]:
    if not g: continue
    s = stats(g)
    m = mae_stats(g)
    sl_n = sum(1 for t in g if t["is_sl"])
    mae_s = f"{m['med']:+.1f}" if m else "?"
    print(f"  {label}")
    print(f"    N={s['n']}  WR={s['wr']:.0f}%  Med={s['med']:+.1f}  Cum={s['cum']:+.1f}  SL={sl_n}  MAE={mae_s}")

print()
