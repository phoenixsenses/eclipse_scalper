# encoding: utf-8
import sqlite3, os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("D:/eclipse_scalper")
conn = sqlite3.connect("file:" + str(ROOT / "data" / "s34_intelligence.db") + "?mode=ro", uri=True)

print("=" * 75)
print("SYSTEM SURVEY")
print("=" * 75)

# All rules N count
rows = conn.execute("""
    SELECT rule_name,
           SUM(CASE WHEN status='CLOSED' THEN 1 ELSE 0 END) as closed,
           SUM(CASE WHEN status='OPEN'   THEN 1 ELSE 0 END) as open_n,
           ROUND(AVG(CASE WHEN status='CLOSED' AND net_bps IS NOT NULL THEN net_bps END),1) as mean,
           ROUND(SUM(CASE WHEN status='CLOSED' AND net_bps IS NOT NULL THEN net_bps ELSE 0 END),1) as cum,
           ROUND(AVG(CASE WHEN status='CLOSED' AND net_bps>0 THEN 1.0 ELSE 0.0 END)*100,0) as wr,
           MAX(entry_ts_ms) as last_ts
    FROM s34_trades GROUP BY rule_name ORDER BY closed DESC
""").fetchall()

print("\n--- ALL RULES (paper trades) ---")
print(f"  {'Rule':<58} {'Closed':>6} {'Open':>4} {'WR':>5} {'Mean':>7} {'Cum':>8}")
print("  " + "-" * 92)
for r in rows:
    name, closed, open_n, mean, cum, wr, last_ts = r
    last = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).strftime("%m/%d") if last_ts else "?"
    mean_s = f"{mean:+.1f}" if mean is not None else "  ?"
    cum_s  = f"{cum:+.1f}" if cum is not None else "  ?"
    wr_s   = f"{wr:.0f}%" if wr is not None else "?"
    print(f"  {name:<58} {closed:>6} {open_n:>4} {wr_s:>5} {mean_s:>7} {cum_s:>8}  [{last}]")

# Check what shadow runner is currently running
print("\n--- SHADOW PAPER RUNNER PROCESS ---")
import subprocess
result = subprocess.run(
    ["powershell", "-Command",
     "Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -like '*shadow_paper*'} | Select-Object -ExpandProperty CommandLine"],
    capture_output=True, text=True, timeout=10
)
print(" ", result.stdout.strip()[:200] if result.stdout.strip() else "NOT RUNNING")

# Executor state
state_path = ROOT / "runtime" / "s34_live_executor_state.json"
if state_path.exists():
    import json
    state = json.loads(state_path.read_text())
    st = state.get("status", {})
    print("\n--- LIVE EXECUTOR STATE ---")
    print(f"  Mode:       {st.get('mode')}")
    print(f"  Updated:    {st.get('updated_at_utc')}")
    print(f"  Mirrored:   {len(state.get('mirrored_trade_ids', {}))}")
    print(f"  Candidates: {st.get('candidate_open_count')}")

# Geo tag summary
tags = conn.execute("SELECT tag, COUNT(*) FROM s34_shadow_geometry_tags GROUP BY tag").fetchall()
print("\n--- SHADOW GEOMETRY TAGS ---")
for tag, n in tags:
    print(f"  {tag}: N={n}")

# .env allowed rules
env_path = ROOT / ".env"
print("\n--- .ENV KEY SETTINGS ---")
for line in env_path.read_text(encoding="utf-8").splitlines():
    if any(k in line for k in ["ALLOWED_RULES", "MARGIN", "LEVERAGE", "CLEAN_GEO", "LIVE_TRADING", "DRY_RUN"]):
        if not line.startswith("#") and "=" in line:
            print(f"  {line.strip()}")

conn.close()
print()
