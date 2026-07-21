import json, sqlite3, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
trades_path = ROOT / "reports" / "research" / "s34" / "S34_SHADOW_PAPER_TRADES.json"
state_path  = ROOT / "runtime" / "s34_live_executor_state.json"

data   = json.loads(trades_path.read_text())
trades = data.get("trades", [])
open_trades = [t for t in trades if t.get("status") == "OPEN"]

print("=== LIVE EXECUTOR STATUS ===")
state = json.loads(state_path.read_text()) if state_path.exists() else {}
st = state.get("status", {})
print("mode           :", st.get("mode"))
print("allowed_rules  :", st.get("allowed_rules"))
print("candidates     :", st.get("candidate_open_count"))
print("updated_at     :", st.get("updated_at_utc"))
mirrored = set(state.get("mirrored_trade_ids", {}).keys())
print("mirrored trades:", len(mirrored), list(mirrored)[:5])

print()
print("=== PAPER TRADES ===")
print("Total trades   :", len(trades))
print("OPEN trades    :", len(open_trades))
for t in open_trades:
    rule = t.get("rule") or {}
    rn = rule.get("name") if isinstance(rule, dict) else str(rule)
    print("  [" + str(t.get("trade_id")) + "] " + str(rn))

print()
print("=== STREAM FRESHNESS ===")
conn = sqlite3.connect("file:" + str(ROOT / "data" / "microstructure.db") + "?mode=ro", uri=True)
last_liq = conn.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0]
last_book = conn.execute("SELECT MAX(ts_ms) FROM book_ticker").fetchone()[0]
conn.close()
now_ms = time.time() * 1000
liq_age  = (now_ms - last_liq)  / 1000 if last_liq  else 9999
book_age = (now_ms - last_book) / 1000 if last_book else 9999
print("Liq  stream    :", round(liq_age,  1), "s ago", "FRESH" if liq_age  < 120 else "STALE")
print("Book stream    :", round(book_age, 1), "s ago", "FRESH" if book_age <  10 else "STALE")
