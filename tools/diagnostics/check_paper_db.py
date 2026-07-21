import sqlite3
from datetime import datetime, timezone

conn = sqlite3.connect("data/paper_trades.db")
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tablolar:", tables)
for t in tables:
    n = conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
    print(f"  {t}: {n} satir")
    if n > 0:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info([{t}])").fetchall()]
        ts_col = next((c for c in cols if "ts" in c.lower() or "time" in c.lower()), None)
        if ts_col:
            row = conn.execute(f"SELECT MAX([{ts_col}]) FROM [{t}]").fetchone()[0]
            if row and row > 1e10:
                dt = datetime.fromtimestamp(row / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                print(f"    Son {ts_col}: {dt}")
        # Son 3 satir
        last = conn.execute(f"SELECT * FROM [{t}] ORDER BY rowid DESC LIMIT 3").fetchall()
        for row in last:
            print(f"    {row}")
conn.close()
