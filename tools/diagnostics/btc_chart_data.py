import sqlite3
conn = sqlite3.connect("data/microstructure.db")
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("tables:", tables)
r1 = conn.execute("SELECT MIN(ts_ms), MAX(ts_ms), COUNT(*) FROM book_ticker WHERE symbol='BTCUSDT'").fetchone()
print("BTC book_ticker: min=%s max=%s n=%s" % r1)
r2 = conn.execute("SELECT MIN(ts_ms), MAX(ts_ms), COUNT(*) FROM liquidations WHERE symbol='BTCUSDT'").fetchone()
print("BTC liquidations: min=%s max=%s n=%s" % r2)
conn.close()
