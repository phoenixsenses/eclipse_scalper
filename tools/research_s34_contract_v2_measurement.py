import json
import sqlite3
import sys

sys.path.insert(0, r"D:\eclipse_scalper")
from ami.geometry import birth_truncated_cascade_geometry as geo
from ami.geometry import birth_truncated_geometry_rehearsal as rehearsal
from ami.geometry import liquidation_source_quality_contract_v2 as v2
from tools.research_s34_knowable_anchor_continuation import reconstruct_anchors

CANON = r"D:\eclipse_scalper\data\ami\canonical.sqlite"
MICRO = r"D:\eclipse_scalper\data\microstructure.db"

conn_c = sqlite3.connect(f"file:{CANON}?mode=ro", uri=True)
_cols = ["signal_id", "setup_id", "source_event_id", "independent_cycle_id",
         "symbol", "direction", "signal_birth_ts"]
signals = [dict(zip(_cols, r)) for r in conn_c.execute(
    f"SELECT {', '.join(_cols)} FROM ami_signal_lifecycle WHERE symbol='ETHUSDT' AND direction='LONG'"
).fetchall()]
event_ids = {s["source_event_id"] for s in signals}
events_by_id = rehearsal.fetch_events_by_id(conn_c, event_ids)
conn_c.close()
assert len(signals) == 220

conn_m = sqlite3.connect(f"file:{MICRO}?mode=ro", uri=True)
all_sell_liqs = rehearsal.fetch_all_sell_liqs(conn_m)
all_anchor_ts = sorted({int(r["anchor_ts_ms"]) for r in events_by_id.values()})
earliest_liq_ts_ms = all_sell_liqs[0]["ts_ms"]
resolved_gaps = conn_m.execute(
    "SELECT start_ts_ms, end_ts_ms FROM gaps WHERE stream='liquidations' AND end_ts_ms IS NOT NULL"
).fetchall()
all_market_liq_ts = [
    r[0] for r in conn_m.execute(
        "SELECT ts_ms FROM liquidations WHERE ts_ms >= ? ORDER BY ts_ms",
        (v2.ALL_MARKET_TRANSITION_TS_MS - v2.CRITICAL_GAP_MS,),
    ).fetchall()
]
conn_m.close()

field_counts = {f: {} for f in geo._FEATURE_FIELDS}
row_counts = {}
limiting_field_counts = {}
per_signal = []

for s in signals:
    ev = events_by_id[s["source_event_id"]]
    anchor_ts = int(ev["anchor_ts_ms"])
    geo_row = geo.reconstruct_signal_geometry(
        all_sell_liqs, anchor_ts, int(s["signal_birth_ts"]), reconstruct_anchors_fn=reconstruct_anchors)
    assert geo_row is not None
    pos = all_anchor_ts.index(anchor_ts)
    prev_anchor_ts_ms = all_anchor_ts[pos - 1] if pos > 0 else None
    fields = v2.classify_signal_fields(
        bucket_start_ts_ms=geo_row["source_window_start_ts_ms"], anchor_ts_ms=anchor_ts,
        prev_anchor_ts_ms=prev_anchor_ts_ms, earliest_liq_ts_ms=earliest_liq_ts_ms,
        resolved_gaps=resolved_gaps, sorted_all_market_liq_ts=all_market_liq_ts,
    )
    for f, d in fields.items():
        field_counts[f][d["status"]] = field_counts[f].get(d["status"], 0) + 1
    row_status = v2.row_level_worst_case({f: d["status"] for f, d in fields.items()})
    row_counts[row_status] = row_counts.get(row_status, 0) + 1

    # which field(s) are the "worst" (limiting) one(s) for this row
    worst = max(d["status"] for d in fields.values())
    order = v2._ROW_WORST_CASE_ORDER
    worst_rank = max(order[d["status"]] for d in fields.values())
    limiters = [f for f, d in fields.items() if order[d["status"]] == worst_rank]
    key = tuple(sorted(limiters)) if row_status != "SOURCE_COMPLETE" else ("<all fields complete>",)
    limiting_field_counts[key] = limiting_field_counts.get(key, 0) + 1

    per_signal.append({
        "signal_id": s["signal_id"], "month": s["signal_birth_ts"],
        "row_status": row_status, "independent_cycle_id": s["independent_cycle_id"],
        "fields": {f: d["status"] for f, d in fields.items()},
    })

print("=== per-field status counts ===")
for f, counts in field_counts.items():
    print(f"  {f}: {dict(sorted(counts.items()))}")

print()
print("=== row-level worst-case status counts ===")
print(" ", dict(sorted(row_counts.items())))

print()
print("=== limiting-field combinations (rows NOT fully complete) ===")
for k, n in sorted(limiting_field_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {n}")

sig_by_id = {s["signal_id"]: s for s in signals}
complete_signals = [sig_by_id[r["signal_id"]] for r in per_signal if r["row_status"] == "SOURCE_COMPLETE"]
rep = rehearsal.compute_population_report(complete_signals)
print()
print("=== CONTRACT-V2 SOURCE_COMPLETE_ONLY population (row-level worst-case) ===")
print(json.dumps(rep, indent=2, default=str))

import datetime as dt
def month_of(ts):
    d = dt.datetime.fromtimestamp(ts/1000, dt.timezone.utc)
    return f"{d.year:04d}-{d.month:02d}"
for r in per_signal:
    r["month"] = month_of(r["month"])

out = {
    "field_counts": field_counts, "row_counts": row_counts,
    "complete_only_population_report": rep,
    "per_signal": per_signal,
}
scratch = r"C:\Users\WINDOW~1\AppData\Local\Temp\claude\D--eclipse-scalper\0e02bf95-0aa6-485e-ba92-2885d5e532d3\scratchpad"
with open(scratch + r"\contract_v2_full_report.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, default=str)
print()
print("WROTE", scratch + r"\contract_v2_full_report.json")
