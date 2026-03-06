from __future__ import annotations

import argparse
from statistics import mean
from pathlib import Path

from tools.micro_edge_lib import extract_best_rule_delta, load_jsonl


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="View rolling micro-edge summary from JSONL.")
    p.add_argument("--in", dest="in_path", default="logs/micro_edge_smoke.jsonl")
    p.add_argument("--symbol", default="")
    p.add_argument("--last", type=int, default=200)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    rows = load_jsonl(Path(str(args.in_path)))
    sym = str(args.symbol or "").strip().upper()
    if sym:
        rows = [r for r in rows if str(r.get("symbol") or "").upper() == sym]
    if args.last > 0 and len(rows) > int(args.last):
        rows = rows[-int(args.last):]
    if not rows:
        print("micro_edge_report: no rows")
        return 0
    baselines = [float(r["baseline_hit_rate"]) for r in rows if r.get("baseline_hit_rate") is not None]
    best_deltas = []
    for r in rows:
        d = extract_best_rule_delta(r)
        if d is None:
            continue
        best_deltas.append(float(d))
    print("micro_edge_report")
    print(f"rows={len(rows)} symbol={(sym if sym else 'ALL')}")
    print(f"baseline_hit_rate_avg={(f'{mean(baselines):.4f}' if baselines else 'n/a')}")
    print(f"best_rule_delta_avg={(f'{mean(best_deltas):+.4f}' if best_deltas else 'n/a')}")
    if rows:
        last = rows[-1]
        print(
            f"latest ts_utc={last.get('ts_utc')} symbol={last.get('symbol')} "
            f"bucket={last.get('bucket_sec')} horizon={last.get('horizon_sec')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
