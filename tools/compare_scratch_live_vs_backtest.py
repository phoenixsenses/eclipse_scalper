from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare live scratch behavior vs backtest scratch calibration.")
    p.add_argument("--trade-db", default="data/paper_trades.db")
    p.add_argument("--backtest-sell-json", default="reports/SCRATCH_CALIBRATION_SELL_UP.json")
    p.add_argument("--backtest-buy-json", default="reports/SCRATCH_CALIBRATION_BUY_UP.json")
    p.add_argument("--out-md", default="reports/SCRATCH_LIVE_VS_BACKTEST.md")
    p.add_argument("--out-json", default="reports/SCRATCH_LIVE_VS_BACKTEST.json")
    return p.parse_args()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if x != x:
            return float(default)
        return x
    except Exception:
        return float(default)


def _load_live(db: Path) -> dict[str, Any]:
    if not db.exists():
        return {"rows": 0, "scratch_frac": 0.0, "by_side": {}}
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        rows = conn.execute(
            "SELECT side, exit_reason FROM trades WHERE side IN ('buy','sell','BUY','SELL')"
        ).fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()
    total = len(rows)
    scratch_total = 0
    by_side: dict[str, dict[str, float]] = {}
    for side, reason in rows:
        s = str(side or "").strip().lower()
        r = str(reason or "").strip().lower()
        sc = 1.0 if "scratch" in r else 0.0
        scratch_total += int(sc)
        cur = by_side.setdefault(s, {"rows": 0.0, "scratch": 0.0})
        cur["rows"] += 1.0
        cur["scratch"] += sc
    for s in list(by_side.keys()):
        row_n = max(1.0, by_side[s]["rows"])
        by_side[s]["scratch_frac"] = float(by_side[s]["scratch"] / row_n)
    return {
        "rows": int(total),
        "scratch_frac": float(scratch_total / max(1, total)),
        "by_side": by_side,
    }


def _load_backtest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = payload.get("baseline", {}) if isinstance(payload, dict) else {}
    return {
        "exists": True,
        "n": int(_safe_float(base.get("n"), 0.0)),
        "scratch_frac": float(_safe_float(base.get("scratch_frac"), 0.0)),
        "best_adverse_bps": float(_safe_float((payload.get("best_adverse") or {}).get("max_adverse_bps"), 0.0)),
        "best_trailing_bps": float(_safe_float((payload.get("best_trailing") or {}).get("trailing_stop_bps_proxy"), 0.0)),
    }


def main() -> int:
    args = _parse_args()
    out_md = Path(str(args.out_md))
    out_json = Path(str(args.out_json))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    live = _load_live(Path(str(args.trade_db)))
    bt_sell = _load_backtest(Path(str(args.backtest_sell_json)))
    bt_buy = _load_backtest(Path(str(args.backtest_buy_json)))
    live_sell = float((live.get("by_side", {}).get("sell", {}) or {}).get("scratch_frac", 0.0))
    live_buy = float((live.get("by_side", {}).get("buy", {}) or {}).get("scratch_frac", 0.0))
    delta_sell = abs(live_sell - float(bt_sell.get("scratch_frac", 0.0))) if bt_sell.get("exists") else None
    delta_buy = abs(live_buy - float(bt_buy.get("scratch_frac", 0.0))) if bt_buy.get("exists") else None

    payload = {
        "status": "ok",
        "live": live,
        "backtest_sell": bt_sell,
        "backtest_buy": bt_buy,
        "delta_sell_abs": delta_sell,
        "delta_buy_abs": delta_buy,
        "needs_recalibration_sell": bool(delta_sell is not None and delta_sell > 0.20),
        "needs_recalibration_buy": bool(delta_buy is not None and delta_buy > 0.20),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Scratch Live vs Backtest",
        "",
        f"- live_rows: `{int(live.get('rows', 0))}`",
        f"- live_sell_scratch_frac: `{live_sell:.2%}`",
        f"- live_buy_scratch_frac: `{live_buy:.2%}`",
        "",
        f"- backtest_sell_scratch_frac: `{float(bt_sell.get('scratch_frac', 0.0)):.2%}`",
        f"- backtest_buy_scratch_frac: `{float(bt_buy.get('scratch_frac', 0.0)):.2%}`",
        "",
        f"- delta_sell_abs: `{(float(delta_sell) if delta_sell is not None else 0.0):.2%}`",
        f"- delta_buy_abs: `{(float(delta_buy) if delta_buy is not None else 0.0):.2%}`",
        f"- needs_recalibration_sell: `{int(payload['needs_recalibration_sell'])}`",
        f"- needs_recalibration_buy: `{int(payload['needs_recalibration_buy'])}`",
    ]
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"compare_scratch_live_vs_backtest: wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

