from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze fill timing and adverse-selection proxy from paper/live trades.")
    p.add_argument("--live-parquet", default="data/live/papertrades_live.parquet")
    p.add_argument("--trade-db", default="data/paper_trades.db")
    p.add_argument("--out-md", default="reports/FILL_TIMING_ANALYSIS.md")
    p.add_argument("--out-json", default="reports/FILL_TIMING_ANALYSIS.json")
    p.add_argument("--last-n", type=int, default=20000)
    return p.parse_args()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if x != x:
            return float(default)
        return x
    except Exception:
        return float(default)


def _load_live(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    if "entry_ts_utc" in df.columns:
        ts = pd.to_datetime(df["entry_ts_utc"], utc=True, errors="coerce")
        df = df.assign(_ts=ts).sort_values("_ts").drop(columns=["_ts"])
    return df.reset_index(drop=True)


def _load_trade_db(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        rows = pd.read_sql_query(
            "SELECT entry_time, exit_time, side, pnl_bps, max_adverse_bps, elapsed_sec, exit_reason FROM trades ORDER BY entry_time ASC",
            conn,
        )
    except Exception:
        rows = pd.DataFrame()
    finally:
        conn.close()
    return rows


def _bucket_fill_delay(delay_sec: float) -> str:
    if delay_sec <= 5.0:
        return "<=5s"
    if delay_sec <= 10.0:
        return "5-10s"
    if delay_sec <= 30.0:
        return "10-30s"
    return ">30s"


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "buckets": {}, "notes": "no_rows"}
    d = df.copy()
    delay_sec = pd.to_numeric(d.get("fill_delay_bars"), errors="coerce").fillna(0.0)
    # In live output delay is in bars; use 1s as conservative conversion if interval unknown.
    delay_sec = delay_sec.astype(float)
    d["_fill_delay_sec"] = delay_sec
    d["_fill_bucket"] = d["_fill_delay_sec"].map(_bucket_fill_delay)
    d["_filled"] = pd.to_numeric(d.get("filled"), errors="coerce").fillna(1).astype(int)
    d["_ttl_expired"] = pd.to_numeric(d.get("ttl_expired"), errors="coerce").fillna(0).astype(int)
    d["_pnl"] = pd.to_numeric(d.get("pnl_net_notional", d.get("pnl_net")), errors="coerce").fillna(0.0)
    d["_adverse_proxy"] = (-d["_pnl"]).clip(lower=0.0)

    out: dict[str, Any] = {"rows": int(len(d)), "buckets": {}}
    by = d.groupby("_fill_bucket", dropna=False)
    for k, g in by:
        out["buckets"][str(k)] = {
            "rows": int(len(g)),
            "frac": float(len(g) / max(1, len(d))),
            "fill_rate": float(g["_filled"].mean()),
            "ttl_expired_rate": float(g["_ttl_expired"].mean()),
            "pnl_mean": float(g["_pnl"].mean()),
            "adverse_proxy_mean": float(g["_adverse_proxy"].mean()),
        }
    out["within_5s_frac"] = float((d["_fill_delay_sec"] <= 5.0).mean())
    out["within_10s_frac"] = float((d["_fill_delay_sec"] <= 10.0).mean())
    out["within_30s_frac"] = float((d["_fill_delay_sec"] <= 30.0).mean())
    return out


def _summary_from_trade_db(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "notes": "no_rows"}
    d = df.copy()
    d["_delay"] = pd.to_numeric(d.get("elapsed_sec"), errors="coerce").fillna(0.0)
    d["_bucket"] = d["_delay"].map(_bucket_fill_delay)
    d["_pnl"] = pd.to_numeric(d.get("pnl_bps"), errors="coerce").fillna(0.0)
    d["_adverse"] = pd.to_numeric(d.get("max_adverse_bps"), errors="coerce").fillna(0.0)
    out: dict[str, Any] = {"rows": int(len(d)), "buckets": {}}
    for k, g in d.groupby("_bucket", dropna=False):
        out["buckets"][str(k)] = {
            "rows": int(len(g)),
            "frac": float(len(g) / max(1, len(d))),
            "pnl_bps_mean": float(g["_pnl"].mean()),
            "max_adverse_bps_mean": float(g["_adverse"].mean()),
        }
    return out


def main() -> int:
    args = _parse_args()
    out_md = Path(str(args.out_md))
    out_json = Path(str(args.out_json))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    live_df = _load_live(Path(str(args.live_parquet)))
    if int(args.last_n) > 0 and len(live_df) > int(args.last_n):
        live_df = live_df.tail(int(args.last_n)).reset_index(drop=True)
    db_df = _load_trade_db(Path(str(args.trade_db)))
    if int(args.last_n) > 0 and len(db_df) > int(args.last_n):
        db_df = db_df.tail(int(args.last_n)).reset_index(drop=True)

    payload = {
        "status": "ok",
        "live_parquet": str(args.live_parquet),
        "trade_db": str(args.trade_db),
        "live_summary": _summary(live_df),
        "trade_db_summary": _summary_from_trade_db(db_df),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Fill Timing Analysis",
        "",
        f"- live_rows: `{int(payload['live_summary'].get('rows', 0))}`",
        f"- trade_db_rows: `{int(payload['trade_db_summary'].get('rows', 0))}`",
        "",
    ]
    ls = payload["live_summary"]
    if ls.get("rows", 0):
        lines.extend(
            [
                "## Live (papertrades_live.parquet)",
                "",
                f"- <=5s: `{float(ls.get('within_5s_frac', 0.0)):.2%}`",
                f"- <=10s: `{float(ls.get('within_10s_frac', 0.0)):.2%}`",
                f"- <=30s: `{float(ls.get('within_30s_frac', 0.0)):.2%}`",
                "",
                "| bucket | rows | frac | fill_rate | ttl_expired | pnl_mean | adverse_proxy_mean |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for k in sorted(ls.get("buckets", {}).keys()):
            r = ls["buckets"][k]
            lines.append(
                f"| {k} | {int(r['rows'])} | {float(r['frac']):.2%} | {float(r['fill_rate']):.2%} | "
                f"{float(r['ttl_expired_rate']):.2%} | {float(r['pnl_mean']):+.8f} | {float(r['adverse_proxy_mean']):.8f} |"
            )
        lines.append("")

    ds = payload["trade_db_summary"]
    if ds.get("rows", 0):
        lines.extend(
            [
                "## Trade DB (pnl/elapsed)",
                "",
                "| bucket | rows | frac | pnl_bps_mean | max_adverse_bps_mean |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for k in sorted(ds.get("buckets", {}).keys()):
            r = ds["buckets"][k]
            lines.append(
                f"| {k} | {int(r['rows'])} | {float(r['frac']):.2%} | {float(r['pnl_bps_mean']):+.6f} | {float(r['max_adverse_bps_mean']):.6f} |"
            )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"analyze_fill_timing: wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

