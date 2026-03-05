from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render risk diagnostics from live outputs.")
    p.add_argument("--live-root", default="data/live")
    p.add_argument("--out", default="reports/risk_diagnostics.md")
    return p.parse_args()


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main() -> int:
    args = _parse_args()
    try:
        live = Path(str(args.live_root))
        positions = _safe_read_parquet(live / "positions_live.parquet")
        trades = _safe_read_parquet(live / "papertrades_live.parquet")
        risk_events = live.parent / "logs" / "risk_events.jsonl"
        event_counts = {}
        if risk_events.exists():
            rows = [json.loads(x) for x in risk_events.read_text(encoding="utf-8").splitlines() if x.strip()]
            event_counts = pd.Series([str(r.get("event", "")) for r in rows]).value_counts().to_dict() if rows else {}
        size_hist = pd.to_numeric(trades.get("trade_notional"), errors="coerce").dropna() if not trades.empty else pd.Series([], dtype=float)
        lines = [
            "# Risk Diagnostics",
            "",
            f"- positions_rows: `{len(positions)}`",
            f"- trades_rows: `{len(trades)}`",
            f"- risk_events: `{sum(int(v) for v in event_counts.values())}`",
            "",
            f"- notional_mean: `{float(size_hist.mean()) if not size_hist.empty else 0.0:.4f}`",
            f"- notional_p95: `{float(size_hist.quantile(0.95)) if not size_hist.empty else 0.0:.4f}`",
            "",
            "## Skip/Kill Events",
            "",
            "| event | count |",
            "|---|---:|",
        ]
        for k in sorted(event_counts):
            lines.append(f"| {k} | {int(event_counts[k])} |")
        out = Path(str(args.out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"report_risk_diagnostics ok out={out}")
        return 0
    except Exception as e:
        print(f"report_risk_diagnostics error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

