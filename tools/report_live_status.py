from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.microphys.live.metrics import load_status


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render live status markdown report.")
    p.add_argument("--live", default="data/live")
    p.add_argument("--out", default="reports/live_status.md")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        live = Path(str(args.live))
        status = load_status(live / "status.json")
        hb = {}
        hb_path = live / "heartbeat.json"
        if hb_path.exists():
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
        out = Path(str(args.out))
        lines = [
            "# Live Status",
            "",
            f"- status_ts: `{status.get('ts_utc', 'n/a')}`",
            f"- state: `{status.get('state', 'unknown')}`",
            f"- reason: `{status.get('reason', 'n/a')}`",
            f"- heartbeat_ts: `{hb.get('ts_utc', 'n/a')}`",
            f"- heartbeat_ok: `{hb.get('ok', False)}`",
            "",
            "## Metrics",
            "",
            f"- data_freshness_sec: `{float(status.get('data_freshness_sec', 0.0) or 0.0):.2f}`",
            f"- missing_bars_pct_1h: `{float(status.get('missing_bars_pct_1h', 0.0) or 0.0):.2f}`",
            f"- spread_median: `{float(status.get('spread_median', 0.0) or 0.0):.8f}`",
            f"- spread_p95: `{float(status.get('spread_p95', 0.0) or 0.0):.8f}`",
            f"- ofi_shift: `{float(status.get('ofi_shift', 0.0) or 0.0):.4f}`",
            f"- regime_shift: `{float(status.get('regime_shift', 0.0) or 0.0):.4f}`",
            f"- signal_rate_per_hour: `{float(status.get('signal_rate_per_hour', 0.0) or 0.0):.4f}`",
            f"- pnl_net_mean: `{float(status.get('pnl_net_mean', 0.0) or 0.0):.8f}`",
            f"- pnl_net_sum: `{float(status.get('pnl_net_sum', 0.0) or 0.0):.8f}`",
            f"- adverse_proxy_rate: `{float(status.get('adverse_proxy_rate', 0.0) or 0.0):.4f}`",
            f"- execution_model: `{status.get('execution_model', 'simple')}`",
            f"- execution_params_loaded: `{bool(status.get('execution_params_loaded', False))}`",
            f"- execution_params_path: `{status.get('execution_params_path', '')}`",
            f"- execution_params_run_id: `{status.get('execution_params_run_id', '')}`",
            f"- active_calibration_path: `{status.get('active_calibration_path', '')}`",
            f"- active_calibration_sha256: `{status.get('active_calibration_sha256', '')}`",
            f"- active_execution_sha256: `{status.get('active_execution_sha256', '')}`",
            f"- active_artifacts_activated_ts: `{status.get('active_artifacts_activated_ts', '')}`",
        ]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"report_live_status ok out={out}")
        return 0
    except Exception as e:
        print(f"report_live_status error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
