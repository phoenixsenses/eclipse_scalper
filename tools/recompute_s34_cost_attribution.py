from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from tools.s34_shadow_paper_runner import _annotate_trade_pnl_usdt, _cost_decomposition


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _can_recompute(trade: dict[str, Any]) -> tuple[bool, str]:
    if trade.get("status") != "CLOSED":
        return False, "NOT_CLOSED"
    required = ("entry_reference_price", "exit_reference_price", "entry_fill", "exit_fill", "gross_bps", "net_bps")
    missing = [key for key in required if trade.get(key) is None]
    if missing:
        return False, "MISSING_" + ",".join(missing)
    return True, ""


def recompute_trade(trade: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(trade)
    entry_fill = out["entry_fill"]
    exit_fill = out["exit_fill"]
    cost = _cost_decomposition(
        str(out.get("direction") or "LONG"),
        float(out["entry_reference_price"]),
        float(out["exit_reference_price"]),
        entry_fill,
        exit_fill,
        float(entry_fill.get("fee_bps") or 0.0),
        float(exit_fill.get("fee_bps") or 0.0),
    )
    original_net = float(out["net_bps"])
    if abs(float(cost["net_bps"]) - original_net) > 1e-6:
        raise RuntimeError(
            f"net_changed trade_id={out.get('trade_id')} original={original_net} recomputed={cost['net_bps']}"
        )
    out.update(cost)
    _annotate_trade_pnl_usdt(out)
    return out


def recompute(trades_json: Path, *, write: bool) -> dict[str, Any]:
    payload = _read_json(trades_json)
    trades = payload.get("trades", []) if isinstance(payload, dict) else []
    updated: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for trade in trades:
        ok, reason = _can_recompute(trade)
        if ok:
            recomputed = recompute_trade(trade)
            rows.append(
                {
                    "trade_id": recomputed.get("trade_id"),
                    "status": "RECOMPUTED",
                    "reason": "",
                    "gross_bps": recomputed.get("gross_bps"),
                    "entry_adverse_bps": recomputed.get("entry_adverse_bps"),
                    "exit_adverse_bps": recomputed.get("exit_adverse_bps"),
                    "mark_to_fill_cost_bps": recomputed.get("mark_to_fill_cost_bps"),
                    "spread_cost_bps": recomputed.get("spread_cost_bps"),
                    "fee_cost_bps": recomputed.get("fee_cost_bps"),
                    "net_bps": recomputed.get("net_bps"),
                    "net_unchanged": abs(float(recomputed["net_bps"]) - float(trade["net_bps"])) <= 1e-6,
                }
            )
            updated.append(recomputed)
        else:
            rows.append(
                {
                    "trade_id": trade.get("trade_id"),
                    "status": "SKIPPED",
                    "reason": reason,
                    "gross_bps": trade.get("gross_bps"),
                    "entry_adverse_bps": trade.get("entry_adverse_bps"),
                    "exit_adverse_bps": trade.get("exit_adverse_bps"),
                    "mark_to_fill_cost_bps": trade.get("mark_to_fill_cost_bps"),
                    "spread_cost_bps": trade.get("spread_cost_bps"),
                    "fee_cost_bps": trade.get("fee_cost_bps"),
                    "net_bps": trade.get("net_bps"),
                    "net_unchanged": True,
                }
            )
            updated.append(trade)
    result = {
        "trades_total": len(trades),
        "closed_recomputed": sum(1 for row in rows if row["status"] == "RECOMPUTED"),
        "closed_not_recomputable": sum(1 for row in rows if row["status"] == "SKIPPED" and row["reason"].startswith("MISSING_")),
        "net_changed_count": sum(1 for row in rows if not row["net_unchanged"]),
        "rows": rows,
    }
    if write:
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["trades"] = updated
            payload["cost_attribution_recomputed"] = True
        else:
            payload = updated
        _write_json(trades_json, payload)
    return result


def write_report(result: dict[str, Any], out_json: Path, out_md: Path) -> None:
    _write_json(out_json, result)
    lines = [
        "# S34 Cost Attribution Recompute",
        "",
        f"- trades_total: `{result['trades_total']}`",
        f"- closed_recomputed: `{result['closed_recomputed']}`",
        f"- closed_not_recomputable: `{result['closed_not_recomputable']}`",
        f"- net_changed_count: `{result['net_changed_count']}`",
        "",
        "| trade_id | status | reason | gross | entry adverse | exit adverse | mark-to-fill | spread | fee | net |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        if row["status"] != "RECOMPUTED" and row["reason"] == "NOT_CLOSED":
            continue
        def fmt(value: Any) -> str:
            return "" if value is None else f"{float(value):.4f}"
        lines.append(
            "| `{trade_id}` | `{status}` | `{reason}` | {gross} | {entry_adv} | {exit_adv} | {mtf} | {spread} | {fee} | {net} |".format(
                trade_id=row.get("trade_id") or "",
                status=row["status"],
                reason=row["reason"],
                gross=fmt(row.get("gross_bps")),
                entry_adv=fmt(row.get("entry_adverse_bps")),
                exit_adv=fmt(row.get("exit_adverse_bps")),
                mtf=fmt(row.get("mark_to_fill_cost_bps")),
                spread=fmt(row.get("spread_cost_bps")),
                fee=fmt(row.get("fee_cost_bps")),
                net=fmt(row.get("net_bps")),
            )
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute S34 cost attribution from stored reference/fill prices.")
    parser.add_argument("--trades-json", default="reports/research/s34/S34_SHADOW_PAPER_TRADES.json")
    parser.add_argument("--out-json", default="reports/research/s34/S34_COST_ATTRIBUTION_RECOMPUTE.json")
    parser.add_argument("--out-md", default="reports/research/s34/S34_COST_ATTRIBUTION_RECOMPUTE.md")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = recompute(Path(args.trades_json), write=bool(args.write))
    write_report(result, Path(args.out_json), Path(args.out_md))
    print(json.dumps({k: result[k] for k in ("trades_total", "closed_recomputed", "closed_not_recomputable", "net_changed_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
