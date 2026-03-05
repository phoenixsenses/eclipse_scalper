from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _pct(v: float) -> str:
    return f"{100.0 * float(v):.2f}%"


def _reason_action(reason_share: Dict[str, float], gate_high_share: float) -> str:
    fees_dom = float(reason_share.get("fees_dominate", 0.0))
    adv_dom = float(reason_share.get("adverse_dominates", 0.0))
    if fees_dom > 0.60:
        return "Next action: fees dominate. Improve fee tier/rebate or increase raw edge."
    if adv_dom > 0.60:
        return "Next action: adverse dominates. Tighten adverse model and event filters."
    if gate_high_share > 0.50:
        return "Next action: gate rejections are high. Consider relaxing volatility/gate thresholds."
    return "Next action: mixed causes. Iterate jointly on costs, adverse filters, and signal quality."


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize failure attribution from rank_passive_pockets_forward JSON.")
    p.add_argument("--in", dest="in_path", required=True, help="Path to rank JSON (contains key 'ranking').")
    p.add_argument("--top-n", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = _args()
    in_path = Path(str(args.in_path))
    if not in_path.exists():
        print(f"missing input: {in_path}")
        return 2
    try:
        payload = json.loads(in_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"invalid json: {exc}")
        return 2

    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        print("invalid payload: missing ranking list")
        return 2
    top_n = max(1, int(args.top_n))
    rows = ranking[:top_n]
    print(f"rows_total={len(ranking)} top_n={len(rows)}")
    print(
        "rank symbol reason gate_reject fill_after_gate fee_bps adverse_bps raw_bps net_bps "
        "pass_core npa_core score_raw_core"
    )
    for i, r in enumerate(rows, start=1):
        print(
            f"{i:>3d} "
            f"{str(r.get('symbol', '')):<8} "
            f"{str(r.get('failure_reason_top', 'mixed')):<16} "
            f"{_pct(_safe_float(r.get('gate_reject_ratio', 0.0))):>8} "
            f"{_pct(_safe_float(r.get('fill_rate_after_gate', 0.0))):>8} "
            f"{_safe_float(r.get('avg_fee_bps', 0.0)):>7.3f} "
            f"{_safe_float(r.get('avg_adverse_bps_on_fills', 0.0)):>8.3f} "
            f"{_safe_float(r.get('avg_raw_return_bps_on_fills', 0.0)):>8.3f} "
            f"{_safe_float(r.get('avg_net_return_bps_on_fills', 0.0)):>8.3f} "
            f"{_pct(_safe_float(r.get('pass_rate_core', 0.0))):>8} "
            f"{_safe_float(r.get('npa_core', 0.0)):>+10.6e} "
            f"{_safe_float(r.get('score_raw_core', 0.0)):>+10.6e}"
        )

    counts = Counter(str(r.get("failure_reason_top", "mixed")) for r in ranking)
    total = len(ranking) if ranking else 1
    reason_share = {k: (v / total) for k, v in counts.items()}
    print("\nFailure Reason Share")
    for k in ["gate_reject", "no_fills", "adverse_dominates", "fees_dominate", "mixed"]:
        v = int(counts.get(k, 0))
        print(f"- {k}: {v}/{total} ({_pct(v/total)})")

    gate_high_share = sum(1 for r in ranking if _safe_float(r.get("gate_reject_ratio", 0.0)) > 0.5) / float(total)
    print()
    print(_reason_action(reason_share, gate_high_share))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

