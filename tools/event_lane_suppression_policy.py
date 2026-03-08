from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.run_summary import build_run_summary


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lane_map(watchboard: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("lane") or ""): row
        for row in (watchboard.get("lanes") or [])
        if isinstance(row, dict) and str(row.get("lane") or "")
    }


def _display_mode(secondary_row: Dict[str, Any], top_lane: str, decision: Dict[str, Any]) -> str:
    if str(secondary_row.get("recommended_action") or "") == "monitor_only":
        return "hide"
    if str(decision.get("lane_a") or "") == top_lane or str(decision.get("lane_b") or "") == top_lane:
        return "degrade"
    return "collapse"


def build_suppression_policy_payload(
    *,
    watchboard_json: str,
    consolidation_json: str,
    out_json: str,
    out_md: str,
) -> Dict[str, Any]:
    watchboard = _load_json(watchboard_json)
    consolidation = _load_json(consolidation_json)
    top_lane = str((watchboard.get("summary") or {}).get("top_lane") or "")
    lanes = _lane_map(watchboard)
    suppression_rules: List[Dict[str, Any]] = []
    for decision in consolidation.get("decisions") or []:
        if not isinstance(decision, dict):
            continue
        if str(decision.get("recommendation") or "") != "candidate_suppress_secondary":
            continue
        secondary_lane = str(decision.get("secondary_lane") or "")
        if not secondary_lane:
            continue
        secondary_row = lanes.get(secondary_lane, {})
        suppression_rules.append(
            {
                "secondary_lane": secondary_lane,
                "when_lane_a": str(decision.get("lane_a") or ""),
                "when_lane_b": str(decision.get("lane_b") or ""),
                "display_mode": _display_mode(secondary_row, top_lane, decision),
                "reason": str(decision.get("reason") or ""),
                "secondary_level": str(secondary_row.get("level") or "quiet"),
                "secondary_action": str(secondary_row.get("recommended_action") or "monitor_only"),
            }
        )
    deduped: Dict[str, Dict[str, Any]] = {}
    order = {"hide": 3, "degrade": 2, "collapse": 1}
    for row in suppression_rules:
        key = str(row.get("secondary_lane") or "")
        current = deduped.get(key)
        if current is None or order.get(str(row.get("display_mode") or ""), 0) > order.get(str(current.get("display_mode") or ""), 0):
            deduped[key] = row
    rules = sorted(deduped.values(), key=lambda row: (order.get(str(row["display_mode"]), 0), row["secondary_lane"]), reverse=True)
    summary = {
        "top_lane": top_lane,
        "rule_count": len(rules),
        "suppressed_lanes": [str(row.get("secondary_lane") or "") for row in rules],
    }
    payload = {
        "watchboard_json": str(watchboard_json),
        "consolidation_json": str(consolidation_json),
        "summary": summary,
        "rules": rules,
    }
    payload["run_summary"] = build_run_summary(
        run_type="event_lane_suppression_policy",
        inputs={"watchboard_json": str(watchboard_json), "consolidation_json": str(consolidation_json)},
        metrics={"rule_count": len(rules), "top_lane": top_lane},
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert consolidation decisions into runtime-facing suppression rules.")
    p.add_argument("--watchboard-json", default="reports/RESEARCH_EVENT_WATCHBOARD.json")
    p.add_argument("--consolidation-json", default="reports/EVENT_LANE_CONSOLIDATION.json")
    p.add_argument("--out-json", default="reports/EVENT_LANE_SUPPRESSION_POLICY.json")
    p.add_argument("--out-md", default="reports/EVENT_LANE_SUPPRESSION_POLICY.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_suppression_policy_payload(
        watchboard_json=str(args.watchboard_json),
        consolidation_json=str(args.consolidation_json),
        out_json=str(args.out_json),
        out_md=str(args.out_md),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# EVENT LANE SUPPRESSION POLICY",
        "",
        f"summary={json.dumps(payload['summary'], ensure_ascii=True, sort_keys=True)}",
        "",
        "| secondary_lane | when_lane_a | when_lane_b | display_mode | secondary_action |",
        "|---|---|---|---|---|",
    ]
    for row in payload["rules"]:
        lines.append(
            f"| {row['secondary_lane']} | {row['when_lane_a']} | {row['when_lane_b']} | {row['display_mode']} | {row['secondary_action']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
