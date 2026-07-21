"""S34 navigation meta-pattern scan.

Research-only:
- Scale-reversal patterns across KNN neighborhood sizes.
- Whether k20 DANGER has a second-order reversal or is just an avoid zone.
- Tag/classification patterns that work in normal or reverse direction.

No live executor, order logic, size, leverage, or .env changes.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_full_followup import (  # noqa: E402
    DEFAULT_DB,
    FEE_BPS,
    NAV_EVENTS,
    horizon_net,
    knn_cards,
    load_jsonl,
    r1,
    r3,
    summary,
)

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_META_PATTERNS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_META_PATTERNS.md"
ILLEGAL_PRE_ENTRY_TAGS = {
    "TAIL_REALIZED",
    "EXIT_2H_ACTUAL_BETTER",
    "EXIT_4H_ACTUAL_BETTER",
    "EXIT_4H_FAVORED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def key_for(row: dict[str, Any]) -> str:
    return str(row.get("event_id") or f"{row.get('symbol')}_{row.get('liq_side')}_{row.get('threshold_usd')}_{row.get('signal_ts_ms')}")


def summarize_rows(rows: list[dict[str, Any]], value_key: str) -> dict[str, Any]:
    return summary([float(r[value_key]) for r in rows if r.get(value_key) is not None and math.isfinite(float(r[value_key]))])


def card_maps(rows: list[dict[str, Any]], ks: list[int]) -> dict[int, dict[str, dict[str, Any]]]:
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for k in ks:
        cards = knn_cards(rows, k=k, strictness="base")
        out[k] = {key_for(c["row"]): c for c in cards}
    return out


def attach_predictions(rows: list[dict[str, Any]], maps: dict[int, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        key = key_for(row)
        item = dict(row)
        preds = {}
        for k, cmap in maps.items():
            c = cmap.get(key)
            if not c:
                preds[f"k{k}"] = "UNKNOWN"
                continue
            preds[f"k{k}"] = c["prediction"]
            if k == 20:
                item["normal_2h_bps"] = float(c["actual_bps"])
                item["reverse_2h_bps"] = float(c["reverse_bps"])
        item["preds"] = preds
        if "normal_2h_bps" not in item and item.get("net_2h_bps") is not None:
            normal = float(item["net_2h_bps"])
            item["normal_2h_bps"] = normal
            item["reverse_2h_bps"] = -normal - 2.0 * FEE_BPS
        out.append(item)
    return out


def scale_transition_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        p = row["preds"]
        k5 = p.get("k5")
        k8 = p.get("k8")
        k10 = p.get("k10")
        k20 = p.get("k20")
        danger_small = sum(1 for k in ("k5", "k8", "k10") if p.get(k) == "DANGER")
        clean_small = sum(1 for k in ("k5", "k8", "k10") if p.get(k) == "CLEAN")
        if k5 == "DANGER" and k20 == "DANGER":
            label = "PERSISTENT_DANGER_k5_to_k20"
        elif k5 == "DANGER" and k20 != "DANGER":
            label = "LOCAL_DANGER_ONLY_k5"
        elif k5 != "DANGER" and k20 == "DANGER":
            label = "EMERGENT_BROAD_DANGER_k20"
        elif k5 == "CLEAN" and k20 == "CLEAN":
            label = "PERSISTENT_CLEAN_k5_to_k20"
        elif k5 == "CLEAN" and k20 == "DANGER":
            label = "CLEAN_TO_DANGER"
        elif k5 == "DANGER" and k20 == "CLEAN":
            label = "DANGER_TO_CLEAN"
        elif danger_small >= 2 and k20 != "DANGER":
            label = "SMALL_SCALE_DANGER_RELEASES"
        elif clean_small >= 2 and k20 != "CLEAN":
            label = "SMALL_SCALE_CLEAN_DECAYS"
        else:
            label = "NO_CLEAR_SCALE_PATTERN"
        groups[label].append(row)

    result = {}
    for label, items in sorted(groups.items()):
        result[label] = {
            "n": len(items),
            "normal_2h": summarize_rows(items, "normal_2h_bps"),
            "reverse_2h": summarize_rows(items, "reverse_2h_bps"),
            "threshold_mix": threshold_mix(items),
            "tag_mix": top_tags(items, 8),
        }
    return result


def danger_count_bins(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ks = ("k5", "k8", "k10", "k12", "k15", "k20")
    for row in rows:
        count = sum(1 for k in ks if row["preds"].get(k) == "DANGER")
        groups[f"danger_count_{count}"].append(row)
    return {
        label: {
            "n": len(items),
            "normal_2h": summarize_rows(items, "normal_2h_bps"),
            "reverse_2h": summarize_rows(items, "reverse_2h_bps"),
            "threshold_mix": threshold_mix(items),
        }
        for label, items in sorted(groups.items(), key=lambda kv: int(kv[0].split("_")[-1]))
    }


def k20_second_order(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    danger = [r for r in rows if r["preds"].get("k20") == "DANGER"]
    out: dict[str, Any] = {"all_k20_danger": {"n": len(danger)}}
    horizons = {"30m": 1800, "1h": 3600, "2h": 7200, "4h": 14_400}
    for label, sec in horizons.items():
        normal_vals = []
        reverse_vals = []
        by_threshold_normal: dict[str, list[float]] = defaultdict(list)
        by_threshold_reverse: dict[str, list[float]] = defaultdict(list)
        by_small_state_reverse: dict[str, list[float]] = defaultdict(list)
        for row in danger:
            normal, reverse = horizon_net(conn, row, sec)
            if normal is None or reverse is None:
                continue
            normal_vals.append(normal)
            reverse_vals.append(reverse)
            thr = f"thr{int(float(row.get('threshold_usd') or 0))}"
            by_threshold_normal[thr].append(normal)
            by_threshold_reverse[thr].append(reverse)
            small = f"k5_{row['preds'].get('k5')}"
            by_small_state_reverse[small].append(reverse)
        out[label] = {
            "normal": summary(normal_vals),
            "reverse": summary(reverse_vals),
            "reverse_of_reverse_interpretation": "normal_direction",
            "by_threshold_normal": {k: summary(v) for k, v in sorted(by_threshold_normal.items())},
            "by_threshold_reverse": {k: summary(v) for k, v in sorted(by_threshold_reverse.items())},
            "reverse_by_k5_state": {k: summary(v) for k, v in sorted(by_small_state_reverse.items())},
        }
    return out


def threshold_mix(rows: list[dict[str, Any]]) -> dict[str, int]:
    mix: dict[str, int] = defaultdict(int)
    for row in rows:
        mix[f"thr{int(float(row.get('threshold_usd') or 0))}"] += 1
    return dict(sorted(mix.items()))


def top_tags(rows: list[dict[str, Any]], n: int) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for tag in row.get("tags") or []:
            counts[str(tag)] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n])


def combo_key(tags: tuple[str, ...]) -> str:
    return "+".join(tags)


def tag_combo_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tags = sorted({str(t) for r in rows for t in (r.get("tags") or []) if str(t) not in ILLEGAL_PRE_ENTRY_TAGS})
    combos: list[tuple[str, ...]] = []
    for size in (1, 2, 3):
        combos.extend(combinations(tags, size))

    candidates = []
    for combo in combos:
        items = [
            r for r in rows
            if set(combo).issubset(set(t for t in (r.get("tags") or []) if str(t) not in ILLEGAL_PRE_ENTRY_TAGS))
        ]
        if len(items) < 40:
            continue
        normal = summarize_rows(items, "normal_2h_bps")
        reverse = summarize_rows(items, "reverse_2h_bps")
        for direction, s in (("NORMAL", normal), ("REVERSE", reverse)):
            t3r = float(s.get("t3r_bps") or 0.0)
            total = float(s.get("sum_bps") or 0.0)
            median_bps = s.get("median_bps")
            tails = int(s.get("tail_lte_minus150_n") or 0)
            if total > 0 and t3r > 0 and median_bps is not None:
                candidates.append(
                    {
                        "combo": combo_key(combo),
                        "direction": direction,
                        "n": len(items),
                        "summary": s,
                        "tail_rate_150": r3(tails / len(items)),
                        "score": r1(t3r - 50.0 * tails),
                    }
                )
    candidates.sort(key=lambda x: (float(x["score"] or -1e9), float(x["summary"].get("t3r_bps") or -1e9)), reverse=True)
    return {
        "top_candidates": candidates[:25],
        "candidate_count": len(candidates),
        "excluded_non_knowable_tags": sorted(ILLEGAL_PRE_ENTRY_TAGS),
    }


def prediction_combo_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        preds = row["preds"]
        keys = [
            f"k5={preds.get('k5')}",
            f"k8={preds.get('k8')}",
            f"k10={preds.get('k10')}",
            f"k20={preds.get('k20')}",
            f"k5_to_k20={preds.get('k5')}->{preds.get('k20')}",
        ]
        for key in keys:
            groups[key].append(row)

    rows_out = []
    for key, items in groups.items():
        if len(items) < 40:
            continue
        normal = summarize_rows(items, "normal_2h_bps")
        reverse = summarize_rows(items, "reverse_2h_bps")
        rows_out.append({"pattern": key, "n": len(items), "normal_2h": normal, "reverse_2h": reverse})
    rows_out.sort(key=lambda r: max(float(r["normal_2h"].get("t3r_bps") or -1e9), float(r["reverse_2h"].get("t3r_bps") or -1e9)), reverse=True)
    return {"patterns": rows_out}


def format_summary(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Navigation Meta-Patterns",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        "## k20 DANGER: second-order reversal?",
        "",
        "Interpretation: if reverse fails at k20, the reverse-of-reverse is simply the normal direction. Both are checked below.",
        "",
    ]
    k20 = result["k20_second_order"]
    for horizon in ("30m", "1h", "2h", "4h"):
        cell = k20[horizon]
        lines.append(f"- `{horizon}` normal: {format_summary(cell['normal'])}")
        lines.append(f"- `{horizon}` reverse: {format_summary(cell['reverse'])}")
    lines.extend(["", "### k20 2h by threshold", ""])
    cell2 = k20["2h"]
    lines.append("| Threshold | Normal | Reverse |")
    lines.append("| --- | --- | --- |")
    for thr in sorted(cell2["by_threshold_normal"]):
        lines.append(
            f"| {thr} | {format_summary(cell2['by_threshold_normal'][thr])} | "
            f"{format_summary(cell2['by_threshold_reverse'][thr])} |"
        )

    lines.extend(["", "## Scale Transition Groups", ""])
    lines.append("| Group | N | Normal 2h | Reverse 2h | Threshold mix |")
    lines.append("| --- | ---: | --- | --- | --- |")
    for group, cell in sorted(result["scale_transition_groups"].items()):
        lines.append(
            f"| {group} | {cell['n']} | {format_summary(cell['normal_2h'])} | "
            f"{format_summary(cell['reverse_2h'])} | `{cell['threshold_mix']}` |"
        )

    lines.extend(["", "## Danger Count Bins", ""])
    lines.append("| Bin | N | Normal 2h | Reverse 2h |")
    lines.append("| --- | ---: | --- | --- |")
    for label, cell in result["danger_count_bins"].items():
        lines.append(
            f"| {label} | {cell['n']} | {format_summary(cell['normal_2h'])} | "
            f"{format_summary(cell['reverse_2h'])} |"
        )

    lines.extend(["", "## Top Tag-Combo Candidates", ""])
    lines.append(f"Excluded non-knowable/outcome tags: `{result['tag_combo_scan']['excluded_non_knowable_tags']}`")
    lines.append("")
    lines.append("| Combo | Direction | N | Summary | Tail150 rate | Score |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: |")
    for row in result["tag_combo_scan"]["top_candidates"][:15]:
        lines.append(
            f"| {row['combo']} | {row['direction']} | {row['n']} | "
            f"{format_summary(row['summary'])} | {row['tail_rate_150']} | {row['score']} |"
        )

    lines.extend(["", "## Prediction Pattern Candidates", ""])
    lines.append("| Pattern | N | Normal 2h | Reverse 2h |")
    lines.append("| --- | ---: | --- | --- |")
    for row in result["prediction_combo_scan"]["patterns"][:20]:
        lines.append(
            f"| {row['pattern']} | {row['n']} | {format_summary(row['normal_2h'])} | "
            f"{format_summary(row['reverse_2h'])} |"
        )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_jsonl(NAV_EVENTS)
    ks = [5, 8, 10, 12, 15, 20]
    maps = card_maps(rows, ks)
    enriched = attach_predictions(rows, maps)
    with sqlite3.connect(DEFAULT_DB) as conn:
        result = {
            "generated_at_utc": utc_now(),
            "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
            "row_count": len(enriched),
            "k20_second_order": k20_second_order(conn, enriched),
            "scale_transition_groups": scale_transition_groups(enriched),
            "danger_count_bins": danger_count_bins(enriched),
            "tag_combo_scan": tag_combo_scan(enriched),
            "prediction_combo_scan": prediction_combo_scan(enriched),
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(result)
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
