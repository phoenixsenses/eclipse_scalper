"""S34 state-machine question tests A1-A10.

Research-only and DAT-01 style: features are computed at/ before anchor time.
Outcomes are used only for evaluation. No live files, env, order logic, or
runtime state are modified.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_freq_tests import (  # noqa: E402
    BASELINE_AVG,
    BASELINE_N,
    BASELINE_WR,
    DOW,
    OUT_JSON as FREQ_OUT_JSON,
    baseline_sync200,
    build_dataset,
    current_long_gate,
    fmt_stat,
    short_rows,
    stat,
    time_exit,
    utc_now,
    verdict,
)


OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_QUESTION_TESTS_A1_A10.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_QUESTION_TESTS_A1_A10.md"


def stat_with_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s = stat(rows)
    return {"summary": s, "decision": verdict(s)}


def recompute_base_score(row: dict[str, Any], *, n2h_thr: int = 3, vdepth_thr: float = 30.0) -> int:
    return sum(
        [
            int(int(row["n2h"]) >= int(n2h_thr)),
            int(float(row["btc4h_bps"]) < 0.0),
            int(float(row["vdepth_bps"]) >= float(vdepth_thr)),
            int(row["session"] == "US"),
            int(float(row["sync_k"]) >= 200_000.0),
        ]
    )


def current_prefilters(row: dict[str, Any]) -> bool:
    return (
        row["close_reason"] == "TIME_EXIT"
        and not row["bull"]
        and float(row["sync_k"]) < 200_000.0
        and row["session"] != "EUROPE"
        and not (row["session"] == "US" and int(row["hour"]) in {13, 14})
        and int(row["dow"]) not in {0, 2}
        and row.get("btc7d_bps") is not None
        and float(row["btc7d_bps"]) < 0.0
    )


def numeric_profile(rows: list[dict[str, Any]], features: list[str]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for feat in features:
        vals = [float(r[feat]) for r in rows if r.get(feat) is not None and math.isfinite(float(r[feat]))]
        out[feat] = {
            "n": len(vals),
            "mean": round(mean(vals), 2) if vals else None,
            "median": round(median(vals), 2) if vals else None,
            "min": round(min(vals), 2) if vals else None,
            "max": round(max(vals), 2) if vals else None,
        }
    return out


def winner_loser_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [r for r in rows if float(r["net_bps"]) > 0]
    losses = [r for r in rows if float(r["net_bps"]) <= 0]
    features = ["hour", "n2h", "vdepth_bps", "btc4h_bps", "eth4h_bps", "sync_k", "running_notional", "elapsed_since_first_sec"]
    return {
        "all": stat(rows),
        "wins_n": len(wins),
        "losses_n": len(losses),
        "winner_profile": numeric_profile(wins, features),
        "loser_profile": numeric_profile(losses, features),
        "winner_sessions": count_values(wins, "session"),
        "loser_sessions": count_values(losses, "session"),
        "winner_dow": count_values(wins, "dow_name"),
        "loser_dow": count_values(losses, "dow_name"),
    }


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[str(row.get(key))] = out.get(str(row.get(key)), 0) + 1
    return dict(sorted(out.items()))


def tests_a(rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    base = baseline_sync200(rows)
    current = current_long_gate(rows)

    # A1
    out["A1_dow_on_time_exit_sync200"] = {
        DOW[i]: stat([r for r in base if int(r["dow"]) == i]) for i in range(7)
    }

    # A2
    out["A2_sync_threshold_curve"] = {
        f"sync_lt_{label}": stat([r for r in time_exit(rows) if float(r["sync_k"]) < thr])
        for label, thr in {
            "100k": 100_000.0,
            "150k": 150_000.0,
            "200k": 200_000.0,
            "300k": 300_000.0,
            "500k": 500_000.0,
            "all": float("inf"),
        }.items()
    }

    # A3
    out["A3_btc7d_threshold_curve"] = {
        f"btc7d_lt_{label}": stat(
            [
                r
                for r in time_exit(rows)
                if float(r["sync_k"]) < 200_000.0
                and r["session"] != "EUROPE"
                and not (r["session"] == "US" and int(r["hour"]) in {13, 14})
                and int(r["dow"]) not in {0, 2}
                and int(r["long_score"]) >= 3
                and r.get("btc7d_bps") is not None
                and float(r["btc7d_bps"]) < limit
            ]
        )
        for label, limit in {
            "0": 0.0,
            "50": 50.0,
            "100": 100.0,
            "200": 200.0,
            "500": 500.0,
            "all": float("inf"),
        }.items()
    }

    # A4
    out["A4_us_13_14_block"] = {
        "exclude_13_only": stat(
            [
                r
                for r in time_exit(rows)
                if r["sync_k"] < 200_000.0
                and r["session"] != "EUROPE"
                and not (r["session"] == "US" and r["hour"] == 13)
                and r["dow"] not in {0, 2}
                and r.get("btc7d_bps") is not None
                and float(r["btc7d_bps"]) < 0
                and r["long_score"] >= 3
            ]
        ),
        "exclude_14_only": stat(
            [
                r
                for r in time_exit(rows)
                if r["sync_k"] < 200_000.0
                and r["session"] != "EUROPE"
                and not (r["session"] == "US" and r["hour"] == 14)
                and r["dow"] not in {0, 2}
                and r.get("btc7d_bps") is not None
                and float(r["btc7d_bps"]) < 0
                and r["long_score"] >= 3
            ]
        ),
        "exclude_13_14": stat(current),
        "blocked_13_14_only": stat(
            [
                r
                for r in time_exit(rows)
                if r["sync_k"] < 200_000.0
                and r["session"] == "US"
                and r["hour"] in {13, 14}
                and r["dow"] not in {0, 2}
                and r.get("btc7d_bps") is not None
                and float(r["btc7d_bps"]) < 0
                and r["long_score"] >= 3
            ]
        ),
    }

    # A5
    a5_added_base1 = [r for r in time_exit(rows) if current_prefilters(r) and int(r["base_score"]) == 1]
    a5_base2 = [r for r in time_exit(rows) if current_prefilters(r) and int(r["base_score"]) == 2]
    out["A5_score_relaxation"] = {
        "new_if_long_score_ge2_base_score1": stat(a5_added_base1),
        "base_score2_silence_score3": stat(a5_base2),
        "current_base_score_ge2": stat(current),
    }

    # A6
    n2h2_added = [
        r
        for r in time_exit(rows)
        if current_prefilters(r)
        and recompute_base_score(r, n2h_thr=2, vdepth_thr=30) + 1 >= 3
        and not (int(r["base_score"]) + 1 >= 3)
    ]
    n2h2_all = [
        r for r in time_exit(rows) if current_prefilters(r) and recompute_base_score(r, n2h_thr=2, vdepth_thr=30) + 1 >= 3
    ]
    out["A6_n2h_relax_2"] = {"added_only": stat(n2h2_added), "all_after_relax": stat(n2h2_all)}

    # A7
    out["A7_vdepth_relax"] = {}
    for thr in (20.0, 25.0, 30.0):
        all_after = [
            r for r in time_exit(rows) if current_prefilters(r) and recompute_base_score(r, n2h_thr=3, vdepth_thr=thr) + 1 >= 3
        ]
        added = [r for r in all_after if r not in current]
        out["A7_vdepth_relax"][f"vdepth_ge_{int(thr)}"] = {
            "all": stat(all_after),
            "added_only": stat(added),
        }

    # A8
    noisy = [r for r in rows if r["threshold_usd"] == 200_000.0 and r["close_reason"] == "NOISY_EARLY_EXIT"]
    noisy_eval = [
        {
            **r,
            "net_bps": float(r["noisy_hold4_net_bps"]),
            "early_exit_net_bps": r["noisy_exit_net_bps"],
            "early_exit_cost_bps": r["noisy_exit_cost_bps"],
        }
        for r in noisy
        if r.get("noisy_hold4_net_bps") is not None and r.get("noisy_exit_net_bps") is not None
    ]
    out["A8_noisy_exit_cost"] = {
        "noisy_exit_realized": stat([{**r, "net_bps": float(r["early_exit_net_bps"])} for r in noisy_eval]),
        "counterfactual_hold_4h": stat(noisy_eval),
        "cost_distribution_bps": numeric_profile(noisy_eval, ["early_exit_cost_bps"]),
    }

    # A9
    out["A9_winner_loser_profiles"] = {
        "full_pipeline_current": winner_loser_profile(current),
        "sync200_baseline": winner_loser_profile(base),
    }

    # A10
    def band(row: dict[str, Any]) -> str:
        n = float(row["running_notional"])
        if n < 300_000:
            return "200K_300K"
        if n < 500_000:
            return "300K_500K"
        if n < 1_000_000:
            return "500K_1M"
        return "1M_plus"

    out["A10_running_notional_bands"] = {
        k: stat([r for r in base if band(r) == k]) for k in ["200K_300K", "300K_500K", "500K_1M", "1M_plus"]
    }

    return out


def render(results: dict[str, Any]) -> str:
    lines = ["# S34 Question Tests A1-A10", "", f"Generated: `{results['generated_at_utc']}`", ""]
    lines.append(f"Baseline sync<200K: {fmt_stat(results['baseline'])}")
    lines.append("")
    for key, val in results["tests"].items():
        lines.append(f"## [{key}]")
        if isinstance(val, dict) and "summary" in val:
            s = val["summary"]
            lines.append(f"{fmt_stat(s)}")
            lines.append(f"Karar: {val.get('decision', verdict(s))}")
        elif isinstance(val, dict):
            for sub, s in val.items():
                if isinstance(s, dict) and "n" in s:
                    lines.append(f"- {sub}: {fmt_stat(s)}")
                elif isinstance(s, dict) and "summary" in s:
                    lines.append(f"- {sub}: {fmt_stat(s['summary'])}")
                elif isinstance(s, dict):
                    lines.append(f"- {sub}:")
                    for sub2, s2 in s.items():
                        if isinstance(s2, dict) and "n" in s2:
                            lines.append(f"  - {sub2}: {fmt_stat(s2)}")
                        elif isinstance(s2, dict) and "summary" in s2:
                            lines.append(f"  - {sub2}: {fmt_stat(s2['summary'])}")
                        elif isinstance(s2, dict):
                            compact = {
                                k: v
                                for k, v in s2.items()
                                if k in {"n", "mean", "median", "min", "max", "wins_n", "losses_n"}
                            }
                            lines.append(f"  - {sub2}: `{compact or s2}`")
                        else:
                            lines.append(f"  - {sub2}: `{s2}`")
                else:
                    lines.append(f"- {sub}: `{s}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    rows, all_rows = build_dataset()
    results = {
        "generated_at_utc": utc_now(),
        "freq_report_json": str(FREQ_OUT_JSON),
        "dataset": {
            "anchors_200k": len(rows),
            "time_exit_200k": len(time_exit(rows)),
            "baseline_sync200_n": len(baseline_sync200(rows)),
            "current_long_n": len(current_long_gate(rows)),
        },
        "baseline": stat(baseline_sync200(rows)),
        "tests": tests_a(rows, all_rows),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render(results), encoding="utf-8")
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
